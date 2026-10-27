#
# FOCaL: Functional Outcome Causal Learning
#
# Cross-fitted, doubly robust meta-learner for the functional conditional
# average treatment effect (F-CATE), with L2-ball confidence sets for
# linear-smoother third stages (Proposition 4.5 of the main text and
# Proposition B.2 of the Supplementary Information).
#

import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Linear-smoother third stage
# ---------------------------------------------------------------------------
class FixedBasisLS(BaseEstimator, RegressorMixin):
    """
    Multi-output least squares on a fixed basis in x (Supplementary Eq. 4).

    The fitted value at x is sum_i w_i(x) D_i with weights that depend on the
    covariates only, which is what the inference theory requires. `basis` is an
    optional callable mapping X (n, p) to features (n, q); an intercept is
    always added. `ridge` adds a penalty on the non-intercept coefficients
    (a tuning parameter that must NOT be selected from the responses).
    """

    def __init__(self, basis=None, ridge=0.0):
        self.basis = basis
        self.ridge = ridge

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        F = X if self.basis is None else self.basis(X)
        return np.column_stack([np.ones(len(F)), F])

    def fit(self, X, D):
        Phi = self._design(X)
        D = np.asarray(D, dtype=float)
        if D.ndim == 1:
            D = D[:, None]
        pen = self.ridge * np.eye(Phi.shape[1])
        pen[0, 0] = 0.0  # do not penalize the intercept
        # H_ maps responses to coefficients: coef = H_ @ D, weights(x) = phi(x) @ H_
        self.H_ = np.linalg.solve(Phi.T @ Phi + pen, Phi.T)
        self.coef_ = self.H_ @ D
        return self

    def predict(self, X):
        return self._design(X) @ self.coef_

    def weights(self, X):
        """Smoother weights w_i(x) for the training observations, shape (n_new, n_train)."""
        return self._design(X) @ self.H_


def _sqdist(A, B):
    """Squared Euclidean distances between the rows of A and B, (len(A), len(B))."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    d2 = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


class KernelRidgeSmoother(BaseEstimator, RegressorMixin):
    """
    Kernel ridge regression with a Gaussian kernel: a linear smoother with weights
    w(x) = k(x, X)(K + ridge I)^{-1}. The bandwidth is set at fit time from the median
    pairwise distance of the covariates (times `bandwidth_scale`), never from the responses.
    """

    def __init__(self, bandwidth_scale=1.0, ridge=1.0):
        self.bandwidth_scale = bandwidth_scale
        self.ridge = ridge

    def _kernel(self, A, B):
        return np.exp(-_sqdist(A, B) / (2 * self.h_ ** 2))

    def fit(self, X, D):
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float)
        if D.ndim == 1:
            D = D[:, None]
        self.X_ = X
        sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 500), replace=False)]
        d2 = _sqdist(sub, sub)
        self.h_ = self.bandwidth_scale * np.sqrt(np.median(d2[d2 > 0]))
        K = self._kernel(X, X)
        self.H_ = np.linalg.solve(K + self.ridge * np.eye(len(X)), np.eye(len(X)))  # (K + ridge I)^{-1}
        self.coef_ = self.H_ @ D
        return self

    def predict(self, X):
        return self._kernel(np.asarray(X, dtype=float), self.X_) @ self.coef_

    def weights(self, X):
        return self._kernel(np.asarray(X, dtype=float), self.X_) @ self.H_


class KNNSmoother(BaseEstimator, RegressorMixin):
    """k-nearest-neighbour regression: a linear smoother with weights 1/k on the k nearest training points."""

    def __init__(self, k=50):
        self.k = k

    def fit(self, X, D):
        self.X_ = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float)
        self.D_ = D[:, None] if D.ndim == 1 else D
        return self

    def weights(self, X):
        X = np.asarray(X, dtype=float)
        d2 = _sqdist(X, self.X_)
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
        W = np.zeros((len(X), len(self.X_)))
        np.put_along_axis(W, idx, 1.0 / self.k, axis=1)
        return W

    def predict(self, X):
        return self.weights(X) @ self.D_


# ---------------------------------------------------------------------------
# FOCaL learner
# ---------------------------------------------------------------------------
class FOCaL:
    def __init__(self, nuisance_mu, nuisance_pi, final_regression, fpca=False, eps=0.01, fpca_third_stage=True):
        """
        Initialize the FOCaL learner.

        Parameters:
        nuisance_mu: regressor for the outcome regressions mu^(a)(x) (fit separately per arm).
        nuisance_pi: classifier with predict_proba for the propensity score pi(x).
        final_regression: regressor for the third stage. Inference (confidence sets)
            requires a linear smoother exposing a `weights(X)` method, e.g. FixedBasisLS.
        fpca: number of principal components used to compress Y for the nuisance
            regressions (and, if fpca_third_stage, for the third stage), or a fraction
            in (0, 1): the number of components that explains that share of the variance
            of the training curves (chosen fold by fold). If False, curves are used on the grid.
        fpca_third_stage: if True (and fpca is set), the third stage is fit on principal
            components of the pseudo-outcome differences; it is then NOT a linear smoother
            (the basis is chosen from the responses) and inference is disabled. If False,
            the third stage is fit on the grid and inference is available.
        eps: propensity scores are clipped to (eps, 1 - eps), matching the strong
            positivity assumption (Assumption 4.1c).
        """
        self.nuisance_mu = clone(nuisance_mu)
        self.nuisance_pi = clone(nuisance_pi)
        self.final_regression = clone(final_regression)
        self.fpca = fpca
        self.eps = eps
        self._third_pca = (fpca is not False) and fpca_third_stage
        self._reset()

    def _reset(self):
        # fold-wise quantities, in fold order
        self.mu1_estimates = []
        self.mu0_estimates = []
        self.pi_estimates = []
        self.Y_fold = []
        self.A_fold = []
        self.X_fold = []
        self._test_idx = []
        # third stage
        self.FCATE = []
        self.FCATE_basis = []
        self.pseudo1 = []
        self.pseudo0 = []
        self.FATE = []
        self.mode = None

    # ------------------------------------------------------------------
    # Stage 1 and 2: nuisances and pseudo-outcomes (cross-fitted)
    # ------------------------------------------------------------------
    def fit(self, X_mu, X_pi, A, Y, n_folds=5, balanced_folds=True, random_state=12, mode='dr', X_final=None):
        """
        Train the nuisance functions, compute the pseudo-outcomes and fit the third stage.

        Parameters:
        X_mu, X_pi: covariates used by the outcome and propensity models (n, p).
        X_final: covariates used by the third stage (default: X_mu).
        A: binary treatment (n,).
        Y: functional outcomes on a common grid (n, m).
        n_folds: number of cross-fitting folds J.
        balanced_folds: stratify folds on A.
        mode: 'dr' (doubly robust pseudo-outcomes, FOCaL), 'plugin' (mu1_hat - mu0_hat,
              a T-learner / plug-in baseline) or 'ipw' (inverse-probability weighting).
              Nuisances are shared, so `set_mode` can switch the pseudo-outcome
              definition without refitting them.
        """
        self._reset()
        X_mu = np.asarray(X_mu, dtype=float)
        X_pi = np.asarray(X_pi, dtype=float)
        A = np.asarray(A).astype(int)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        X_final = X_mu if X_final is None else np.asarray(X_final, dtype=float)
        self._X_all = X_final
        self.n_, self.m_ = Y.shape
        self.J_ = n_folds

        if balanced_folds:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            gen_kf = kf.split(X_mu, A)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            gen_kf = kf.split(X_mu)

        for train_index, test_index in gen_kf:
            X_mu_train, X_mu_test = X_mu[train_index], X_mu[test_index]
            X_pi_train, X_pi_test = X_pi[train_index], X_pi[test_index]
            A_train, A_test = A[train_index], A[test_index]
            Y_test = Y[test_index]

            # Optional compression of the training curves for the nuisance regressions
            if self.fpca is not False:
                pca = PCA(n_components=self.fpca)
                Y_train = pca.fit_transform(Y[train_index])
            else:
                Y_train = Y[train_index]

            # Outcome regressions, fit separately on treated and controls
            mu1 = clone(self.nuisance_mu).fit(X_mu_train[A_train == 1], Y_train[A_train == 1]).predict(X_mu_test)
            mu0 = clone(self.nuisance_mu).fit(X_mu_train[A_train == 0], Y_train[A_train == 0]).predict(X_mu_test)
            if self.fpca is not False:
                mu1 = pca.inverse_transform(np.asarray(mu1).reshape(-1, pca.n_components_))
                mu0 = pca.inverse_transform(np.asarray(mu0).reshape(-1, pca.n_components_))
            mu1 = np.asarray(mu1, dtype=float).reshape(len(test_index), -1)
            mu0 = np.asarray(mu0, dtype=float).reshape(len(test_index), -1)

            # Propensity score, clipped to (eps, 1 - eps)
            pi = clone(self.nuisance_pi).fit(X_pi_train, A_train).predict_proba(X_pi_test)[:, 1]
            pi = np.clip(pi, self.eps, 1 - self.eps)

            self.mu1_estimates.append(mu1)
            self.mu0_estimates.append(mu0)
            self.pi_estimates.append(pi)
            self.Y_fold.append(Y_test)
            self.A_fold.append(A_test)
            self.X_fold.append(X_final[test_index])
            self._test_idx.append(test_index)

        self.set_mode(mode)
        return self

    def _pseudo_outcomes(self, j, mode):
        """Pseudo-outcomes of fold j (Eq. 7 of the main text for mode 'dr')."""
        mu1, mu0, pi = self.mu1_estimates[j], self.mu0_estimates[j], self.pi_estimates[j]
        Y, A = self.Y_fold[j], self.A_fold[j][:, None]
        if mode == 'dr':
            p1 = mu1 + A * (Y - mu1) / pi[:, None]
            p0 = mu0 + (1 - A) * (Y - mu0) / (1 - pi[:, None])
        elif mode == 'plugin':
            p1, p0 = mu1, mu0
        elif mode == 'ipw':
            p1 = A * Y / pi[:, None]
            p0 = (1 - A) * Y / (1 - pi[:, None])
        else:
            raise ValueError("mode must be 'dr', 'plugin' or 'ipw'")
        return p1, p0

    # ------------------------------------------------------------------
    # Stage 3: final regression on each held-out fold
    # ------------------------------------------------------------------
    def set_mode(self, mode='dr'):
        """(Re)compute pseudo-outcomes with the given mode and refit the third stage."""
        self.mode = mode
        self.FCATE, self.FCATE_basis, self.pseudo1, self.pseudo0, self.FATE = [], [], [], [], []
        for j in range(self.J_):
            p1, p0 = self._pseudo_outcomes(j, mode)
            diff = p1 - p0
            self.pseudo1.append(p1)
            self.pseudo0.append(p0)
            self.FATE.append(diff)
            if self._third_pca:
                pca = PCA(n_components=self.fpca)
                target = pca.fit_transform(diff)
                self.FCATE_basis.append(deepcopy(pca))
            else:
                target = diff
            self.FCATE.append(clone(self.final_regression).fit(self.X_fold[j], target))
        return self

    # ------------------------------------------------------------------
    # Helpers to put fold-wise quantities back in the original row order
    # ------------------------------------------------------------------
    def _row_order(self, fold_arrays):
        first = np.asarray(fold_arrays[0])
        out = np.empty((self.n_,) + first.shape[1:], dtype=float)
        for idx, arr in zip(self._test_idx, fold_arrays):
            out[idx] = np.asarray(arr)
        return out

    def get_pseudo_diff(self):
        """Pseudo-outcome differences gamma1_hat - gamma0_hat, (n, m), original row order."""
        return self._row_order(self.FATE)

    def get_mu_hat(self):
        """Cross-fitted outcome regressions (mu0_hat, mu1_hat), each (n, m), original row order."""
        return self._row_order(self.mu0_estimates), self._row_order(self.mu1_estimates)

    def get_pi_hat(self):
        """Cross-fitted propensity scores (n,), original row order."""
        return self._row_order(self.pi_estimates)

    def get_FATE(self, average=True, trim=0.0):
        """
        Functional average treatment effect: the sample mean of the pseudo-outcome
        differences (the sample mean is a linear smoother with weights 1/n).
        `trim` drops the fraction of curves with the largest L2 norm before averaging.
        With average=False, the per-observation differences are returned in row order.
        """
        D = self.get_pseudo_diff()
        if not average:
            return D
        if trim > 0:
            norms = np.linalg.norm(D, axis=1)
            D = D[norms <= np.quantile(norms, 1 - trim)]
        return D.mean(axis=0)

    # ------------------------------------------------------------------
    # Prediction: cross-fitted average of the fold-specific estimators (Eq. 9)
    # ------------------------------------------------------------------
    def predict(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        preds = []
        for j, model in enumerate(self.FCATE):
            p = np.asarray(model.predict(X_new))
            if self._third_pca:
                p = self.FCATE_basis[j].inverse_transform(p.reshape(-1, self.FCATE_basis[j].n_components_))
            preds.append(p.reshape(len(X_new), -1))
        return np.mean(preds, axis=0)

    # ------------------------------------------------------------------
    # Inference for linear-smoother third stages
    # ------------------------------------------------------------------
    def weights(self, X_new):
        """
        Pooled cross-fitted smoother weights omega_i(x) = omega_i^[j](x) / J
        (Supplementary Eq. 5), shape (n_new, n), original row order.
        """
        if self._third_pca or not hasattr(self.FCATE[0], 'weights'):
            raise ValueError("inference requires a grid third stage (fpca=False or fpca_third_stage=False) "
                             "with a `weights` method")
        X_new = np.asarray(X_new, dtype=float)
        W = np.zeros((len(X_new), self.n_))
        for j, model in enumerate(self.FCATE):
            W[:, self._test_idx[j]] = model.weights(X_new) / self.J_
        return W

    def residuals(self):
        """Third-stage residual curves e_i = (gamma1_hat - gamma0_hat)(D_i) - theta_hat(X_i), (n, m)."""
        return self.get_pseudo_diff() - self.predict(self._X_all)

    @staticmethod
    def _inference_from(theta_hat, W, E, alpha, dt, n_sim, random_state):
        """
        L2-ball radius and pointwise/sup-type calibrations from weights W (k, n)
        and residual curves E (n, m).

        The covariance operator of theta_hat(x) is estimated by
            sum_i omega_i(x)^2 e_i e_i^T = r_n(x)^2 Sigma_hat(x)      (Supplementary Eq. 11),
        whose eigen-decomposition is obtained from the SVD of the rows omega_i(x) e_i.
        The L2 norm on the grid is the quadrature norm ||f||^2 = dt * sum_t f(t)^2.
        """
        rng = np.random.default_rng(random_state)
        k, n = W.shape
        m = E.shape[1]
        out = {
            'estimate': theta_hat,
            'r_n': np.sqrt((W ** 2).sum(axis=1)),              # standard-error scale
            'radius': np.zeros(k),                                # L2-ball radius c_hat * r_n
            'se': np.zeros((k, m)),                               # pointwise standard errors
            'q_pointwise': np.zeros(k),                           # z_{1-alpha/2}
            'q_maxt': np.zeros(k),                                # quantile of max_t |G(t)|/se(t)
            'q_bonf': np.zeros(k),                                # Bonferroni over the grid
        }
        from scipy.stats import norm as _norm
        z = _norm.ppf(1 - alpha / 2)
        zb = _norm.ppf(1 - alpha / (2 * m))
        for a in range(k):
            Ew = W[a][:, None] * E                                 # (n, m) rows omega_i e_i
            # covariance of theta_hat(x) on the grid = Ew^T Ew = V diag(s^2) V^T
            _, s, Vt = np.linalg.svd(Ew, full_matrices=False)
            xi = rng.standard_normal((len(s), n_sim))
            G = (Vt.T * s) @ xi                                    # (m, n_sim) Gaussian draws
            norms = np.sqrt(dt * (G ** 2).sum(axis=0))
            se = np.sqrt((Ew ** 2).sum(axis=0))
            out['radius'][a] = np.quantile(norms, 1 - alpha)
            out['se'][a] = se
            out['q_pointwise'][a] = z
            out['q_bonf'][a] = zb
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.abs(G) / np.where(se > 0, se, np.inf)[:, None]
            out['q_maxt'][a] = np.quantile(ratio.max(axis=0), 1 - alpha)
        return out

    def confidence_set(self, X_new, alpha=0.05, dt=None, n_sim=20000, random_state=0):
        """
        L2-ball confidence set for theta*(x) at each row of X_new (Eq. 16 of the main text):
            C_alpha(x) = { f : ||f - theta_hat(x)|| <= radius(x) }.
        Also returns pointwise standard errors and the max-t and Bonferroni multipliers,
        which calibrate sup-type bands over the observation grid for comparison only.
        """
        if dt is None:
            dt = 1.0 / (self.m_ - 1)
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return self._inference_from(self.predict(X_new), self.weights(X_new), self.residuals(),
                                    alpha, dt, n_sim, random_state)

    def fate_confidence_set(self, alpha=0.05, dt=None, n_sim=20000, random_state=0):
        """L2-ball confidence set for the FATE: weights 1/n, residuals around the sample mean."""
        if dt is None:
            dt = 1.0 / (self.m_ - 1)
        D = self.get_pseudo_diff()
        fate = D.mean(axis=0, keepdims=True)
        W = np.full((1, self.n_), 1.0 / self.n_)
        return self._inference_from(fate, W, D - fate, alpha, dt, n_sim, random_state)
