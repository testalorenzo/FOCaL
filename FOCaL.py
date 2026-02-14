#
# FOCaL
#

from numpy import zeros_like, array, vstack
from copy import deepcopy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA


class FOCaL:
    def __init__(self, nuisance_mu, nuisance_pi, final_regression, fpca=False):
        """
        Initialize the FOCaL learner.

        Parameters:
        nuisance_mu: A function that estimates the expected outcome given treatment and covariates.
        nuisance_pi: A function that estimates the propensity score given treatment and covariates.
        final_regression: A function that performs the final regression step to estimate the functional outcome.
        fpca: Number of components for functional PCA. If False, no PCA is applied.
        """
        self.nuisance_mu = clone(nuisance_mu)
        self.nuisance_pi = clone(nuisance_pi)
        self.final_regression = clone(final_regression)
        self.mu1_estimates = []
        self.mu0_estimates = []
        self.pi_estimates = []
        self.FCATE = []
        self.FCATE_basis = []
        self.FATE = []
        self.fpca = fpca
        self._test_idx = []
        self.pseudo1 = []
        self.pseudo0 = []

    def fit(self, X_mu, X_pi, A, Y, n_folds=5, balanced_folds=True, random_state=12, clip_pi=True, DR=True):
        """
        Train the nuisance functions and compute pseudo-outcomes.

        Parameters:
        X_mu, X_pi: Covariates.
        A: Treatment assignments.
        Y: Functional outcomes.
        n_folds: Number of folds for cross-fitting.
        random_state: Random seed for reproducibility.
        clip_pi: Whether to clip propensity scores to avoid division by zero.
        DR: Whether to use doubly robust estimation.

        Returns:
        difference: The estimated treatment effect as a difference of pseudo-outcomes.
        """
        # reset stored results on each fit call
        self.mu1_estimates = []
        self.mu0_estimates = []
        self.pi_estimates = []
        self.FCATE = []
        self.FCATE_basis = []
        self.FATE = []
        self._test_idx = []

        if Y.ndim != 1 and Y.shape[1] == 1:
            Y = Y.ravel()

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

            if self.fpca is not False:
                pca = PCA(n_components=self.fpca)
                Y_train = pca.fit_transform(Y[train_index])
                Y_test = Y[test_index]
            else:
                Y_train, Y_test = Y[train_index], Y[test_index]

            if Y_train.ndim != 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()

            # Outcome regression (fit on treated / control separately)
            mu1_estimates = self.nuisance_mu.fit(X_mu_train[A_train == 1], Y_train[A_train == 1]).predict(X_mu_test)
            mu0_estimates = self.nuisance_mu.fit(X_mu_train[A_train == 0], Y_train[A_train == 0]).predict(X_mu_test)

            # Propensity score regression
            pi_estimates = self.nuisance_pi.fit(X_pi_train, A_train).predict_proba(X_pi_test)[:,1]

            if clip_pi:
                pi_estimates = pi_estimates.clip(1e-6, 1-1e-6)

            # Reverse PCA if needed (nuisance outputs were in PCA space)
            if self.fpca is not False:
                mu1_estimates = pca.inverse_transform(mu1_estimates.reshape(-1, self.fpca))
                mu0_estimates = pca.inverse_transform(mu0_estimates.reshape(-1, self.fpca))

            # Store fold-wise predictions and indices
            self.mu1_estimates.append(mu1_estimates)
            self.mu0_estimates.append(mu0_estimates)
            self.pi_estimates.append(pi_estimates)
            self._test_idx.append(test_index)

            if Y.ndim == 1:
                mu1_estimates = mu1_estimates[:, None]
                mu0_estimates = mu0_estimates[:, None]
                Y_test = Y_test[:, None]
            
            # Compute pseudo-outcome (doubly-robust if DR=True)
            if DR:
                pseudo_outcome1 = mu1_estimates + A_test[:,None]*(Y_test - mu1_estimates)/pi_estimates[:,None]
                pseudo_outcome0 = mu0_estimates + (1-A_test[:,None])*(Y_test - mu0_estimates)/(1-pi_estimates[:,None])
            else:
                pseudo_outcome1 = mu1_estimates
                pseudo_outcome0 = mu0_estimates

            # Store pseudo outcomes separately
            self.pseudo1.append(pseudo_outcome1)
            self.pseudo0.append(pseudo_outcome0)

            # Also store difference for CATE
            difference = pseudo_outcome1 - pseudo_outcome0
            self.FATE.append(difference)

            # Refit fPCA for final regression (unchanged)
            if self.fpca is not False:
                pca = PCA(n_components=self.fpca)
                difference_model = pca.fit_transform(difference)
            else:
                difference_model = difference

            if difference_model.ndim != 1 and difference_model.shape[1] == 1:
                difference_model = difference_model.ravel()

            FCATE_split = self.final_regression.fit(X_mu_test, difference_model)
            if self.fpca is not False:
                self.FCATE_basis.append(deepcopy(pca))
            self.FCATE.append(deepcopy(FCATE_split))

    def predict(self, X_mu):
        """
        Predict the functional outcome for new data.

        Parameters:
        X_mu: New covariates.

        Returns:
        Predicted functional outcomes.
        """
        predictions = [fcate.predict(X_mu) for fcate in self.FCATE]

        if self.fpca is not False:
            predictions = np.array([
                self.FCATE_basis[i].inverse_transform(predictions[i].reshape(-1, self.fpca))
                for i in range(len(predictions))
            ])
        else:
            predictions = np.array(predictions)

        return predictions.mean(axis=0)
    
    def predict_Y(self, A):
        """
        Retrieve per-observation pseudo outcomes under factual treatment A.

        Parameters
        ----------
        A : array-like of shape (n_samples,)
            Treatment assignments.

        Returns
        -------
        Y_pred : array of shape (n_samples, n_time)
            Predicted factual pseudo outcomes.
        """
        n = sum(len(idx) for idx in self._test_idx)
        first = np.asarray(self.pseudo1[0])
        out_dim = 1 if first.ndim == 1 else first.shape[1]
        Y_pred = np.empty((n, out_dim))

        for idxs, po1, po0 in zip(self._test_idx, self.pseudo1, self.pseudo0):
            po1, po0 = np.asarray(po1), np.asarray(po0)
            if po1.ndim == 1: po1 = po1[:, None]
            if po0.ndim == 1: po0 = po0[:, None]
            Y_pred[idxs] = A[idxs, None] * po1 + (1 - A[idxs, None]) * po0

        if out_dim == 1:
            return Y_pred.ravel()
        return Y_pred


    def get_FATE(self, average=True):
        """
        Get the estimated functional average treatment effect.

        Returns:
        FATE: The estimated functional average treatment effect.
        """
        if len(self.FATE) == 0:
            return None
        stacked = vstack(self.FATE)
        if average:
            return stacked.mean(axis=0)
        return stacked

