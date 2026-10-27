"""
Shared helpers for the three FOCaL applications (nhanes, encode, lcl).

Every application follows the same recipe:
  1. fit_focal(): cross-fitted FOCaL with gradient-boosting nuisances and an
     additive-spline least-squares third stage (a linear smoother, so that the
     smoother weights are available for inference);
  2. bands(): at a set of covariate profiles, the FOCaL estimate with
       - the constant-width simultaneous band (Gaussian multiplier bootstrap,
         Remark 4.4 of the sup-norm note),
       - the studentized simultaneous band (Proposition 4.5 of the note),
       - the L2-ball radius (Proposition 4.5 of the manuscript),
       - pointwise standard errors;
  3. fate_bands(): the same for the functional average treatment effect;
  4. plotting helpers with one fixed style.

The FOCaL class itself lives in ../FOCaL.py (repository root) and is imported from there.
"""

import os
import sys
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # FOCaL.py at the repository root
from FOCaL import FOCaL, FixedBasisLS  # noqa: E402


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def default_nuisances(seed=0, max_iter=300, early_stopping='auto'):
    """
    Gradient boosting for the outcome regressions (one booster per fPCA score) and the propensity.
    early_stopping=True (with max_iter=500) gives the settings of the simulation study: the number
    of trees is chosen on a held-out 10% of each training fold.
    """
    boost = dict(max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                 early_stopping=early_stopping, random_state=seed)
    mu = MultiOutputRegressor(HistGradientBoostingRegressor(**boost))
    pi = HistGradientBoostingClassifier(**boost)
    return mu, pi


class AdditiveSplines:
    """
    Additive cubic B-spline basis in each (standardized) covariate; knots fixed a priori.
    knots='uniform': n_knots equispaced knots over the range of each covariate.
    knots='quantile': knots at the distinct quantiles of each covariate (ties, e.g. the 30% of
    genes with TPM = 0, merge knots); binary covariates enter linearly.
    """

    def __init__(self, n_knots=6, knots='uniform'):
        self.n_knots = n_knots
        self.knots = knots

    def fit(self, Xs):
        if self.knots == 'uniform':
            self.t_ = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False).fit(Xs)
            return self
        self.parts_ = []
        for j in range(Xs.shape[1]):
            x = Xs[:, [j]]
            if len(np.unique(x)) <= 2:
                self.parts_.append(None)
                continue
            k = np.unique(np.quantile(x, np.linspace(0, 1, self.n_knots)))
            self.parts_.append(SplineTransformer(knots=k[:, None], degree=3, include_bias=False).fit(x))
        return self

    def __call__(self, Xs):
        if self.knots == 'uniform':
            return self.t_.transform(Xs)
        return np.column_stack([Xs[:, [j]] if t is None else t.transform(Xs[:, [j]])
                                for j, t in enumerate(self.parts_)])


def fit_focal(X, A, Y, seed=0, n_folds=5, fpca=10, n_knots=6, knots='uniform', ridge=1.0, eps=0.02,
              max_iter=300, early_stopping='auto', nuisances=None):
    """
    Standardize X, fit FOCaL with cross-fitting, return (focal, scaler).
    The third stage is least squares on an additive spline basis (FixedBasisLS),
    fit on the grid so that smoother weights exist. Propensities are clipped to
    (eps, 1 - eps), the strong-positivity assumption of the paper. fpca may be a
    variance fraction (e.g. 0.99), see FOCaL. nuisances: optional (mu, pi) pair of sklearn
    estimators replacing the default gradient boosting (used for learner sensitivity analyses).
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    basis = AdditiveSplines(n_knots, knots).fit(Xs)
    mu, pi = default_nuisances(seed, max_iter, early_stopping) if nuisances is None else nuisances
    focal = FOCaL(nuisance_mu=mu, nuisance_pi=pi,
                  final_regression=FixedBasisLS(basis=basis, ridge=ridge),
                  fpca=fpca, fpca_third_stage=False, eps=eps)
    focal.fit(Xs, Xs, A, Y, n_folds=n_folds, balanced_folds=True, random_state=seed, mode='dr')
    return focal, scaler


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _calibrate(theta_hat, W, E, alpha, dt, n_sim=20000, seed=0):
    """All calibrations from smoother weights W (k, n) and residual curves E (n, m)."""
    rng = np.random.default_rng(seed)
    k, n = W.shape
    m = E.shape[1]
    out = {'estimate': theta_hat, 'se': np.zeros((k, m)), 'half_sup': np.zeros(k),
           'q_stud': np.zeros(k), 'radius': np.zeros(k), 'z': norm.ppf(1 - alpha / 2), 'alpha': alpha}
    for a in range(k):
        Ew = W[a][:, None] * E                     # rows omega_i(x) e_i
        G = rng.standard_normal((n_sim, n)) @ Ew   # multiplier draws, covariance sum_i omega_i^2 e_i e_i^T
        se = np.sqrt((Ew ** 2).sum(axis=0))
        out['se'][a] = se
        out['half_sup'][a] = np.quantile(np.abs(G).max(axis=1), 1 - alpha)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(G) / np.where(se > 0, se, np.inf)[None, :]
        out['q_stud'][a] = np.quantile(ratio.max(axis=1), 1 - alpha)
        out['radius'][a] = np.quantile(np.sqrt(dt * (G ** 2).sum(axis=1)), 1 - alpha)
    return out


def bands(focal, scaler, X_profiles, alpha=0.05, dt=None, seed=0):
    """FOCaL estimate and bands at the rows of X_profiles (raw covariate scale)."""
    if dt is None:
        dt = 1.0 / (focal.m_ - 1)
    Xs = scaler.transform(np.atleast_2d(X_profiles))
    return _calibrate(focal.predict(Xs), focal.weights(Xs), focal.residuals(), alpha, dt, seed=seed)


def contrast_bands(focal, scaler, X_profiles, pairs, alpha=0.05, dt=None, seed=0):
    """
    Bands for differences theta_hat(x_k) - theta_hat(x_l) between profiles, (k, l) in pairs.
    A difference of linear smoothers is a linear smoother with weights omega_i(x_k) - omega_i(x_l),
    so the same calibrations apply (residuals as for the single profiles).
    """
    if dt is None:
        dt = 1.0 / (focal.m_ - 1)
    Xs = scaler.transform(np.atleast_2d(X_profiles))
    est, W = focal.predict(Xs), focal.weights(Xs)
    Wc = np.vstack([W[k] - W[l] for k, l in pairs])
    Ec = np.vstack([est[k] - est[l] for k, l in pairs])
    return _calibrate(Ec, Wc, focal.residuals(), alpha, dt, seed=seed)


def fate_bands(focal, alpha=0.05, dt=None, seed=0):
    """FATE (sample mean of pseudo-outcome differences, weights 1/n) and its bands."""
    if dt is None:
        dt = 1.0 / (focal.m_ - 1)
    D = focal.get_pseudo_diff()
    fate = D.mean(axis=0, keepdims=True)
    W = np.full((1, focal.n_), 1.0 / focal.n_)
    return _calibrate(fate, W, D - fate, alpha, dt, seed=seed)


def ball_envelope(est, w, E, alpha=0.05, dt=None, n_sim=20000, seed=0):
    """
    L2-ball radius for the curve est = sum_i w_i (response_i) with residual curves E (n, m),
    and the pointwise envelope (min, max) of the multiplier draws est + G that lie inside the ball.
    Returns (radius, lower envelope, upper envelope, number of draws inside).
    """
    if dt is None:
        dt = 1.0 / (E.shape[1] - 1)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_sim, len(w))) @ (w[:, None] * E)
    norms = np.sqrt(dt * (G ** 2).sum(axis=1))
    radius = np.quantile(norms, 1 - alpha)
    inside = est + G[norms <= radius]
    return radius, inside.min(axis=0), inside.max(axis=0), len(inside)


def overlap_report(pi_hat, A, eps=0.02):
    """Text summary of the estimated propensities, to be pasted in NOTES.md."""
    q = np.quantile(pi_hat, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    lines = [f"propensity quantiles 1/5/25/50/75/95/99%: {np.round(q, 3).tolist()}",
             f"share with pi_hat < 0.05: {(pi_hat < 0.05).mean():.3f}, > 0.95: {(pi_hat > 0.95).mean():.3f}",
             f"share at the clipping bounds ({eps}, {1 - eps}): {((pi_hat <= eps) | (pi_hat >= 1 - eps)).mean():.3f}",
             f"treated share: {A.mean():.3f} (n = {len(A)})"]
    return "\n".join(lines)


def significance(est, half):
    """Index ranges where zero is outside est +/- half (half may be a vector)."""
    mask = (est - half > 0) | (est + half < 0)
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def profiles_along(X, cols, var, quantiles=(0.1, 0.5, 0.9), center='median'):
    """Covariate profiles that move one variable over its quantiles, others at the median/mean."""
    base = np.median(X, axis=0) if center == 'median' else X.mean(axis=0)
    j = list(cols).index(var)
    vals = np.quantile(X[:, j], quantiles)
    P = np.tile(base, (len(vals), 1))
    P[:, j] = vals
    return P, vals


def surface_along(focal, scaler, X, cols, var, n_grid=40, center='median', lo=0.02, hi=0.98):
    """F-CATE surface over (t, var) with var between its lo and hi quantiles, others at the median/mean."""
    base = np.median(X, axis=0) if center == 'median' else X.mean(axis=0)
    j = list(cols).index(var)
    vals = np.linspace(*np.quantile(X[:, j], [lo, hi]), n_grid)
    G = np.tile(base, (n_grid, 1))
    G[:, j] = vals
    return vals, focal.predict(scaler.transform(G))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COLORS = {'est': '#1f4e9c', 'stud': '#9ecae1', 'sup': '#e6550d', 'zero': '#222222',
          'treated': '#d62728', 'control': '#1f77b4', 'cuts': ['#1b9e77', '#7570b3', '#d95f02']}


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 300})


def plot_band(ax, t, inf, a, label=None, color=None, show_sup=True, show_stud=True):
    """Estimate with the studentized band (fill) and the constant-width band (dashed)."""
    est, se = inf['estimate'][a], inf['se'][a]
    c = color or COLORS['est']
    if show_stud:
        ax.fill_between(t, est - inf['q_stud'][a] * se, est + inf['q_stud'][a] * se,
                        color=c, alpha=0.18, lw=0)
    if show_sup:
        ax.plot(t, est - inf['half_sup'][a], color=c, lw=0.9, ls='--')
        ax.plot(t, est + inf['half_sup'][a], color=c, lw=0.9, ls='--')
    ax.plot(t, est, color=c, lw=2, label=label)
    ax.axhline(0, color=COLORS['zero'], lw=0.7)


def band_legend(ax, alpha):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lvl = int(round(100 * (1 - alpha)))
    handles = [Patch(facecolor='#999999', alpha=0.3, label=f'{lvl}% studentized simultaneous band'),
               Line2D([], [], color='#999999', ls='--', lw=0.9, label=f'{lvl}% constant-width simultaneous band')]
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + handles, l + [x.get_label() for x in handles], frameon=False)


def plot_surface(ax, t, vals, S, var_label, t_label, cmap='RdBu_r'):
    """Heatmap of the F-CATE surface, symmetric diverging scale around zero."""
    lim = np.abs(S).max()
    im = ax.pcolormesh(t, vals, S, cmap=cmap, vmin=-lim, vmax=lim, shading='auto')
    ax.set_xlabel(t_label)
    ax.set_ylabel(var_label)
    return im
