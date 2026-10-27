#
# Simulation study for FOCaL (Methods, Section "Simulation study")
#
# Base design: n = 5000, grid of m = 100 points on [0, 1], four covariates,
# four nuisance-specification scenarios. For each scenario the script reports
#   - ARMSE of the final F-CATE estimator, int_x ||theta_hat(x) - theta*(x)|| dx,
#     approximated on an independent evaluation sample;
#   - L2 error of the FATE;
#   - empirical coverage and radius of the L2-ball confidence sets at fixed
#     covariate profiles and for the FATE (plus pointwise / max-t / Bonferroni
#     grid bands, reported for comparison only);
#   - nuisance diagnostics and runtime;
# for FOCaL (doubly robust; functional nuisances on fPCA scores, grid third stage,
# L2 confidence sets), two scalar DR-learner baselines applied to the discretized
# outcome (pointwise: separate nuisances and third stage at each grid point, with
# pointwise / max-t / Bonferroni grid bands; basis-wise: scalar DR-learner on each fPCA
# score with Bonferroni over scores), the plug-in / T-learner, IPW, and a non-robust
# function-on-scalar regression with treatment-by-covariate interactions.
# Also reported per trial: FATE from the final regression (average of theta_hat(x))
# next to the pseudo-outcome mean, propensity calibration slope, share of clipped
# propensities and number of non-converged nuisance fits.
# One-factor-at-a-time variations of the design are run with `--variations`.
#
# Note on the pointwise baseline: for outcome learners that are linear in the responses
# (OLS, ridge, splines) fitting one regression per grid point and projecting on the fPCA
# scores gives the same fit as regressing the scores and reconstructing, so FOCaL and the
# pointwise DR-learner coincide (up to the discarded components) in the parametric design;
# they differ in inference (L2 set vs grid bands) and, for learners that are nonlinear in
# the response (variation 'ML nuisances': gradient boosting, 10 boosters per arm and fold
# for FOCaL against 100 for the pointwise learner), in the fit itself.
#
# ARMSE(theta_hat, theta*) = E_x ||theta_hat(x) - theta*(x)||, the expectation over the
# covariate distribution (N(0, I)) approximated on an independent sample of n_eval
# points, the L2 norm over [0, 1] by the trapezoid-free quadrature dt * sum_t f(t)^2,
# dt = 1 / (m - 1).
#
# Usage:
#   python simulations.py --n_trials 1000              # base design
#   python simulations.py --variations --n_trials 200  # design variations
#
# Outputs: results/*.csv and results/sim_summary.pdf; logs in batch_logs/.
#

import argparse
import logging
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # trials run in parallel processes; one thread each avoids oversubscription
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm as normal
from sklearn.gaussian_process.kernels import Matern
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import SplineTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # FOCaL.py at the repository root
from FOCaL import FOCaL, FixedBasisLS  # noqa: E402


# ===========================================================================
# Configuration
# ===========================================================================
BASE = dict(
    n=5000,             # sample size
    n_time=100,         # grid size m
    p_noise=0,          # extra covariates that enter neither mu nor pi (p = 4 + p_noise)
    overlap=1.0,        # scale of the propensity linear predictor (larger = weaker overlap)
    nu=5.5,             # Matern smoothness of coefficient and noise curves
    length_scale=0.25,  # Matern length scale
    alpha_coef=2.0,     # Matern amplitude of beta_0, ..., beta_4
    alpha_effect=10.0,  # Matern amplitude of beta_5 (treatment effect)
    alpha_noise=10.0,   # Matern amplitude of the noise curves
    # means of the coefficient curves (reported in the manuscript)
    coef_means=(210.0, 27.4, 13.7, 13.7, 13.7, 10.0),
    effect='shifted',     # effect = delta(a) w(t) beta_5(t) h(X_1); 'shifted': w = 1, h = X_1 + 1 (FATE = beta_5);
                          # 'centered': w = 1, h = X_1 (FATE = 0); 'localized': w = Gaussian bump at t = 0.4, h = X_1 + 1;
                          # 'signchange': w = cos(2 pi t), h = X_1 + 1; 'nonlinear': w = 1, h = sin(pi X_1) + 1
    fpca_nuisance=10,     # fPCA scores used by FOCaL's functional nuisance regressions (False: one regression per grid point)
    nuisance='linear',    # 'linear': OLS / logistic (misspecified through Z); 'boosting': gradient boosting on the same inputs
    boost_iter=500, boost_lr=0.05, boost_leaves=15, boost_min_leaf=20, boost_l2=0.0, boost_early=True,
                          # boosting settings: early stopping on a held-out 10% of the fold (sklearn default) picks the
                          # number of trees; without it (50 trees, rate 0.1) the propensity is overfit (calibration slope 0.76)
    third_stage='linear', # basis of the least-squares third stage in x: 'linear' (1, x) or 'splines' (additive cubic B-splines)
    spline_knots=15,      # knots per covariate of the spline third stage (placed at covariate quantiles, fixed a priori)
    spline_ridge=0.01,    # ridge penalty of the spline third stage (fixed a priori, never selected from the responses);
                          # a flexible third stage trades error for negligible smoothing bias, which the bands require
    n_basis_dr=10,        # fPCA scores of the basis-wise scalar DR-learner
    separate_arms=False,  # treated arm with its own coefficient curves (effect = difference)
    heavy_tails=False,    # multivariate t_3 noise curves instead of Gaussian
    sparse=False,         # sparse, noisy observation + reconstruction on a Fourier basis
    n_obs_sparse=8,       # number of observed points per curve when sparse
    sigma_obs=2.0,        # measurement-error sd when sparse
    eps=0.01,             # clipping of the propensity score
    n_folds=5,
    alpha=0.05,           # confidence level 1 - alpha
    beta_seed=0,          # coefficient curves are drawn ONCE with this seed (fixed target)
    n_eval=1000,          # evaluation points for the ARMSE integral over x
)

SCENARIOS = {
    'Correct specification': ('X', 'X'),
    'Regression misspecified': ('Z', 'X'),
    'Propensity misspecified': ('X', 'Z'),
    'Fully misspecified': ('Z', 'Z'),
}

METHODS = ['FOCaL', 'Pointwise DR', 'Basis-wise DR', 'Plug-in', 'IPW', 'FoS-interaction']

# X_1 quantiles of the profiles at which coverage is assessed (other covariates at 0)
PROFILE_QUANTILES = (0.1, 0.5, 0.9)

VARIATIONS = {
    'n=500': dict(n=500),
    'n=1000': dict(n=1000),
    'n=2000': dict(n=2000),
    'overlap x0.5': dict(overlap=0.5),
    'overlap x2': dict(overlap=2.0),
    'p=10': dict(p_noise=6),
    'nu=1.5': dict(nu=1.5),
    'separate arms': dict(separate_arms=True),
    'heavy tails': dict(heavy_tails=True),
    'sparse noisy': dict(sparse=True),
    'centered effect': dict(effect='centered'),
    'localized effect': dict(effect='localized'),
    'sign-changing effect': dict(effect='signchange'),
    'nonlinear modifier': dict(effect='nonlinear'),
    'nonlinear modifier + splines': dict(effect='nonlinear', third_stage='splines'),
    'nonlinear modifier + splines (n=2000)': dict(effect='nonlinear', third_stage='splines', n=2000),
    'ML nuisances': dict(nuisance='boosting', n=2000, effect='nonlinear', third_stage='splines'),
}


# ===========================================================================
# Data generating process (Methods, Eqs. 17-21)
# ===========================================================================
def matern_cov(t_grid, amplitude, length_scale, nu):
    """Matern covariance matrix on the grid (Eq. 18)."""
    return amplitude ** 2 * Matern(length_scale=length_scale, nu=nu)(t_grid.reshape(-1, 1))


def draw_coefficients(cfg):
    """
    Coefficient curves beta_0, ..., beta_5 (and a second set for the treated arm
    if separate_arms): Gaussian processes with Matern covariance around fixed means.
    Drawn once per configuration so that the F-CATE target is fixed across replications.
    """
    rng = np.random.default_rng(cfg['beta_seed'])
    t = np.linspace(0, 1, cfg['n_time'])
    cov_coef = matern_cov(t, cfg['alpha_coef'], cfg['length_scale'], cfg['nu'])
    cov_eff = matern_cov(t, cfg['alpha_effect'], cfg['length_scale'], cfg['nu'])
    means = cfg['coef_means']
    betas = np.array([means[j] + rng.multivariate_normal(np.zeros(cfg['n_time']), cov_coef)
                      for j in range(5)])                      # beta_0..beta_4, (5, m)
    beta5 = means[5] + rng.multivariate_normal(np.zeros(cfg['n_time']), cov_eff)
    betas_treated = None
    if cfg['separate_arms']:
        # treated arm: independent coefficient curves around the same means
        betas_treated = np.array([means[j] + rng.multivariate_normal(np.zeros(cfg['n_time']), cov_coef)
                                  for j in range(5)])
    return dict(t=t, betas=betas, beta5=beta5, betas_treated=betas_treated)


def true_mu(X, coef, cfg, a):
    """mu^(a)(x) on the grid for the rows of X, (n, m)."""
    X4 = X[:, :4]
    if cfg['separate_arms']:
        B = coef['betas_treated'] if a == 1 else coef['betas']
        return B[0] + X4 @ B[1:]
    mu = coef['betas'][0] + X4 @ coef['betas'][1:]
    if a == 1:
        mu = mu + np.outer(effect_modifier(X4[:, 0], cfg), effect_curve(coef, cfg))
    return mu


def effect_curve(coef, cfg):
    """w(t) beta_5(t): the time profile of the treatment effect."""
    t = coef['t']
    w = {'localized': np.exp(-((t - 0.4) / 0.08) ** 2), 'signchange': np.cos(2 * np.pi * t)}.get(cfg['effect'], np.ones_like(t))
    return w * coef['beta5']


def effect_modifier(x1, cfg):
    """h(x_1): how the effect scales with the first covariate."""
    if cfg['effect'] == 'centered':
        return x1
    if cfg['effect'] == 'nonlinear':
        return np.sin(np.pi * x1) + 1.0
    return x1 + 1.0


def true_fcate(X, coef, cfg):
    """theta*(x) = mu^(1)(x) - mu^(0)(x)."""
    return true_mu(X, coef, cfg, 1) - true_mu(X, coef, cfg, 0)


def true_propensity(X, cfg):
    lin = -X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] - 0.1 * X[:, 3]
    return 1.0 / (1.0 + np.exp(-cfg['overlap'] * lin))


def misspecified_covariates(X):
    """Non-linear transformations Z_1, ..., Z_4 (Eq. 21); extra columns are passed through."""
    x = X[:, :4]
    Z = np.column_stack([
        np.exp(x[:, 0] / 2),
        x[:, 1] / (1 + np.exp(x[:, 0])) + 10,
        (x[:, 0] * x[:, 2] / 25 + 0.6) ** 3,
        (x[:, 1] + x[:, 3] + 20) ** 2,
    ])
    return np.column_stack([Z, X[:, 4:]]) if X.shape[1] > 4 else Z


def sparse_reconstruction(Y, t, cfg, rng):
    """
    Observation layer: each curve is observed at n_obs_sparse random grid points with
    Gaussian measurement error and reconstructed by ridge least squares on a Fourier
    basis (1, sin, cos up to frequency 3). Mimics the SHARE preprocessing.
    """
    n, m = Y.shape
    K = 3
    basis = np.column_stack([np.ones(m)] + [f(2 * np.pi * k * t) for k in range(1, K + 1) for f in (np.sin, np.cos)])
    Y_rec = np.empty_like(Y)
    for i in range(n):
        idx = rng.choice(m, cfg['n_obs_sparse'], replace=False)
        y_obs = Y[i, idx] + rng.normal(0, cfg['sigma_obs'], len(idx))
        B = basis[idx]
        coef = np.linalg.solve(B.T @ B + 1e-2 * np.eye(B.shape[1]), B.T @ y_obs)
        Y_rec[i] = basis @ coef
    return Y_rec


def reconstruction_truth(cfg, coef, profiles, rng, n_mc=20000):
    """
    Sparse design: the estimand identified from reconstructed curves. The reconstruction is linear
    in the observations and its sampling points are independent of (X, A, noise), so
    E[rec(Y) | X, A = a] = E_idx[rec(mu_a(X))]; the F-CATE on reconstructed curves is the
    Monte Carlo mean of the reconstructed noiseless potential-outcome curves. Returned for the
    covariate profiles and for the FATE (an average over n_mc draws of X).
    """
    t = coef['t']
    def rec_effect(X):
        return (sparse_reconstruction(true_mu(X, coef, cfg, 1), t, cfg, rng)
                - sparse_reconstruction(true_mu(X, coef, cfg, 0), t, cfg, rng))
    prof = np.vstack([rec_effect(np.tile(x, (n_mc, 1))).mean(axis=0) for x in profiles])
    fate = rec_effect(rng.normal(size=(n_mc, profiles.shape[1]))).mean(axis=0)
    return dict(prof=prof, fate=fate)


def generate_data(cfg, coef, seed):
    """One replication: returns dict with X, Z, A, Y, true quantities."""
    rng = np.random.default_rng(seed)
    n, m = cfg['n'], cfg['n_time']
    X = rng.normal(size=(n, 4 + cfg['p_noise']))
    pi = true_propensity(X, cfg)
    A = rng.binomial(1, pi)

    mu0, mu1 = true_mu(X, coef, cfg, 0), true_mu(X, coef, cfg, 1)
    cov_noise = matern_cov(coef['t'], cfg['alpha_noise'], cfg['length_scale'], cfg['nu'])
    # independent noise curves for the two potential outcomes, one per subject
    eps0 = rng.multivariate_normal(np.zeros(m), cov_noise, size=n)
    eps1 = rng.multivariate_normal(np.zeros(m), cov_noise, size=n)
    if cfg['heavy_tails']:
        # multivariate t_3 scale mixture: heavy-tailed curves with the same correlation
        eps0 *= np.sqrt(3.0 / rng.chisquare(3, size=n))[:, None]
        eps1 *= np.sqrt(3.0 / rng.chisquare(3, size=n))[:, None]
    Y0, Y1 = mu0 + eps0, mu1 + eps1
    Y = A[:, None] * Y1 + (1 - A[:, None]) * Y0
    if cfg['sparse']:
        Y = sparse_reconstruction(Y, coef['t'], cfg, rng)

    return dict(X=X, Z=misspecified_covariates(X), A=A, Y=Y, pi=pi, mu0=mu0, mu1=mu1)


# ===========================================================================
# Estimators
# ===========================================================================
def l2_norm(F, dt):
    """Quadrature L2 norm of each row of F."""
    F = np.atleast_2d(F)
    return np.sqrt(dt * (F ** 2).sum(axis=1))


def fit_focal(data, scenario, cfg, seed, fpca, Y=None):
    """
    FOCaL with linear/logistic nuisances and least squares on (1, x) as third stage.
    fpca = K: the outcome regressions are fit on K fPCA scores of the training curves
    (functional nuisances); fpca = False: one regression per grid point, which is the
    scalar DR-learner applied pointwise (same third stage at each point).
    Returns the fitted object (mode 'dr') and the number of non-converged nuisance fits;
    other pseudo-outcome modes are obtained with set_mode without refitting.
    """
    X_mu = data['X'] if scenario[0] == 'X' else data['Z']
    X_pi = data['X'] if scenario[1] == 'X' else data['Z']
    if cfg['nuisance'] == 'boosting':
        boost = dict(max_iter=cfg['boost_iter'], learning_rate=cfg['boost_lr'], max_leaf_nodes=cfg['boost_leaves'],
                     min_samples_leaf=cfg['boost_min_leaf'], l2_regularization=cfg['boost_l2'],
                     early_stopping=cfg['boost_early'], random_state=seed)
        mu = MultiOutputRegressor(HistGradientBoostingRegressor(**boost))
        pi = HistGradientBoostingClassifier(**boost)
    else:
        mu, pi = LinearRegression(), LogisticRegression(max_iter=1000)
    basis = None
    if cfg['third_stage'] == 'splines':
        spl = SplineTransformer(n_knots=cfg['spline_knots'], degree=3, knots='quantile', include_bias=False).fit(data['X'])
        basis = spl.transform
    # a small ridge keeps the spline design well conditioned (not selected from the responses)
    focal = FOCaL(mu, pi, FixedBasisLS(basis=basis, ridge=cfg['spline_ridge'] if basis is not None else 0.0),
                  fpca=fpca, fpca_third_stage=False, eps=cfg['eps'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        # the third stage always uses the original covariates X (the F-CATE is a function of x)
        focal.fit(X_mu, X_pi, data['A'], data['Y'] if Y is None else Y, n_folds=cfg['n_folds'],
                  random_state=seed, X_final=data['X'])
    n_nonconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return focal, n_nonconverged


def fit_basiswise(data, scenario, cfg, seed):
    """
    Basis-wise scalar DR-learner: the curves are projected on K fPCA components (fit on the
    observed curves, an unsupervised step), a scalar DR-learner is run on each score, and
    the F-CATE curve is reconstructed from the K score effects. Returns the fitted object,
    the components (K, m), and the count of non-converged nuisance fits.
    """
    pca = PCA(n_components=cfg['n_basis_dr']).fit(data['Y'])
    scores = pca.transform(data['Y'])
    focal, n_nonconverged = fit_focal(data, scenario, cfg, seed, fpca=False, Y=scores)
    return focal, pca.components_, n_nonconverged


def fos_interaction(data, scenario):
    """
    Non-robust competitor: function-on-scalar least squares of Y on (1, W, A, A*W),
    with W = X or Z according to the outcome-model specification. The F-CATE estimate
    at x is the coefficient of A plus the interaction terms evaluated at x.
    """
    W = data['X'] if scenario[0] == 'X' else data['Z']
    A = data['A'][:, None]
    design = np.column_stack([np.ones(len(W)), W, A, A * W])
    coef = np.linalg.lstsq(design, data['Y'], rcond=None)[0]   # (1 + 2p + 1, m)
    p = W.shape[1]
    b_A, b_AW = coef[1 + p], coef[2 + p:]

    def predict(X_new):
        W_new = X_new if scenario[0] == 'X' else misspecified_covariates(X_new)
        return b_A + W_new @ b_AW
    return predict


# ===========================================================================
# One replication
# ===========================================================================
def grid_band_rows(inf, a, truth, dt, truth_rec=None):
    """
    Coverage of the L2 ball and of the pointwise / max-t / Bonferroni grid bands for curve a of inf.
    truth_rec (sparse design only): the reconstruction-level estimand, i.e. the effect on the
    reconstructed curves, which is what every method sees; cover_L2_rec is its L2-ball coverage.
    """
    err = inf['estimate'][a] - truth
    band_err = np.abs(err) / np.where(inf['se'][a] > 0, inf['se'][a], np.inf)
    rec = {} if truth_rec is None else dict(
        cover_L2_rec=float(l2_norm(inf['estimate'][a] - truth_rec, dt)[0] <= inf['radius'][a]),
        bias_rec=l2_norm(truth_rec - truth, dt)[0])
    return dict(
        cover_L2=float(l2_norm(err, dt)[0] <= inf['radius'][a]), radius=inf['radius'][a], r_n=inf['r_n'][a], **rec,
        cover_pointwise_all_t=float(band_err.max() <= inf['q_pointwise'][a]),
        cover_pointwise_mean_t=float(np.mean(band_err <= inf['q_pointwise'][a])),
        cover_maxt=float(band_err.max() <= inf['q_maxt'][a]),
        cover_bonf=float(band_err.max() <= inf['q_bonf'][a]),
        width_pointwise=np.mean(2 * inf['q_pointwise'][a] * inf['se'][a]),
        width_maxt=np.mean(2 * inf['q_maxt'][a] * inf['se'][a]),
        width_bonf=np.mean(2 * inf['q_bonf'][a] * inf['se'][a]))


def run_trial(trial, cfg, coef, scenario_name, X_eval, theta_eval, profiles, theta_prof, fate_true, dt, rec=None):
    scenario = SCENARIOS[scenario_name]
    data = generate_data(cfg, coef, seed=trial)
    rows, cov_rows = [], []
    targets = [f'X1 q{int(100 * qq)}' for qq in PROFILE_QUANTILES] + ['FATE']

    # FOCaL (functional nuisances), the pointwise scalar DR-learner and the basis-wise one
    t0 = time.perf_counter()
    focal, nonconv = fit_focal(data, scenario, cfg, seed=trial, fpca=cfg['fpca_nuisance'])
    time_focal = time.perf_counter() - t0
    t0 = time.perf_counter()
    focal_pw, nonconv_pw = fit_focal(data, scenario, cfg, seed=trial, fpca=False)
    time_pw = time.perf_counter() - t0
    t0 = time.perf_counter()
    focal_b, components, nonconv_b = fit_basiswise(data, scenario, cfg, seed=trial)
    time_b = time.perf_counter() - t0

    # nuisance diagnostics (cross-fitted predictions versus the truth), for the functional nuisances
    mu0_hat, mu1_hat = focal.get_mu_hat()
    pi_hat = focal.get_pi_hat()
    logit = np.log(pi_hat / (1 - pi_hat))
    calib = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], data['A'])
    nuis = dict(
        mu_rmse=np.mean(np.r_[l2_norm(mu0_hat - data['mu0'], dt), l2_norm(mu1_hat - data['mu1'], dt)]),
        pi_mae=np.mean(np.abs(pi_hat - data['pi'])),
        pi_brier=np.mean((pi_hat - data['A']) ** 2),
        pi_calib_slope=calib.coef_[0, 0],                          # 1 = calibrated propensities
        clip_share=np.mean((pi_hat <= cfg['eps']) | (pi_hat >= 1 - cfg['eps'])),
        n_nonconverged=nonconv + nonconv_pw + nonconv_b,
    )

    for method in METHODS:
        t_inf = 0.0
        if method == 'FoS-interaction':
            t1 = time.perf_counter()
            predict = fos_interaction(data, scenario)
            fit_time = time.perf_counter() - t1
            fate_hat = predict(data['X']).mean(axis=0)
        elif method == 'Pointwise DR':
            predict, fate_hat, fit_time = focal_pw.predict, focal_pw.get_FATE(), time_pw
            t1 = time.perf_counter()
            inf_prof = focal_pw.confidence_set(profiles, alpha=cfg['alpha'], dt=dt, random_state=trial)
            inf_fate = focal_pw.fate_confidence_set(alpha=cfg['alpha'], dt=dt, random_state=trial)
            t_inf = time.perf_counter() - t1
            for q, name in enumerate(targets):
                inf, truth, a = (inf_fate, fate_true, 0) if name == 'FATE' else (inf_prof, theta_prof[q], q)
                truth_rec = None if rec is None else (rec['fate'] if name == 'FATE' else rec['prof'][q])
                cov_rows.append(dict(trial=trial, scenario=scenario_name, method=method, profile=name,
                                     **grid_band_rows(inf, a, truth, dt, truth_rec)))
        elif method == 'Basis-wise DR':
            fit_time = time_b
            predict = lambda X_new: focal_b.predict(X_new) @ components        # reconstruct the curve
            fate_hat = focal_b.get_FATE() @ components
            t1 = time.perf_counter()
            inf_prof = focal_b.confidence_set(profiles, alpha=cfg['alpha'], dt=1.0, random_state=trial)
            inf_fate = focal_b.fate_confidence_set(alpha=cfg['alpha'], dt=1.0, random_state=trial)
            t_inf = time.perf_counter() - t1
            for q, name in enumerate(targets):
                # Bonferroni over the K score-wise intervals: all K true scores covered at once
                inf, truth, a = (inf_fate, fate_true, 0) if name == 'FATE' else (inf_prof, theta_prof[q], q)
                score_err = np.abs(inf['estimate'][a] - truth @ components.T)
                cov_rows.append(dict(trial=trial, scenario=scenario_name, method=method, profile=name,
                                     cover_bonf=float((score_err <= inf['q_bonf'][a] * inf['se'][a]).all()),
                                     width_bonf=np.mean(2 * inf['q_bonf'][a] * inf['se'][a])))
        else:
            focal.set_mode({'FOCaL': 'dr', 'Plug-in': 'plugin', 'IPW': 'ipw'}[method])
            predict, fate_hat, fit_time = focal.predict, focal.get_FATE(), time_focal
            if method == 'FOCaL':
                # inference is reported for the doubly robust pseudo-outcomes only (Supplementary Eq. 11)
                t1 = time.perf_counter()
                inf_prof = focal.confidence_set(profiles, alpha=cfg['alpha'], dt=dt, random_state=trial)
                inf_fate = focal.fate_confidence_set(alpha=cfg['alpha'], dt=dt, random_state=trial)
                t_inf = time.perf_counter() - t1
                for q, name in enumerate(targets):
                    inf, truth, a = (inf_fate, fate_true, 0) if name == 'FATE' else (inf_prof, theta_prof[q], q)
                    truth_rec = None if rec is None else (rec['fate'] if name == 'FATE' else rec['prof'][q])
                    cov_rows.append(dict(trial=trial, scenario=scenario_name, method=method, profile=name,
                                         **grid_band_rows(inf, a, truth, dt, truth_rec)))
        theta_hat_eval = predict(X_eval)
        rows.append(dict(trial=trial, scenario=scenario_name, method=method,
                         ARMSE=np.mean(l2_norm(theta_hat_eval - theta_eval, dt)),
                         FATE_error=l2_norm(fate_hat - fate_true, dt)[0],                   # pseudo-outcome mean
                         FATE_reg_error=l2_norm(theta_hat_eval.mean(axis=0) - fate_true, dt)[0],  # mean of theta_hat(x)
                         runtime_fit=fit_time, runtime_inference=t_inf,
                         n_nan=int(np.isnan(theta_hat_eval).sum()), **nuis))
    return rows, cov_rows


def run_design(cfg, n_trials, label, log):
    """Run all scenarios for one design; returns (metrics, coverage) data frames."""
    coef = draw_coefficients(cfg)
    dt = 1.0 / (cfg['n_time'] - 1)
    rng = np.random.default_rng(10 ** 6 + cfg['beta_seed'])
    X_eval = rng.normal(size=(cfg['n_eval'], 4 + cfg['p_noise']))
    theta_eval = true_fcate(X_eval, coef, cfg)
    profiles = np.zeros((len(PROFILE_QUANTILES), 4 + cfg['p_noise']))
    profiles[:, 0] = normal.ppf(PROFILE_QUANTILES)
    theta_prof = true_fcate(profiles, coef, cfg)
    fate_true = true_fcate(rng.normal(size=(200000, 4 + cfg['p_noise'])), coef, cfg).mean(axis=0)
    rec = reconstruction_truth(cfg, coef, profiles, rng) if cfg['sparse'] else None

    rows, cov_rows = [], []
    for scenario_name in SCENARIOS:
        log.info(f'[{label}] scenario: {scenario_name}')
        results = Parallel(n_jobs=cfg['n_jobs'])(
            delayed(run_trial)(trial, cfg, coef, scenario_name, X_eval, theta_eval, profiles, theta_prof, fate_true, dt, rec)
            for trial in tqdm(range(n_trials), desc=f'{label} | {scenario_name}'))
        for r, c in results:
            rows += r
            cov_rows += c
    metrics, coverage = pd.DataFrame(rows), pd.DataFrame(cov_rows)
    metrics.insert(0, 'design', label)
    coverage.insert(0, 'design', label)
    return metrics, coverage


# ===========================================================================
# Summaries and figure
# ===========================================================================
def summarize(metrics, coverage, log):
    m = metrics.groupby(['design', 'scenario', 'method'])[['ARMSE', 'FATE_error', 'FATE_reg_error', 'runtime_fit', 'runtime_inference']].agg(['mean', 'std'])
    c = coverage.groupby(['design', 'scenario', 'method', 'profile'])[
        ['cover_L2', 'radius', 'cover_pointwise_all_t', 'cover_pointwise_mean_t', 'cover_maxt', 'cover_bonf',
         'width_pointwise', 'width_maxt', 'width_bonf']].mean()
    nuis = metrics[metrics.method == 'FOCaL'].groupby(['design', 'scenario'])[
        ['mu_rmse', 'pi_mae', 'pi_brier', 'pi_calib_slope', 'clip_share', 'n_nonconverged', 'n_nan']].mean()
    for name, table in [('ARMSE / FATE error / runtime', m), ('Coverage', c), ('Nuisance diagnostics', nuis)]:
        log.info(f'\n{name}\n{table.round(3).to_string()}')
    return m, c, nuis


def make_figure(metrics, coverage, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    base_m = metrics[metrics.design == 'base']
    base_c = coverage[(coverage.design == 'base') & (coverage.method == 'FOCaL')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=base_m, x='scenario', y='ARMSE', hue='method', showfliers=False, ax=axes[0])
    axes[0].set_title('F-CATE error (ARMSE)')
    axes[0].tick_params(axis='x', rotation=20)
    sns.boxplot(data=base_m, x='scenario', y='FATE_error', hue='method', showfliers=False, ax=axes[1])
    axes[1].set_title('FATE error')
    axes[1].tick_params(axis='x', rotation=20)
    cov_tab = base_c.groupby(['scenario', 'profile'])['cover_L2'].mean().reset_index()
    sns.barplot(data=cov_tab, x='scenario', y='cover_L2', hue='profile', ax=axes[2])
    axes[2].axhline(1 - BASE['alpha'], color='k', ls='--')
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Coverage of the L2 confidence set (FOCaL)')
    axes[2].tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(path)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--variations', action='store_true', help='run the one-factor design variations')
    parser.add_argument('--only', type=str, default=None, help='comma-separated subset of variation names')
    parser.add_argument('--n', type=int, default=None, help='override the base sample size (for quick tests)')
    parser.add_argument('--out', type=str, default='results', help='output directory (use a scratch directory for tests)')
    parser.add_argument('--n_jobs', type=int, default=max(1, os.cpu_count() - 2), help='parallel processes over trials')
    args = parser.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    os.makedirs('batch_logs', exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(f'batch_logs/simulations_{stamp}.log'), logging.StreamHandler()])
    log = logging.getLogger('simulations')

    cfg = dict(BASE, n_jobs=args.n_jobs)
    if args.n is not None:
        cfg['n'] = args.n

    if not args.variations:
        metrics, coverage = run_design(cfg, args.n_trials, 'base', log)
        suffix = 'base'
    else:
        names = list(VARIATIONS) if args.only is None else [s.strip() for s in args.only.split(',')]
        ms, cs = [], []
        for name in names:
            cfg_v = dict(cfg, **VARIATIONS[name])
            m_v, c_v = run_design(cfg_v, args.n_trials, name, log)
            ms.append(m_v)
            cs.append(c_v)
        metrics, coverage = pd.concat(ms, ignore_index=True), pd.concat(cs, ignore_index=True)
        suffix = 'variations'

    metrics.to_csv(f'{out}/metrics_{suffix}.csv', index=False)
    coverage.to_csv(f'{out}/coverage_{suffix}.csv', index=False)
    m, c, nuis = summarize(metrics, coverage, log)
    m.to_csv(f'{out}/summary_metrics_{suffix}.csv')
    c.to_csv(f'{out}/summary_coverage_{suffix}.csv')
    nuis.to_csv(f'{out}/summary_nuisance_{suffix}.csv')
    if not args.variations:
        make_figure(metrics, coverage, f'{out}/sim_summary.pdf')
    log.info('done')
