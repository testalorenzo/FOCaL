"""
03_diagnostics.py -- design, overlap, balance, prediction and sensitivity diagnostics for the SHARE
application (Reviewer 2, Major 6; Reviewer 1), for hypertension (main) and high cholesterol.

Reads data/processed/share.npz and results/results_<cond>_<outcome>.npz (02_fit.py). Produces
  results/diagnostics_<cond>.csv      overlap, inverse-probability weights, effective sample sizes,
                                      standardized mean differences before / after weighting
  results/sensitivity_<cond>.csv      FATE and age-cut verdicts under: fPCA 90/95/99%, 6/10/15 knots (6 = main), other nuisance learners,
                                      seeds 0/1/2, re-smoothed curves (see below), never-vs-always contrast
  results/confounding_<cond>.csv      unmeasured-confounding bias analysis for the FATE (see below)
  results/fig_diagnostics_<cond>.pdf  propensities, balance, cross-fitted outcome predictions vs observed
                                      means, residual curves, sensitivity of the FATE
Re-smoothing sensitivity (Choice 8 of NOTES.md): only smoothed curves are available locally, so the
curves are read at the eight approximate wave months (0, 30, 54, 84, 108, 132, 156, 186 after the
first interview) and re-smoothed with cubic smoothing splines under four roughness penalties
(GCV as in the original preprocessing, and lambda = 1e1, 1e3, 1e5); FOCaL is refit on each version.
Unmeasured confounding (Choice 9): (i) the share of the estimated FATE that would have to be
confounding bias for zero to enter the L2 set, 1 - radius / ||FATE||; (ii) the constant shift (in
outcome units) that brings zero into the set; (iii) an E-value at the month of the largest
standardized effect, converting the standardized difference to a risk ratio with RR = exp(0.91 d)
(VanderWeele and Ding, 2017).
Run from the share/ folder after 02_fit.py:  python3 03_diagnostics.py   (about 15 minutes)
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from focal_apps import fit_focal, bands, contrast_bands, fate_bands, style, COLORS  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ALPHA, SEED = 0.05, 0
FIT = dict(fpca=0.99, n_knots=6, knots='quantile', ridge=0.01, max_iter=500, early_stopping=True)
CONDITIONS = ['hypertension', 'high_cholesterol']
OUTCOMES = {'casp': 'Y_casp', 'mobility': 'Y_mobility'}
WAVE_MONTHS = np.array([0, 30, 54, 84, 108, 132, 156, 186])
LAMBDAS = {'GCV': None, 'lambda=1e1': 1e1, 'lambda=1e3': 1e3, 'lambda=1e5': 1e5}


def other_learners(seed):
    """Alternative nuisance learners (Choice 10): the main analysis uses gradient boosting."""
    return {
        'linear / logistic nuisances': (LinearRegression(), LogisticRegression(max_iter=2000)),
        'random-forest nuisances': (RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=4),
                                    RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=4)),
        'neural-network nuisances (previous version)': (
            make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(10, 10), activation='logistic', alpha=1e-3,
                                                         learning_rate='adaptive', max_iter=5000, random_state=seed)),
            LogisticRegression(max_iter=2000)),
    }
t0 = time.time()

d = np.load('data/processed/share.npz', allow_pickle=True)
X, cols, months = d['X'], list(d['X_cols']), d['months']
m = len(months); dt = 1.0 / (m - 1)
j_age, j_fem = cols.index('age'), cols.index('female')
PAIRS = [(2, 0), (5, 3), (4, 1)]


def smd(x, A, w=None):
    """Standardized mean difference of x between A = 1 and A = 0 (weights w, pooled unweighted SD)."""
    w = np.ones_like(x) if w is None else w
    m1 = np.average(x[A == 1], weights=w[A == 1]); m0 = np.average(x[A == 0], weights=w[A == 0])
    s = np.sqrt((x[A == 1].var() + x[A == 0].var()) / 2)
    return (m1 - m0) / s


def verdict(inf, a):
    est = inf['estimate'][a]
    l2 = np.sqrt(dt * (est ** 2).sum())
    return l2, inf['radius'][a], l2 > inf['radius'][a]


def resmooth(Y, lam):
    """Read each curve at the wave months and re-smooth it with a cubic smoothing spline."""
    out = np.empty_like(Y)
    for i in range(len(Y)):
        spl = make_smoothing_spline(WAVE_MONTHS.astype(float), Y[i, WAVE_MONTHS], lam=lam)
        out[i] = spl(months.astype(float))
    return out


style()
for cond in CONDITIONS:
    A = d[f'A_{cond}'].astype(int)
    diag_rows, sens_rows, conf_rows = [], [], []
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8))
    for r_, (outcome, key) in enumerate(OUTCOMES.items()):
        Y = d[key]
        res = np.load(f'results/results_{cond}_{outcome}.npz', allow_pickle=True)
        pi_hat, P = res['pi_hat'], res['cut_profiles']
        fate_est, fate_rad = res['fate_estimate'][0], float(res['fate_radius'][0])
        # --- overlap, weights, ESS, balance ---------------------------------------------------
        w = np.where(A == 1, 1 / pi_hat, 1 / (1 - pi_hat))
        for arm in (1, 0):
            wa = w[A == arm]
            diag_rows.append({'outcome': outcome, 'quantity': f'weights_arm{arm}', 'n': int((A == arm).sum()),
                              'mean': wa.mean(), 'max': wa.max(), 'ESS': wa.sum() ** 2 / (wa ** 2).sum()})
        diag_rows.append({'outcome': outcome, 'quantity': 'propensity', 'n': len(A), 'mean': pi_hat.mean(),
                          'q01': np.quantile(pi_hat, .01), 'q99': np.quantile(pi_hat, .99),
                          'share_lt_0.05': (pi_hat < .05).mean(), 'share_gt_0.95': (pi_hat > .95).mean(),
                          'share_clipped': ((pi_hat <= .02) | (pi_hat >= .98)).mean()})
        for k, c in enumerate(cols):
            diag_rows.append({'outcome': outcome, 'quantity': f'SMD_{c}', 'n': len(A),
                              'unweighted': smd(X[:, k], A), 'weighted': smd(X[:, k], A, w)})
        # --- cross-fitted outcome predictions vs observed means, residual curves --------------
        mu0, mu1 = res['mu_hat']
        mu_obs = np.where(A[:, None] == 1, mu1, mu0)
        E_nuis = Y - mu_obs
        diag_rows.append({'outcome': outcome, 'quantity': 'nuisance_rmse_time_avg', 'n': len(A),
                          'mean': np.sqrt((E_nuis ** 2).mean()), 'sd_Y': Y.std()})
        E3 = res['residuals']
        diag_rows.append({'outcome': outcome, 'quantity': 'third_stage_resid_rmse_time_avg', 'n': len(A),
                          'mean': np.sqrt((E3 ** 2).mean())})
        # --- unmeasured confounding ------------------------------------------------------------
        l2 = np.sqrt(dt * (fate_est ** 2).sum())
        share = max(0.0, 1 - fate_rad / l2)
        # constant shift b with ||FATE - b|| = radius: solve the quadratic in b
        mean_f = fate_est.mean(); var_f = fate_est.var()
        disc = fate_rad ** 2 * (m - 1) / m - var_f      # ||f - b||^2 = (var_f + (mean_f - b)^2) * m/(m-1)
        shift = np.sqrt(disc) if disc > 0 else np.nan     # smallest |b - mean_f| needed; nan: set never reaches 0 by a shift
        shift_needed = np.nan if np.isnan(shift) else max(0.0, abs(mean_f) - shift)
        sd_pool = np.sqrt((Y[A == 1].var(axis=0) + Y[A == 0].var(axis=0)) / 2)
        dstd = fate_est / sd_pool
        tmax = int(np.argmax(np.abs(dstd)))
        rr = np.exp(0.91 * abs(dstd[tmax]))
        evalue = rr + np.sqrt(rr * (rr - 1))
        conf_rows.append({'outcome': outcome, 'L2_norm': l2, 'L2_radius': fate_rad,
                          'share_of_effect_needed_as_bias': share, 'constant_shift_needed': shift_needed,
                          'month_max_std_effect': months[tmax], 'max_std_effect_d': dstd[tmax],
                          'RR_equivalent': rr, 'E_value': evalue})
        # --- sensitivity: fPCA, knots, seeds, re-smoothing, secondary contrast ------------------
        def record(name, inf_f, inf_c):
            l2f, rf, rej = verdict(inf_f, 0)
            row = {'outcome': outcome, 'analysis': name, 'FATE_L2': l2f, 'FATE_radius': rf, 'FATE_reject': rej,
                   'FATE_max_abs': np.abs(inf_f['estimate'][0]).max(),
                   'FATE_max_dev_from_main': np.abs(inf_f['estimate'][0] - fate_est).max()}
            for k, lab in enumerate(['men_q10', 'men_q50', 'men_q90', 'women_q10', 'women_q50', 'women_q90']):
                l2c, rc, rejc = verdict(inf_c, k)
                row[f'cut_{lab}_reject'] = rejc
            sens_rows.append(row)
            return inf_f['estimate'][0]

        fates = {'main (fPCA 99%, 6 knots, seed 0)': fate_est}
        record('main (fPCA 99%, 6 knots, seed 0)',
               {'estimate': res['fate_estimate'], 'radius': res['fate_radius']},
               {'estimate': res['cut_estimate'], 'radius': res['cut_radius']})
        for fp in (0.90, 0.95):
            f_, s_ = fit_focal(X, A, Y, seed=SEED, **dict(FIT, fpca=fp))
            fates[f'fPCA {int(100 * fp)}%'] = record(f'fPCA {int(100 * fp)}%', fate_bands(f_, ALPHA, dt, SEED),
                                                    bands(f_, s_, P, ALPHA, dt, SEED))
        for nk in (10, 15):
            f_, s_ = fit_focal(X, A, Y, seed=SEED, **dict(FIT, n_knots=nk))
            record(f'{nk} knots', fate_bands(f_, ALPHA, dt, SEED), bands(f_, s_, P, ALPHA, dt, SEED))
        for s in (1, 2):
            f_, s_ = fit_focal(X, A, Y, seed=s, **FIT)
            fates[f'seed {s}'] = record(f'seed {s}', fate_bands(f_, ALPHA, dt, s), bands(f_, s_, P, ALPHA, dt, s))
        for lab, lam in LAMBDAS.items():
            Yr = resmooth(Y, lam)
            f_, s_ = fit_focal(X, A, Yr, seed=SEED, **FIT)
            fates[f're-smoothed, {lab}'] = record(f're-smoothed curves, {lab}', fate_bands(f_, ALPHA, dt, SEED),
                                                  bands(f_, s_, P, ALPHA, dt, SEED))
        for lab, (mu_, pi_) in other_learners(SEED).items():
            f_, s_ = fit_focal(X, A, Y, seed=SEED, nuisances=(mu_, pi_), **FIT)
            fates[lab] = record(lab, fate_bands(f_, ALPHA, dt, SEED), bands(f_, s_, P, ALPHA, dt, SEED))
        record('secondary: condition at every wave vs never',
               {'estimate': res['sec_fate_estimate'], 'radius': res['sec_fate_radius']},
               {'estimate': res['sec_cut_estimate'], 'radius': res['sec_cut_radius']})
        np.savez(f'results/fate_variants_{cond}_{outcome}.npz', months=months, **{k.replace(' ', '_').replace(',', '').replace('/', '').replace('(', '').replace(')', ''): v for k, v in fates.items()})
        print(f'{cond}/{outcome}: sensitivity done at {time.time() - t0:.0f} s')

        # --- figure row ---------------------------------------------------------------------------
        INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
        ax = axes[r_, 0]
        edges = np.linspace(0, 0.7, 29)
        ax.hist(pi_hat[A == 0], bins=edges, color=COLORS['control'], alpha=.8, label=f'untreated (n = {(A == 0).sum()})')
        ax.hist(pi_hat[A == 1], bins=edges, color=COLORS['treated'], alpha=.8, label=f'treated (n = {A.sum()})')
        ax.set_xlabel(r'Cross-fitted propensity score $\hat\pi(x)$'); ax.set_ylabel('Subjects'); ax.legend(frameon=False)
        ax.set_title(f'{outcome.upper() if outcome == "casp" else "Mobility index"}: overlap', loc='left')
        ax = axes[r_, 1]
        sm_u = [abs(smd(X[:, k], A)) for k in range(len(cols))]; sm_w = [abs(smd(X[:, k], A, w)) for k in range(len(cols))]
        yy = np.arange(len(cols))
        for y0, u, v in zip(yy, sm_u, sm_w):
            ax.plot([u, v], [y0, y0], color=GRID, lw=2, zorder=1)
        ax.scatter(sm_u, yy, color='#b5b4ae', s=36, zorder=2, label='before weighting')
        ax.scatter(sm_w, yy, color=COLORS['est'], s=36, zorder=3, label='after inverse probability weighting')
        ax.set_yticks(yy); ax.set_yticklabels([c.replace('_', ' ') for c in cols]); ax.axvline(0.1, color=INK2, ls=(0, (3, 3)), lw=.8)
        ax.set_xlabel('|standardized mean difference|'); ax.legend(frameon=False, loc='upper right'); ax.set_title('Covariate balance', loc='left')
        ax.set_xlim(0, max(sm_u) * 1.15)
        ax = axes[r_, 2]
        for arm, c, lab in ((1, COLORS['treated'], 'treated'), (0, COLORS['control'], 'untreated')):
            ax.plot(months, Y[A == arm].mean(axis=0), color=c, lw=2, label=f'observed mean, {lab}')
            ax.plot(months, mu_obs[A == arm].mean(axis=0), color=c, lw=1.2, ls=(0, (4, 2)), label=f'cross-fitted regression, {lab}')
        ax.set_xlabel('Months since first interview'); ax.set_title('Outcome regressions', loc='left'); ax.legend(frameon=False, fontsize=7.5)
        ax = axes[r_, 3]
        ax.fill_between(months, fate_est - fate_rad, fate_est + fate_rad, color=COLORS['est'], alpha=.10, lw=0,
                        label='main FATE $\\pm$ radius of its global set')
        first = True
        for lab, f_ in fates.items():
            if lab.startswith('main'):
                continue
            ax.plot(months, f_, color='#8c8b85', lw=0.9, alpha=0.9, label=f'{len(fates) - 1} alternative settings' if first else None)
            first = False
        ax.plot(months, fate_est, color=COLORS['est'], lw=2.2, label='main analysis')
        ax.axhline(0, color=INK2, lw=.8, ls=(0, (3, 3))); ax.set_xlabel('Months since first interview'); ax.set_title('FATE under alternative settings', loc='left')
        ax.legend(frameon=False, fontsize=8, loc='lower left' if fate_est.mean() < 0 else 'upper left')
        for a in axes[r_]:
            a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f'results/fig_diagnostics_{cond}.pdf'); fig.savefig(f'results/fig_diagnostics_{cond}.png'); plt.close(fig)
    pd.DataFrame(diag_rows).to_csv(f'results/diagnostics_{cond}.csv', index=False)
    pd.DataFrame(sens_rows).to_csv(f'results/sensitivity_{cond}.csv', index=False)
    pd.DataFrame(conf_rows).to_csv(f'results/confounding_{cond}.csv', index=False)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(diag_rows).round(3).to_string(index=False))
    print(pd.DataFrame(sens_rows).round(3).to_string(index=False))
    print(pd.DataFrame(conf_rows).round(3).to_string(index=False))
print(f'total runtime {time.time() - t0:.0f} s')
