"""
02_fit.py -- FOCaL on the SHARE data: chronic condition at baseline -> CASP and mobility trajectories.

Everything goes through the shared helper ../focal_apps.py, with the settings of the ENCODE
application and of the simulation study (NOTES.md, Choices 3-5): fPCA compression keeping 99% of
the variance of the training curves, early-stopped gradient-boosting nuisances, additive cubic
B-splines with knots at covariate quantiles and ridge 0.01 as third stage (a linear smoother, so
the L2 confidence sets of Proposition 4.5 are available), 5 folds balanced on the treatment,
propensities clipped to (0.02, 0.98).

For each condition (hypertension: main text; high cholesterol: supplement) and each outcome
(CASP, mobility index):
  1. main fit, treatment = status at baseline, all 1518 subjects;
  2. overlap report (results/overlap_<cond>_<outcome>.txt);
  3. FATE with its L2 set; plug-in FATE; FATE refits with seeds 1 and 2;
  4. F-CATE cuts at age quantiles 10/50/90 for men and women (education, children, vaccination,
     smoking at their medians); contrasts old - young within gender and women - men at median age;
     age x time surfaces for men and women;
  5. secondary analysis: the contrast of the previous version of the paper (condition at every
     wave vs never), on its 996 / 1060 subjects, FATE and cuts only.
Results: results/results_<cond>_<outcome>.npz and results/summary.csv (one row per target).
Run from the share/ folder:  python3 02_fit.py   (about 10 minutes)
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, '..')
from focal_apps import fit_focal, bands, contrast_bands, fate_bands, overlap_report  # noqa: E402

ALPHA, SEED = 0.05, 0
FIT = dict(fpca=0.99, n_knots=6, knots='quantile', ridge=0.01, max_iter=500, early_stopping=True)
CONDITIONS = ['hypertension', 'high_cholesterol']
OUTCOMES = {'casp': 'Y_casp', 'mobility': 'Y_mobility'}
AGE_Q = (0.1, 0.5, 0.9)
t0 = time.time()

d = np.load('data/processed/share.npz', allow_pickle=True)
X, cols, months = d['X'], list(d['X_cols']), d['months']
m = len(months)
dt = 1.0 / (m - 1)                                   # L2 norm on [0, 1] with the grid quadrature of the simulations
j_age, j_fem = cols.index('age'), cols.index('female')


def age_gender_profiles(X):
    """Six profiles: age at its 10/50/90% quantiles for men (female = 0) and women (female = 1)."""
    base = np.median(X, axis=0)
    ages = np.quantile(X[:, j_age], AGE_Q)
    P = []
    for fem in (0, 1):
        for a in ages:
            p = base.copy(); p[j_age] = a; p[j_fem] = fem
            P.append(p)
    return np.array(P), ages


def age_surface(focal, scaler, X, fem, n_grid=40):
    """F-CATE over (month, age) at the given gender, other covariates at their medians."""
    base = np.median(X, axis=0)
    ages = np.linspace(*np.quantile(X[:, j_age], [0.02, 0.98]), n_grid)
    G = np.tile(base, (n_grid, 1)); G[:, j_age] = ages; G[:, j_fem] = fem
    return ages, focal.predict(scaler.transform(G))


rows = []


def summarize(tag, cond, outcome, inf, a, n_used):
    est, se = inf['estimate'][a], inf['se'][a]
    l2 = np.sqrt(dt * (est ** 2).sum())
    rows.append({'condition': cond, 'outcome': outcome, 'target': tag, 'n': n_used,
                 'max_abs_estimate': np.abs(est).max(), 'argmax_month': months[np.argmax(np.abs(est))],
                 'estimate_month0': est[0], 'estimate_month191': est[-1],
                 'L2_norm': l2, 'L2_radius': inf['radius'][a], 'L2_reject': l2 > inf['radius'][a],
                 'max_se': se.max()})


# profile contrasts (indices into the six profiles: 0-2 men by age, 3-5 women by age)
PAIRS = [(2, 0), (5, 3), (4, 1)]
PAIR_NAMES = ['men_old_minus_young', 'women_old_minus_young', 'women_minus_men_median_age']

for cond in CONDITIONS:
    A = d[f'A_{cond}'].astype(int)
    sec_mask = d[f'A_{cond}_always'] | d[f'A_{cond}_never']   # secondary contrast of the previous version
    for outcome, key in OUTCOMES.items():
        Y = d[key]
        print(f'\n=== {cond} / {outcome}: n = {len(A)}, treated = {A.sum()} ===')
        focal, scaler = fit_focal(X, A, Y, seed=SEED, **FIT)
        n_pc = [PCA(FIT['fpca']).fit(Y[np.setdiff1d(np.arange(len(A)), idx)]).n_components_ for idx in focal._test_idx]
        pi_hat = focal.get_pi_hat()
        report = overlap_report(pi_hat, A) + f'\nfPCA components per training fold (99% variance): {n_pc}'
        print(report)
        with open(f'results/overlap_{cond}_{outcome}.txt', 'w') as f:
            f.write(report + '\n')

        fate = fate_bands(focal, alpha=ALPHA, dt=dt, seed=SEED)
        P, ages = age_gender_profiles(X)
        cuts = bands(focal, scaler, P, alpha=ALPHA, dt=dt, seed=SEED)
        contr = contrast_bands(focal, scaler, P, PAIRS, alpha=ALPHA, dt=dt, seed=SEED)
        ages_grid, S_men = age_surface(focal, scaler, X, 0)
        _, S_women = age_surface(focal, scaler, X, 1)
        focal.set_mode('plugin'); fate_plugin = focal.get_FATE(); focal.set_mode('dr')
        fate_seeds = {}
        for s in (1, 2):
            f_s, _ = fit_focal(X, A, Y, seed=s, **FIT)
            fate_seeds[s] = fate_bands(f_s, alpha=ALPHA, dt=dt, seed=s)

        # secondary: condition at every wave vs never (uses future status; previous version of the paper)
        A_sec = d[f'A_{cond}_always'][sec_mask].astype(int)
        f_sec, sc_sec = fit_focal(X[sec_mask], A_sec, Y[sec_mask], seed=SEED, **FIT)
        fate_sec = fate_bands(f_sec, alpha=ALPHA, dt=dt, seed=SEED)
        cuts_sec = bands(f_sec, sc_sec, P, alpha=ALPHA, dt=dt, seed=SEED)
        pi_sec = f_sec.get_pi_hat()

        summarize('FATE_dr', cond, outcome, fate, 0, len(A))
        for s in (1, 2):
            summarize(f'FATE_dr_seed{s}', cond, outcome, fate_seeds[s], 0, len(A))
        rows.append({'condition': cond, 'outcome': outcome, 'target': 'FATE_plugin', 'n': len(A),
                     'max_abs_estimate': np.abs(fate_plugin).max(), 'argmax_month': months[np.argmax(np.abs(fate_plugin))],
                     'estimate_month0': fate_plugin[0], 'estimate_month191': fate_plugin[-1],
                     'L2_norm': np.sqrt(dt * (fate_plugin ** 2).sum()),
                     'max_se': np.abs(fate_plugin - fate['estimate'][0]).max()})   # here: max |plug-in - DR|
        for k in range(6):
            summarize(f'FCATE_{"men" if k < 3 else "women"}_age_q{int(100 * AGE_Q[k % 3])}', cond, outcome, cuts, k, len(A))
        for k, name in enumerate(PAIR_NAMES):
            summarize(f'contrast_{name}', cond, outcome, contr, k, len(A))
        summarize('FATE_dr_secondary_always_vs_never', cond, outcome, fate_sec, 0, int(sec_mask.sum()))
        for k in range(6):
            summarize(f'FCATE_secondary_{"men" if k < 3 else "women"}_age_q{int(100 * AGE_Q[k % 3])}', cond, outcome,
                      cuts_sec, k, int(sec_mask.sum()))

        np.savez(f'results/results_{cond}_{outcome}.npz',
                 months=months, alpha=ALPHA, dt=dt, n=len(A), treated=A.sum(), n_pc=n_pc, X_cols=np.array(cols),
                 fate_estimate=fate['estimate'], fate_se=fate['se'], fate_radius=fate['radius'],
                 fate_plugin=fate_plugin, fate_seed1=fate_seeds[1]['estimate'], fate_seed2=fate_seeds[2]['estimate'],
                 cut_profiles=P, cut_ages=ages, cut_estimate=cuts['estimate'], cut_se=cuts['se'], cut_radius=cuts['radius'],
                 contrast_pairs=np.array(PAIRS), contrast_names=np.array(PAIR_NAMES),
                 contrast_estimate=contr['estimate'], contrast_se=contr['se'], contrast_radius=contr['radius'],
                 surface_ages=ages_grid, surface_men=S_men, surface_women=S_women,
                 pi_hat=pi_hat, pseudo_diff=focal.get_pseudo_diff(), residuals=focal.residuals(),
                 cut_weights=focal.weights(scaler.transform(P)), mu_hat=np.stack(focal.get_mu_hat()),
                 sec_mask=sec_mask, sec_pi_hat=pi_sec, sec_fate_estimate=fate_sec['estimate'], sec_fate_radius=fate_sec['radius'],
                 sec_cut_estimate=cuts_sec['estimate'], sec_cut_radius=cuts_sec['radius'],
                 sec_pseudo_diff=f_sec.get_pseudo_diff(), sec_residuals=f_sec.residuals(),
                 sec_cut_weights=f_sec.weights(sc_sec.transform(P)))
        print(f'  done at {time.time() - t0:.0f} s; FATE max |est| {np.abs(fate["estimate"][0]).max():.3f}, '
              f'L2 {rows[-1 - 6 - 1 - 3 - 6 - 1 - 2]["L2_norm"]:.3f}')

summary = pd.DataFrame(rows)
summary.to_csv('results/summary.csv', index=False)
pd.set_option('display.width', 250)
print(summary.round(3).to_string(index=False))
print(f'total runtime {time.time() - t0:.0f} s')
