"""
03_fit.py -- FOCaL on the ENCODE data: CpG island at the promoter -> H3K4me3 profile around the TSS.

Everything goes through the shared helper ../focal_apps.py:
  1. fit_focal(X, A, Y, seed=0, **FIT): cross-fitted FOCaL with the settings of the simulation study
     (Choices 7-8 of NOTES.md): fPCA compression keeping 99% of the variance of the training curves,
     early-stopped boosting nuisances, additive cubic splines with 15 knots at covariate quantiles
     and ridge 0.01 as third stage.
  2. Overlap: propensity quantiles (results/overlap.txt) and histogram (results/fig_overlap).
     A pre-registered rule said to refit on 0.05 <= pi_hat <= 0.95 if more than 5% of the
     propensities were clipped; none was, so no such refit exists.
  3. Covariates: the four pre-treatment covariates fixed by the DNA sequence and the annotation
     (gene length, number of transcripts, distance to the nearest other gene start, strand).
     Expression is NOT in X: it is partly a consequence of the island (and of H3K4me3), so it is
     not a pre-treatment covariate (Choice 6 and Caveats of NOTES.md). The target is the total
     effect of the island.
  4. FATE with simultaneous bands; F-CATE cuts at the 10/50/90% quantiles of the distance to the
     nearest other gene start, other covariates at their medians; the surface over (position,
     distance); differences between the cuts (q10 - q50, q10 - q90, q50 - q90), whose weights are
     differences of smoother weights; the same q90 - q10 contrast along gene length and along
     the number of transcripts (heterogeneity checked on every continuous covariate); the plug-in
     FATE; FATE refits with seeds 1 and 2 (stability); the FATE with expression added to X
     (effect at fixed activity, secondary).
  5. Results to results/full_results.npz (curves, bands, weights and residuals for 05_plot_l2.py)
     and results/summary.csv (one row per target).

Run from the encode/ folder:  python3 03_fit.py
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from focal_apps import (fit_focal, bands, contrast_bands, fate_bands, overlap_report,  # noqa: E402
                        significance, profiles_along, surface_along, style, COLORS)

ALPHA, SEED, VAR = 0.05, 0, 'log10_dist_tss'
EXPR = 'log1p_TPM'                               # expression: not pre-treatment, used only in the secondary fit
OTHER_VARS = ['log10_length', 'n_transcripts']   # other continuous covariates, heterogeneity checked
# nuisance and third-stage settings, as in the simulation study (code/simulations.py, BASE)
FIT = dict(fpca=0.99, n_knots=15, knots='quantile', ridge=0.01, max_iter=500, early_stopping=True)
PAIRS = [(0, 1), (0, 2), (1, 2)]                 # cut contrasts q10 - q50, q10 - q90, q50 - q90
t0 = time.time()

d = np.load('data/processed/encode.npz', allow_pickle=True)
X_all, A, Y, cols_all, positions = d['X'], d['A'], d['Y'], list(d['X_cols']), d['positions']
keep = [c != EXPR for c in cols_all]
X, cols = X_all[:, keep], [c for c, k in zip(cols_all, keep) if k]   # pre-treatment covariates
dt = 1.0 / Y.shape[1]                            # 80 bins of equal width on [0, 1]: L2 norm of the step function
print(f'n = {len(A)}, m = {Y.shape[1]}, treated share = {A.mean():.3f}')

# ---------------------------------------------------------------------------
# Main fit and overlap diagnostics
# ---------------------------------------------------------------------------
focal, scaler = fit_focal(X, A, Y, seed=SEED, **FIT)
print(f'main fit done in {time.time() - t0:.0f} s')
n_pc = [PCA(FIT['fpca']).fit(Y[idx_tr]).n_components_ for idx_tr in
        (np.setdiff1d(np.arange(len(A)), idx) for idx in focal._test_idx)]
print(f'fPCA components per training fold (99% of the variance): {n_pc}')
pi_hat = focal.get_pi_hat()
report = overlap_report(pi_hat, A)
print(report)
with open('results/overlap.txt', 'w') as f:
    f.write(report + '\n')

style()
fig, ax = plt.subplots(figsize=(5.2, 3.4))
edges = np.linspace(0, 1, 41)
ax.hist(pi_hat[A == 0], bins=edges, color=COLORS['control'], alpha=0.6, label='no CpG island (A = 0)')
ax.hist(pi_hat[A == 1], bins=edges, color=COLORS['treated'], alpha=0.6, label='CpG island (A = 1)')
ax.set_xlabel(r'cross-fitted propensity $\hat\pi(x)$'); ax.set_ylabel('genes'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('results/fig_overlap.pdf'); fig.savefig('results/fig_overlap.png'); plt.close(fig)

# ---------------------------------------------------------------------------
# Targets: FATE, cuts along the distance to the nearest gene start, surface, contrasts, plug-in, seeds
# ---------------------------------------------------------------------------
fate = fate_bands(focal, alpha=ALPHA, dt=dt, seed=SEED)
P, qvals = profiles_along(X, cols, VAR, quantiles=(0.1, 0.5, 0.9), center='median')
cuts = bands(focal, scaler, P, alpha=ALPHA, dt=dt, seed=SEED)
contr = contrast_bands(focal, scaler, P, PAIRS, alpha=ALPHA, dt=dt, seed=SEED)
svals, S = surface_along(focal, scaler, X, cols, VAR, n_grid=40, center='median')
other = {}                                       # q90 - q10 contrast along the other continuous covariates
for v in OTHER_VARS:
    P_v, q_v = profiles_along(X, cols, v, quantiles=(0.1, 0.5, 0.9), center='median')
    other[v] = (q_v, contrast_bands(focal, scaler, P_v, [(2, 0)], alpha=ALPHA, dt=dt, seed=SEED))

focal.set_mode('plugin')
fate_plugin = focal.get_FATE()
focal.set_mode('dr')

fate_seeds = {}
for s in (1, 2):
    f_s, _ = fit_focal(X, A, Y, seed=s, **FIT)
    fate_seeds[s] = fate_bands(f_s, alpha=ALPHA, dt=dt, seed=s)
    print(f'seed {s} refit done, max |FATE| = {np.abs(fate_seeds[s]["estimate"][0]).max():.3f}')

# Secondary: expression added to X, i.e. the effect at fixed activity (Caveats in NOTES.md)
f_fa, _ = fit_focal(X_all, A, Y, seed=SEED, **FIT)
fate_fixact = fate_bands(f_fa, alpha=ALPHA, dt=dt, seed=SEED)
print(f'FATE at fixed activity (expression in X): max {fate_fixact["estimate"][0].max():.3f}')

# ---------------------------------------------------------------------------
# Summary table and saved curves
# ---------------------------------------------------------------------------
rows = []


def summarize(name, inf, a):
    """One summary row for curve a of the band dictionary inf."""
    est, se = inf['estimate'][a], inf['se'][a]
    l2 = np.sqrt(dt * (est ** 2).sum())
    maxt = np.max(np.abs(est) / np.where(se > 0, se, np.inf))
    as_bp = lambda runs: '; '.join(f'{positions[i]:.0f}..{positions[j]:.0f}' for i, j in runs)
    rows.append({'target': name, 'max_abs_estimate': np.abs(est).max(),
                 'argmax_position_bp': positions[np.argmax(np.abs(est))],
                 'L2_norm': l2, 'L2_radius': inf['radius'][a], 'L2_reject': l2 > inf['radius'][a],
                 'half_width_const': inf['half_sup'][a],
                 'max_t_stat': maxt, 'q_stud': inf['q_stud'][a], 'maxt_reject': maxt > inf['q_stud'][a],
                 'max_se': se.max(),
                 'sig_ranges_stud_bp': as_bp(significance(est, inf['q_stud'][a] * se)),
                 'sig_ranges_const_bp': as_bp(significance(est, inf['half_sup'][a]))})


summarize('FATE_dr', fate, 0)
for k, q in enumerate((10, 50, 90)):
    summarize(f'FCATE_cut_dist_q{q}', cuts, k)
for k, (a, b) in enumerate(PAIRS):
    summarize(f'FCATE_contrast_dist_q{(10, 50, 90)[a]}_minus_q{(10, 50, 90)[b]}', contr, k)
for v in OTHER_VARS:
    summarize(f'FCATE_contrast_{v}_q90_minus_q10', other[v][1], 0)
for s in (1, 2):
    summarize(f'FATE_dr_seed{s}', fate_seeds[s], 0)
summarize('FATE_dr_fixed_activity', fate_fixact, 0)
rows.append({'target': 'FATE_plugin', 'max_abs_estimate': np.abs(fate_plugin).max(),
             'argmax_position_bp': positions[np.argmax(np.abs(fate_plugin))],
             'L2_norm': np.sqrt(dt * (fate_plugin ** 2).sum()),
             'max_se': np.abs(fate_plugin - fate['estimate'][0]).max()})   # here: max |plug-in - DR|
summary = pd.DataFrame(rows)
summary.to_csv('results/summary.csv', index=False)
print(summary.round(4).to_string(index=False))

np.savez('results/full_results.npz',
         positions=positions, alpha=ALPHA, dt=dt, n=len(A), treated_share=A.mean(), n_pc=n_pc,
         X_cols=np.array(cols), het_var=VAR,
         fate_estimate=fate['estimate'], fate_se=fate['se'], fate_half_sup=fate['half_sup'],
         fate_q_stud=fate['q_stud'], fate_radius=fate['radius'], fate_plugin=fate_plugin,
         fate_seed1=fate_seeds[1]['estimate'], fate_seed2=fate_seeds[2]['estimate'],
         cut_profiles=P, cut_quantile_values=qvals, cut_estimate=cuts['estimate'], cut_se=cuts['se'],
         cut_half_sup=cuts['half_sup'], cut_q_stud=cuts['q_stud'], cut_radius=cuts['radius'],
         contrast_pairs=np.array(PAIRS), contrast_estimate=contr['estimate'], contrast_se=contr['se'],
         contrast_half_sup=contr['half_sup'], contrast_q_stud=contr['q_stud'], contrast_radius=contr['radius'],
         fate_fixact=fate_fixact['estimate'], fate_fixact_se=fate_fixact['se'],
         fate_fixact_q_stud=fate_fixact['q_stud'], fate_fixact_radius=fate_fixact['radius'],
         **{f'other_{v}_quantile_values': other[v][0] for v in OTHER_VARS},
         **{f'other_{v}_contrast': other[v][1]['estimate'] for v in OTHER_VARS},
         **{f'other_{v}_radius': other[v][1]['radius'] for v in OTHER_VARS},
         surface_values=svals, surface=S, pi_hat=pi_hat,
         # weights and residual curves, so that 05_plot_l2.py can redraw the multiplier process
         residuals=focal.residuals(), cut_weights=focal.weights(scaler.transform(P)),
         pseudo_diff=focal.get_pseudo_diff())
print(f'done; total runtime {time.time() - t0:.0f} s')
