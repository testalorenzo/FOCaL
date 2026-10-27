"""
07_supp.py -- supplementary material for the ENCODE application:
  1. sensitivity of the F-CATE cuts and contrasts to the spline basis of the third stage (6, 10, 15 = main,
     25 quantile knots; nuisances refit each time with the same seed), to the cross-fitting seed (from
     03_fit.py) and the plug-in FATE;
  2. the secondary fit with expression added to the covariates (effect at equal activity): FATE and
     F-CATE cuts along expression (10/50/90% quantiles) with their L2 sets, and contrasts;
  3. figures: results/fig_supp_overlap.pdf (propensities), results/fig_supp_secondary.pdf (secondary fit);
  4. results/supp_sensitivity.csv and the LaTeX block results/encode_supp.tex (table + figures).
Run from the encode/ folder after 03_fit.py:  python3 07_supp.py   (about 10 minutes)
"""

import sys
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '..')
from focal_apps import fit_focal, bands, contrast_bands, fate_bands, profiles_along, ball_envelope  # noqa: E402

ALPHA, SEED, VAR, EXPR = 0.05, 0, 'log10_dist_tss', 'log1p_TPM'
FIT = dict(fpca=0.99, n_knots=15, knots='quantile', ridge=0.01, max_iter=500, early_stopping=True)
COL = {'treated': '#D62728', 'untreated': '#1F77B4', 'fate': 'blue', 'cuts': ['#a99ff0', '#6250d6', '#2e2380']}
t0 = time.time()

d = np.load('data/processed/encode.npz', allow_pickle=True)
r = np.load('results/full_results.npz', allow_pickle=True)
X_all, A, Y, cols_all, pos = d['X'], d['A'], d['Y'], list(d['X_cols']), d['positions']
keep = [c != EXPR for c in cols_all]
X, cols = X_all[:, keep], [c for c, k in zip(cols_all, keep) if k]
dt = 1.0 / Y.shape[1]
summary = pd.read_csv('results/summary.csv').set_index('target')


def l2(est):
    return np.sqrt(dt * (est ** 2).sum())


rows = []


def add(analysis, target, inf, a):
    est = inf['estimate'][a]
    rows.append({'analysis': analysis, 'target': target, 'max_abs': np.abs(est).max(),
                 'argmax_bp': pos[np.argmax(np.abs(est))], 'L2_norm': l2(est), 'L2_radius': inf['radius'][a],
                 'reject': l2(est) > inf['radius'][a]})


# --- 1. basis sensitivity -------------------------------------------------------------------------------
P, qd = profiles_along(X, cols, VAR, quantiles=(0.1, 0.5, 0.9), center='median')
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_NAMES = ['close - typical', 'close - far', 'typical - far']
for nk in (6, 10, 15, 25):
    f, sc = fit_focal(X, A, Y, seed=SEED, **dict(FIT, n_knots=nk))
    lab = f'{nk} knots' + (' (main)' if nk == 15 else '')
    add(lab, 'FATE', fate_bands(f, ALPHA, dt, SEED), 0)
    cuts = bands(f, sc, P, ALPHA, dt, SEED)
    for k, q in enumerate((10, 50, 90)):
        add(lab, f'F-CATE, distance q{q}', cuts, k)
    contr = contrast_bands(f, sc, P, PAIRS, ALPHA, dt, SEED)
    for k, nm in enumerate(PAIR_NAMES):
        add(lab, f'contrast {nm}', contr, k)
    for v, nm in (('log10_length', 'gene length q90 - q10'), ('n_transcripts', 'transcripts q90 - q10')):
        P_v, _ = profiles_along(X, cols, v, quantiles=(0.1, 0.5, 0.9), center='median')
        add(lab, f'contrast {nm}', contrast_bands(f, sc, P_v, [(2, 0)], ALPHA, dt, SEED), 0)
    print(f'{lab} done at {time.time() - t0:.0f} s')

# seeds and plug-in from 03_fit.py
for s in (1, 2):
    rows.append({'analysis': f'seed {s}', 'target': 'FATE', 'max_abs': summary.loc[f'FATE_dr_seed{s}', 'max_abs_estimate'],
                 'argmax_bp': summary.loc[f'FATE_dr_seed{s}', 'argmax_position_bp'], 'L2_norm': summary.loc[f'FATE_dr_seed{s}', 'L2_norm'],
                 'L2_radius': summary.loc[f'FATE_dr_seed{s}', 'L2_radius'], 'reject': bool(summary.loc[f'FATE_dr_seed{s}', 'L2_reject'])})
rows.append({'analysis': 'outcome regression (T-learner)', 'target': 'FATE', 'max_abs': summary.loc['FATE_plugin', 'max_abs_estimate'],
             'argmax_bp': summary.loc['FATE_plugin', 'argmax_position_bp'], 'L2_norm': summary.loc['FATE_plugin', 'L2_norm'],
             'L2_radius': np.nan, 'reject': np.nan})

# --- 2. secondary fit with expression --------------------------------------------------------------------
f2, sc2 = fit_focal(X_all, A, Y, seed=SEED, **FIT)
fate2 = fate_bands(f2, ALPHA, dt, SEED)
P2, qe = profiles_along(X_all, cols_all, EXPR, quantiles=(0.1, 0.5, 0.9), center='median')
cuts2 = bands(f2, sc2, P2, ALPHA, dt, SEED)
contr2 = contrast_bands(f2, sc2, P2, PAIRS, ALPHA, dt, SEED)
add('expression in X (secondary)', 'FATE', fate2, 0)
for k, q in enumerate((10, 50, 90)):
    add('expression in X (secondary)', f'F-CATE, expression q{q}', cuts2, k)
for k, nm in enumerate(['low - median', 'low - high', 'median - high']):
    add('expression in X (secondary)', f'contrast {nm}', contr2, k)
pi2 = f2.get_pi_hat()
print(f'secondary fit done at {time.time() - t0:.0f} s; expression quantiles {np.round(qe, 2)}')

sens = pd.DataFrame(rows)
sens.to_csv('results/supp_sensitivity.csv', index=False)
pd.set_option('display.width', 220)
print(sens.round(3).to_string(index=False))

# --- 3. figures ------------------------------------------------------------------------------------------
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 11})
pi_hat = r['pi_hat']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, p, title in ((axes[0], pi_hat, 'main fit (pre-treatment covariates)'), (axes[1], pi2, 'secondary fit (expression added)')):
    edges = np.linspace(0, 1, 41)
    ax.hist(p[A == 0], bins=edges, color=COL['untreated'], alpha=0.65, label=f'no CpG island (n = {(A == 0).sum():,})')
    ax.hist(p[A == 1], bins=edges, color=COL['treated'], alpha=0.65, label=f'CpG island (n = {A.sum():,})')
    ax.set_xlabel(r'cross-fitted propensity score $\hat\pi(x)$'); ax.set_ylabel('genes'); ax.set_title(title)
    ax.legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig('results/fig_supp_overlap.pdf'); fig.savefig('results/fig_supp_overlap.png', dpi=110); plt.close(fig)

W2, E2 = f2.weights(sc2.transform(P2)), f2.residuals()
D2 = f2.get_pseudo_diff()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
ax = axes[0]
fate_main = r['fate_estimate'][0]
_, lo, hi, _ = ball_envelope(fate_main, np.full(len(A), 1 / len(A)), r['pseudo_diff'] - fate_main, ALPHA, dt, seed=0)
ax.fill_between(pos, lo, hi, color=COL['fate'], alpha=0.22, lw=0)
ax.plot(pos, fate_main, color=COL['fate'], lw=2, label='FATE, main fit (total effect)')
f2e = fate2['estimate'][0]
_, lo, hi, _ = ball_envelope(f2e, np.full(len(A), 1 / len(A)), D2 - f2e, ALPHA, dt, seed=0)
ax.fill_between(pos, lo, hi, color='#52514e', alpha=0.22, lw=0)
ax.plot(pos, f2e, color='#52514e', lw=2, ls='--', label='FATE, expression in the covariates')
ax.set_title('average effect'); ax.legend(frameon=False, loc='upper left')
ax = axes[1]
for k, (c, lab) in enumerate(zip(COL['cuts'], ('low', 'median', 'high'))):
    est = cuts2['estimate'][k]
    _, lo, hi, _ = ball_envelope(est, W2[k], E2, ALPHA, dt, seed=0)
    ax.fill_between(pos, lo, hi, color=c, alpha=0.18, lw=0)
    ax.plot(pos, est, color=c, lw=2, label=f'expression {lab} (q{(10, 50, 90)[k]}, log(1+TPM) = {qe[k]:.1f})')
ax.set_title('F-CATE at three expression levels'); ax.legend(frameon=False, loc='upper left', fontsize=9.5)
ax = axes[2]
for (k, l), c, lab in zip([(1, 0), (1, 2)], ['#eb6834', '#1baf7a'], ['median $-$ low', 'median $-$ high']):
    est = cuts2['estimate'][k] - cuts2['estimate'][l]
    _, lo, hi, _ = ball_envelope(est, W2[k] - W2[l], E2, ALPHA, dt, seed=0)
    ax.fill_between(pos, lo, hi, color=c, alpha=0.2, lw=0)
    ax.plot(pos, est, color=c, lw=2, label=lab)
ax.set_title('differences between expression levels'); ax.legend(frameon=False, loc='upper left')
for ax in axes:
    ax.axhline(0, color='k', lw=0.8, ls=':'); ax.axvline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('position relative to the gene start (bp)'); ax.set_ylabel(r'$\hat\tau$')
fig.tight_layout(); fig.savefig('results/fig_supp_secondary.pdf'); fig.savefig('results/fig_supp_secondary.png', dpi=110); plt.close(fig)

# --- 4. LaTeX block ----------------------------------------------------------------------------------------
yn = lambda v: '--' if pd.isna(v) else ('yes' if v else 'no')
fmt = lambda v: '--' if pd.isna(v) else f'{v:.3f}'
tab = ['\\begin{tabular}{llrrrrc}', '\\toprule',
       'Analysis & Target & max $|\\hat\\theta|$ & at (bp) & $\\|\\hat\\theta\\|$ & radius & rejects \\\\']
order = [f'{nk} knots' + (' (main)' if nk == 15 else '') for nk in (15, 6, 10, 25)] + ['seed 1', 'seed 2', 'outcome regression (T-learner)', 'expression in X (secondary)']
for an in order:
    sub = sens[sens.analysis == an]
    tab.append('\\midrule'); tab.append(f'\\multicolumn{{7}}{{l}}{{\\textit{{{an}}}}} \\\\')
    for _, rw in sub.iterrows():
        tab.append(f"& {rw.target} & {rw.max_abs:.2f} & {rw.argmax_bp:+.0f} & {rw.L2_norm:.3f} & {fmt(rw.L2_radius)} & {yn(rw.reject)} \\\\")
tab += ['\\bottomrule', '\\end{tabular}']
block = r'''% ---------------------------------------------------------------------------
% Supplement block for the ENCODE application (applications/encode/07_supp.py). Table rows generated
% from results/supp_sensitivity.csv. Whole block typeset in teal (revision).
% ---------------------------------------------------------------------------
{\color{teal}
\subsection{ENCODE study}

\begin{table}[h]
    \centering
    \caption{Variables employed in the ENCODE application (19,864 protein-coding genes of the cell line GM12878).}
    \label{tab:covariates_encode}
    \begin{tabular}{p{7.2cm}ll}
        \toprule
        \textbf{Variable name} & \textbf{Type} & \textbf{Usage} \\
        \midrule
        CpG island within 500 bases of the gene start & Binary & Treatment \\
        Gene length ($\log_{10}$ bases) & Scalar & Covariate \\
        Number of transcripts & Scalar & Covariate \\
        Distance to the nearest start of another gene ($\log_{10}(1+\text{bases})$) & Scalar & Covariate \\
        Reading direction (strand) & Binary & Covariate \\
        Expression ($\log(1+\text{TPM})$) & Scalar & Covariate, secondary analysis only \\
        H3K4me3 profile over the gene start $\pm 2{,}000$ bases ($\log_2(1+\text{fold change})$ in 80 bins of 50 bases) & Functional & Outcome \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{gfx/ENCODE_OVERLAP.pdf}
    \caption{Cross-fitted propensity scores in the ENCODE application, by treatment group. Left: main fit, with the four pre-treatment covariates (1st and 99th percentiles 0.13 and 0.94; 0.2\% of the genes above 0.95; none at the clipping bounds 0.02 and 0.98). Right: secondary fit with expression added to the covariates (99th percentile 0.97, 9.6\% above 0.95, none clipped): highly expressed genes almost all have an island.}
    \label{fig:encode_overlap}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{gfx/ENCODE_SECONDARY.pdf}
    \caption{Secondary ENCODE analysis with expression added to the covariates, i.e.\ the effect of the island among genes of equal activity. Left: FATE of the main fit (total effect, solid) and of the secondary fit (dashed), each with its 95\% global confidence set; adding expression halves the effect. Middle: F-CATE at the 10th, 50th and 90th percentiles of expression (other covariates at their medians) with their 95\% global sets. Right: differences between the median level and the two others, with their sets: the effect at equal activity is largest at median expression. Global sets are $\Ltwo$ balls drawn as the envelope of the simulated curves in the set.}
    \label{fig:encode_secondary}
\end{figure}

\begin{table}[h]
    \centering
    \caption{Results and sensitivity of the ENCODE application. For each target: the largest absolute value of the estimated curve and the position where it occurs (bases relative to the gene start, gene body on the positive side), the $\Ltwo$ norm of the estimate, the radius of its 95\% global confidence set, and whether the set excludes the zero function (``rejects''). Contrasts are differences between F-CATE cuts at the 10th (close), 50th (typical) and 90th (far) percentiles of the distance to the nearest other gene start, and between the 90th and 10th percentiles of gene length and of the number of transcripts. The main analysis uses 15 knots; the 6-, 10- and 25-knot rows refit the whole pipeline with another spline basis for the third stage, the seed rows with other cross-fitting splits; the outcome regression row is the difference of the two estimated outcome regressions (T-learner) with the same nuisance models; the last block is the secondary analysis of Figure~\ref{fig:encode_secondary}. The FATE does not depend on the third stage.}
    \label{tab:encode_sens}
    \footnotesize\setlength{\tabcolsep}{4pt}
''' + '\n'.join(tab) + r'''
\end{table}
}
'''
open('results/encode_supp.tex', 'w').write(block)
print(f'wrote results/encode_supp.tex; total runtime {time.time() - t0:.0f} s')
