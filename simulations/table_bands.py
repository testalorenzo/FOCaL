"""
table_bands.py -- supplementary table comparing inference procedures in the base simulation design
(results/summary_coverage_base.csv, 1000 repetitions): whole-curve coverage and width of
  - FOCaL's L2 confidence set (coverage of the true curve; radius),
  - the grid bands of the pointwise scalar DR-learner: pointwise 95% intervals at each of the 100 grid
    points (coverage = all 100 points covered), max-t band calibrated on the maximum over the grid,
    Bonferroni band (mean width over the grid for each),
  - the Bonferroni-over-scores set of the basis-wise scalar DR-learner (coverage of the 10 true scores;
    its width lives in the score space and is not comparable, so it is not shown).
Output: results/sim_bands_table.tex. Run from the simulations/ folder.
"""

import pandas as pd

c = pd.read_csv('results/summary_coverage_base.csv')
SCEN = ['Correct specification', 'Regression misspecified', 'Propensity misspecified', 'Fully misspecified']
PROF = [('FATE', 'FATE'), ('X1 q10', 'F-CATE, $X_1$ at q10'), ('X1 q50', 'F-CATE, $X_1$ at q50'), ('X1 q90', 'F-CATE, $X_1$ at q90')]
f = lambda v: f'{v:.2f}'
rows = ['\\begin{tabular}{llccccccccc}', '\\toprule',
        ' & & \\multicolumn{2}{c}{\\FOCAL{}, $\\Ltwo$ set} & \\multicolumn{2}{c}{pointwise} & \\multicolumn{2}{c}{max-$t$} & \\multicolumn{2}{c}{Bonferroni} & basis-wise \\\\',
        '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-11}',
        'Scenario & Target & cov. & radius & cov. & width & cov. & width & cov. & width & cov. \\\\']
for s in SCEN:
    rows.append('\\midrule')
    for k, (p, lab) in enumerate(PROF):
        fo = c[(c.scenario == s) & (c.method == 'FOCaL') & (c.profile == p)].iloc[0]
        pw = c[(c.scenario == s) & (c.method == 'Pointwise DR') & (c.profile == p)].iloc[0]
        bw = c[(c.scenario == s) & (c.method == 'Basis-wise DR') & (c.profile == p)].iloc[0]
        rows.append(f"{s if k == 0 else ''} & {lab} & {f(fo.cover_L2)} & {f(fo.radius)} & {f(pw.cover_pointwise_all_t)} & {f(pw.width_pointwise)} & "
                    f"{f(pw.cover_maxt)} & {f(pw.width_maxt)} & {f(pw.cover_bonf)} & {f(pw.width_bonf)} & {f(bw.cover_bonf)} \\\\")
rows += ['\\bottomrule', '\\end{tabular}']
block = r'''% ---------------------------------------------------------------------------
% Supplementary table: inference procedures compared in the base simulation design (code/table_bands.py,
% from code/results/summary_coverage_base.csv). Whole block typeset in teal (revision).
% ---------------------------------------------------------------------------
{\color{teal}
\begin{table}[h]
    \centering
    \caption{Inference procedures compared in the base simulation design (1000 repetitions, $n = 5000$). For \FOCAL{}, the empirical coverage of the true curve by the 95\% $\Ltwo$ confidence set and the average radius of the set. For the pointwise scalar DR-learner, which fits separate nuisance models and a separate third stage at each of the 100 grid points, three grid bands built from its pointwise standard errors: pointwise 95\% intervals (coverage = the true curve lies inside the interval at every grid point), a band calibrated on the $(1-\alpha)$-quantile of the maximum over the grid of the absolute studentized Gaussian process (max-$t$), and a Bonferroni band; width is the average width over the grid. For the basis-wise scalar DR-learner, which applies the scalar DR-learner to each of the 10 functional principal component scores (a special case of \FOCAL{} with a third stage on a fixed basis), the coverage of the vector of true scores by Bonferroni-adjusted intervals (its width is in the space of the scores and is not comparable). Targets: the FATE and the F-CATE at the 10th, 50th and 90th percentiles of $X_1$ with the other covariates at zero. Pointwise intervals undercover the whole curve by construction; the max-$t$ band and the $\Ltwo$ set reach nominal coverage whenever at least one nuisance model is correct; the Bonferroni band is conservative and about 30\% wider than the max-$t$ band. Under double misspecification, all procedures fail at the FATE and the median profile, where the bias is largest.}
    \label{tab:sim_bands}
    \footnotesize\setlength{\tabcolsep}{4pt}
''' + '\n'.join(rows) + r'''
\end{table}
}
'''
open('results/sim_bands_table.tex', 'w').write(block)
print('\n'.join(rows[5:]))
