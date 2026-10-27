"""
05_supp_table.py -- write the sensitivity table (LaTeX tabular) to results/sensitivity_table.tex
from results/sensitivity_<cond>.csv (03_diagnostics.py). Run from the share/ folder.
"""

import re
import pandas as pd

LAB = {'main (fPCA 99%, 6 knots, seed 0)': 'Main analysis (boosting, fPCA 99\\%, 6 knots, seed 0)',
       'fPCA 90%': 'fPCA 90\\% of the variance', 'fPCA 95%': 'fPCA 95\\% of the variance',
       '10 knots': 'Third stage with 10 knots', '15 knots': 'Third stage with 15 knots',
       'seed 1': 'Cross-fitting seed 1', 'seed 2': 'Cross-fitting seed 2',
       'linear / logistic nuisances': 'Linear / logistic nuisance models',
       'random-forest nuisances': 'Random-forest nuisance models',
       'neural-network nuisances (previous version)': 'Neural-network / logistic nuisance models (previous version)',
       're-smoothed curves, GCV': 'Re-smoothed curves, GCV', 're-smoothed curves, lambda=1e1': 'Re-smoothed curves, $\\lambda=10$',
       're-smoothed curves, lambda=1e3': 'Re-smoothed curves, $\\lambda=10^3$', 're-smoothed curves, lambda=1e5': 'Re-smoothed curves, $\\lambda=10^5$',
       'secondary: condition at every wave vs never': 'Condition at every wave vs never (previous design)'}
yn = lambda v: 'yes' if v else 'no'
rows = ['\\begin{tabular}{lcccccc}', '\\toprule',
        ' & \\multicolumn{3}{c}{CASP} & \\multicolumn{3}{c}{Mobility index} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
        'Analysis & $\\|\\hat\\theta\\|$ & radius & rejects & $\\|\\hat\\theta\\|$ & radius & rejects \\\\']
for cond, title in (('hypertension', 'Hypertension at enrolment'), ('high_cholesterol', 'High cholesterol at enrolment')):
    se = pd.read_csv(f'results/sensitivity_{cond}.csv')
    rows += ['\\midrule', f'\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\']
    for a, lab in LAB.items():
        if a not in set(se.analysis):
            continue
        c = se[(se.outcome == 'casp') & (se.analysis == a)].iloc[0]
        m = se[(se.outcome == 'mobility') & (se.analysis == a)].iloc[0]
        rows.append(f"{lab} & {c.FATE_L2:.2f} & {c.FATE_radius:.2f} & {yn(c.FATE_reject)} & "
                    f"{m.FATE_L2:.3f} & {m.FATE_radius:.3f} & {yn(m.FATE_reject)} \\\\")
rows += ['\\bottomrule', '\\end{tabular}']
open('results/sensitivity_table.tex', 'w').write('\n'.join(rows))
print('table rewritten with', len(rows) - 6, 'rows')
