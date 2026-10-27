"""
05_plot_l2.py -- the ENCODE results through the L2-ball confidence sets (Proposition 4.5).

For the FATE and the three F-CATE cuts, the 95% L2 ball { f : ||f - theta_hat||_{L2} <= radius }
is drawn as the pointwise envelope of the simulated curves theta_hat + G that fall inside it,
G being a Gaussian multiplier draw with the estimated covariance. The envelope is not a
confidence band: a curve inside the shading need not belong to the ball. The calibrated
statement is in each title: zero is in the set iff ||theta_hat||_{L2} <= radius.

Uses the weights and residual curves saved by 03_fit.py (no refit). Run from the encode/ folder.
Output: results/fig_l2ball.pdf/.png, results/l2ball_summary.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from focal_apps import ball_envelope, style, COLORS  # noqa: E402

ALPHA = 0.05
r = np.load('results/full_results.npz', allow_pickle=True)
pos, n = r['positions'], int(r['n'])
dt = float(r['dt'])                               # L2 quadrature weight saved by 03_fit.py (1/m)
D = r['pseudo_diff']
fate = D.mean(axis=0)

# (name, estimate, smoother weights, residual curves)
targets = [('FATE', fate, np.full(n, 1.0 / n), D - fate)]
for k, q in enumerate((10, 50, 90)):
    targets.append((f'F-CATE, distance q{q} (log10 bp = {r["cut_quantile_values"][k]:.2f})',
                    r['cut_estimate'][k], r['cut_weights'][k], r['residuals']))

style()
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
rows = []
for ax, (name, est, w, E) in zip(axes.ravel(), targets):
    radius, lo, hi, n_in = ball_envelope(est, w, E, ALPHA, dt, seed=0)
    l2 = np.sqrt(dt * (est ** 2).sum())
    rows.append({'target': name, 'L2_norm': l2, 'radius': radius, 'zero_in_set': l2 <= radius})
    ax.fill_between(pos, lo, hi, color='#9ecae1', alpha=0.6, lw=0,
                    label=f'envelope of the {n_in} simulated curves in the 95% L2 ball')
    ax.plot(pos, est, color=COLORS['est'], lw=2, label='FOCaL estimate')
    ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.6, ls=':')
    verdict = 'zero inside the set' if l2 <= radius else 'zero outside the set'
    ax.set_title(f'{name}\n$\\|\\hat\\theta\\|_{{L_2}}$ = {l2:.3f}, radius = {radius:.3f}: {verdict}', fontsize=10)
    ax.grid(axis='y', color='#e5e5e5', lw=0.6)
for ax in axes[1]:
    ax.set_xlabel('position relative to TSS (bp; gene body to the right)')
for ax in axes[:, 0]:
    ax.set_ylabel(r'effect on $\log_2(1 + \mathrm{fold\ change})$')
axes[0, 0].legend(frameon=False, fontsize=8.5, loc='upper right')
fig.suptitle('ENCODE: 95% L2-ball confidence sets (envelope of the curves in the set)', fontsize=12)
fig.tight_layout()
fig.savefig('results/fig_l2ball.pdf'); fig.savefig('results/fig_l2ball.png', dpi=160)
pd.DataFrame(rows).to_csv('results/l2ball_summary.csv', index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
