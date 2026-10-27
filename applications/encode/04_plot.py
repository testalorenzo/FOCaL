"""
04_plot.py -- figures for the ENCODE application.

results/fig_main.pdf/.png: (a) FATE with the studentized (shaded) and constant-width (dashed)
    95% simultaneous bands and the plug-in estimate; (b) F-CATE cuts at the 10/50/90% quantiles
    of the distance to the nearest other gene start with studentized bands; (c) heatmap of the
    F-CATE surface over (position, distance).
results/fig_data.pdf/.png: mean H3K4me3 profiles of the two groups with 25-75% bands.

Run from the encode/ folder:  python3 04_plot.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from focal_apps import style, plot_band, band_legend, plot_surface, COLORS  # noqa: E402

style()
XLAB = 'position relative to TSS (bp; gene body to the right)'
YLAB = r'effect on $\log_2(1 + \mathrm{fold\ change})$'
r = np.load('results/full_results.npz', allow_pickle=True)
t, alpha = r['positions'], float(r['alpha'])
fate = {k: r['fate_' + k] for k in ('estimate', 'se', 'half_sup', 'q_stud')}
cuts = {k: r['cut_' + k] for k in ('estimate', 'se', 'half_sup', 'q_stud')}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
plot_band(axes[0], t, fate, 0, label='FATE (FOCaL)')
axes[0].plot(t, r['fate_plugin'], color='#888888', lw=1.2, ls='-.', label='plug-in (T-learner)')
band_legend(axes[0], alpha)
axes[0].set_ylim(bottom=-0.15)
axes[0].set_title('(a) functional average treatment effect')

for k, (q, c) in enumerate(zip((10, 50, 90), COLORS['cuts'])):
    plot_band(axes[1], t, cuts, k, label=f'log10 distance q{q} = {r["cut_quantile_values"][k]:.2f}',
              color=c, show_sup=False)
axes[1].legend(frameon=False, title=f'{int(round(100 * (1 - alpha)))}% studentized bands', title_fontsize=8.5)
axes[1].set_title('(b) F-CATE at distance quantiles')

im = plot_surface(axes[2], t, r['surface_values'], r['surface'], 'log10 distance to nearest gene start (bp)', XLAB)
fig.colorbar(im, ax=axes[2], label='F-CATE')
axes[2].set_title('(c) F-CATE surface (other covariates at median)')
for ax in axes:
    ax.axvline(0, color=COLORS['zero'], lw=0.6, ls=':')
    ax.set_xlabel(XLAB)
axes[0].set_ylabel(YLAB); axes[1].set_ylabel(YLAB)
fig.tight_layout()
fig.savefig('results/fig_main.pdf'); fig.savefig('results/fig_main.png'); plt.close(fig)

d = np.load('data/processed/encode.npz', allow_pickle=True)
Y, A = d['Y'], d['A']
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for a, c, lab in ((0, COLORS['control'], 'no CpG island (A = 0)'), (1, COLORS['treated'], 'CpG island (A = 1)')):
    ax.fill_between(t, np.quantile(Y[A == a], 0.25, axis=0), np.quantile(Y[A == a], 0.75, axis=0),
                    color=c, alpha=0.2, lw=0)
    ax.plot(t, Y[A == a].mean(axis=0), color=c, lw=2, label=f'{lab}, n = {(A == a).sum()}')
ax.axvline(0, color=COLORS['zero'], lw=0.6, ls=':')
ax.set_xlabel(XLAB); ax.set_ylabel(r'H3K4me3 $\log_2(1 + \mathrm{fold\ change})$')
ax.set_title('mean profile with 25-75% band'); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('results/fig_data.pdf'); fig.savefig('results/fig_data.png'); plt.close(fig)
print('wrote results/fig_main and results/fig_data')
