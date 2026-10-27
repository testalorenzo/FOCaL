"""
06_figure_main_text.py -- main-text figure for the ENCODE application, in the style of the
paper's application figures (framed blocks, covariate densities, FATE, 3-D F-CATE surface,
surface cuts), using only the L2 global confidence sets of Proposition 4.5.

Blocks:
  (a) covariate densities by treatment group;
  (b) observed H3K4me3 profiles of the two groups (mean and interquartile band);
  (c) FATE (pre-treatment covariates, total effect) with the 95% global confidence set;
  (d) F-CATE surface over position and distance to the nearest other gene start;
  (e) cuts of the surface at the 10/50/90% quantiles of that distance with their 95% global sets;
  (f) differences between cuts (q10 - q50, q10 - q90) with their 95% global sets (a difference
      of linear smoothers has weights omega_i(x_k) - omega_i(x_l)).
The global set is an L2 ball; it is drawn as the pointwise envelope of the simulated curves
that belong to the set (see the caption), with "zero in the set" tested by norm vs radius.
Colors: treated/untreated red/blue as in the paper; distance levels on one violet ramp
(ordinal, light = close); the two differences orange and aqua (validated with the dataviz checker).

Reads data/processed/encode.npz and results/full_results.npz (weights and residual curves saved
by 03_fit.py). Run from the encode/ folder.
Output: results/figure_encode_main.pdf/.png
"""

import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.transforms import Bbox
import seaborn as sns

sys.path.insert(0, '..')
from focal_apps import ball_envelope  # noqa: E402

ALPHA = 0.05
COLORS = {'treated': '#D62728', 'untreated': '#1F77B4', 'fate': 'blue', 'total': '#52514e',
          'cuts': ['#a99ff0', '#6250d6', '#2e2380'], 'contrasts': ['#eb6834', '#1baf7a']}
POS_LABEL = 'position relative to the gene start (bp)'
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14.5})

# ---------------------------------------------------------------------------
# Data and results
# ---------------------------------------------------------------------------
d = np.load('data/processed/encode.npz', allow_pickle=True)
r = np.load('results/full_results.npz', allow_pickle=True)
A, Y = d['A'], d['Y']
cols = list(r['X_cols'])                          # covariates of the main fit (pre-treatment)
X = d['X'][:, [list(d['X_cols']).index(c) for c in cols]]
pos = r['positions']
dt = float(r['dt'])                               # L2 quadrature weight saved by 03_fit.py (1/m)
n = len(A)
D = r['pseudo_diff']
fate = D.mean(axis=0)
qvals = r['cut_quantile_values']
W, E = r['cut_weights'], r['residuals']
labels = {'log1p_TPM': 'gene activity, log(1+TPM)', 'log10_length': 'gene length, log10 bp',
          'n_transcripts': 'number of gene versions', 'log10_dist_tss': 'distance to next gene start, log10 bp',
          'strand_plus': 'reading direction'}
level = ['close', 'typical', 'far']


def l2_set(est, w, E):
    """Radius, envelope of the 95% L2 ball and whether zero belongs to it."""
    radius, lo, hi, _ = ball_envelope(est, w, E, ALPHA, dt, seed=0)
    return radius, lo, hi, np.sqrt(dt * (est ** 2).sum()) <= radius


def gene_axis(ax):
    """Zero line, gene-start marker and position label."""
    ax.axhline(0, color='k', lw=1, ls=':')
    ax.axvline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel(POS_LABEL)
    ax.set_xticks([-2000, -1000, 0, 1000, 2000])


# ---------------------------------------------------------------------------
# Layout: row 1 = (a) densities + (b) observed profiles; row 2 = (c) FATE, (d) surface, (e) cuts, (f) contrasts
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(32, 15))
outer = gridspec.GridSpec(2, 1, height_ratios=[0.8, 1.0], hspace=0.8)
top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.55, width_ratios=[1, 1, 1, 1, 0.35, 1.6])
bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.65, width_ratios=[1, 1.15, 1, 1])

# (a) covariate densities
for k, c in enumerate(cols):
    ax = fig.add_subplot(top[0, k])
    x = X[:, k]
    if c == 'strand_plus':
        ax.bar([0, 1], [x[A == 0].mean(), x[A == 1].mean()], color=[COLORS['untreated'], COLORS['treated']], alpha=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['no island', 'island'])
        ax.set_ylabel('share read left to right')
    else:
        sns.kdeplot(x[A == 1], fill=True, color=COLORS['treated'], alpha=0.6, common_norm=False, ax=ax)
        sns.kdeplot(x[A == 0], fill=True, color=COLORS['untreated'], alpha=0.6, common_norm=False, ax=ax)
        ax.set_ylabel('density' if k == 0 else '')
    ax.set_title(labels[c], fontsize=15)
    ax.set_xlabel('')

# (b) observed profiles
ax = fig.add_subplot(top[0, 5])
for a, c, lab in ((0, COLORS['untreated'], 'no CpG island'), (1, COLORS['treated'], 'CpG island')):
    ax.fill_between(pos, np.quantile(Y[A == a], 0.25, axis=0), np.quantile(Y[A == a], 0.75, axis=0),
                    color=c, alpha=0.25, lw=0)
    ax.plot(pos, Y[A == a].mean(axis=0), color=c, lw=2.2, label=f'{lab} (n = {(A == a).sum():,})')
ax.axvline(0, color='k', lw=0.8, ls=':')
ax.set_xlabel(POS_LABEL); ax.set_ylabel(r'H3K4me3, $\log_2(1+\mathrm{fc})$')
ax.set_xticks([-2000, -1000, 0, 1000, 2000])
ax.annotate('gene body', xy=(1900, 0.1), xytext=(700, 0.1), fontsize=14, va='center',
            arrowprops=dict(arrowstyle='->', color='k', lw=1))
ax.set_title('mean and interquartile band', fontsize=15)
ax.legend(frameon=False, loc='upper left')

# (c) FATE with the L2 global confidence set; dashed: fit without expression in X
ax_f = fig.add_subplot(bottom[0, 0])
radius, lo, hi, zero_in = l2_set(fate, np.full(n, 1.0 / n), D - fate)
ax_f.fill_between(pos, lo, hi, color=COLORS['fate'], alpha=0.25, lw=0, label='95% global confidence set')
ax_f.plot(pos, fate, color=COLORS['fate'], lw=2.4, label='FATE')
gene_axis(ax_f)
ax_f.set_ylabel(r'$\hat\tau$')
ax_f.set_title('FATE', fontsize=17)
ax_f.set_ylim(-0.25, 1.75)
ax_f.legend(frameon=False, loc='upper left')
ax_f.text(0.5, 0.03, f'$\\|\\hat\\tau\\|_{{L_2}}$ = {np.sqrt(dt * (fate ** 2).sum()):.2f} > radius = {radius:.3f}: '
          f'zero {"in" if zero_in else "not in"} the set', transform=ax_f.transAxes, ha='center', va='bottom', fontsize=14)

# (d) F-CATE surface (3-D, as in the paper)
ax_s = fig.add_subplot(bottom[0, 1], projection='3d')
S, sv = r['surface'], r['surface_values']
T, V = np.meshgrid(pos, sv)
lim = np.abs(S).max()
surf = ax_s.plot_surface(T, V, S, cmap=cm.coolwarm, vmin=-lim, vmax=lim, linewidth=0, alpha=0.9)
ax_s.set_xlabel('position (bp)', labelpad=10); ax_s.set_ylabel('log10 distance (bp)', labelpad=10)
ax_s.set_zlabel(r'$\hat\tau$', labelpad=6)
ax_s.set_title('F-CATE(distance to next gene start)', fontsize=17)
ax_s.view_init(elev=28, azim=-55)
cbar = fig.colorbar(surf, ax=ax_s, shrink=0.55, pad=0.1)

# (e) surface cuts with their L2 global sets
ax_c = fig.add_subplot(bottom[0, 2])
zero_verdicts = []
for k, c in enumerate(COLORS['cuts']):
    est = r['cut_estimate'][k]
    radius, lo, hi, zero_in = l2_set(est, W[k], E)
    zero_verdicts.append(zero_in)
    ax_c.fill_between(pos, lo, hi, color=c, alpha=0.18, lw=0)
    ax_c.plot(pos, est, color=c, lw=2.2,
              label=f'next gene start {level[k]}: {10 ** qvals[k] - 1:,.0f} bp (q{(10, 50, 90)[k]})')
ax_c.text(0.03, 0.03, 'zero not in the 95% global set at any cut' if not any(zero_verdicts) else
          f'zero in the set at {sum(zero_verdicts)} cut(s)', transform=ax_c.transAxes, va='bottom', fontsize=14)
gene_axis(ax_c)
ax_c.set_ylabel(r'$\hat\tau$')
ax_c.set_title('F-CATE at three distances', fontsize=17)
ax_c.legend(frameon=False, loc='upper right', fontsize=13.5)
ax_c.set_ylim(-0.6, 3.0)

# (f) differences between activity levels with their L2 global sets
ax_d = fig.add_subplot(bottom[0, 3])
for (k, l), c in zip([(0, 1), (0, 2)], COLORS['contrasts']):
    est = r['cut_estimate'][k] - r['cut_estimate'][l]
    radius, lo, hi, zero_in = l2_set(est, W[k] - W[l], E)
    ax_d.fill_between(pos, lo, hi, color=c, alpha=0.2, lw=0)
    ax_d.plot(pos, est, color=c, lw=2.2,
              label=f'{level[k]} $-$ {level[l]}: zero {"in" if zero_in else "not in"} the set')
gene_axis(ax_d)
ax_d.set_ylabel(r'difference in $\hat\tau$')
ax_d.set_title('differences between distances', fontsize=17)
ax_d.legend(frameon=False, loc='upper left', fontsize=13.5)
ax_d.set_ylim(-0.8, 1.6)


# ---------------------------------------------------------------------------
# Framed blocks with titles, as in the paper's application figures
# ---------------------------------------------------------------------------
def add_block_frame(fig, bbox, title, mask_frac=0.3, pad_x=0.012, pad_y=0.035, extra_top=1.6):
    x0, y0 = bbox.x0 - pad_x, bbox.y0 - pad_y
    w, h = bbox.width + 2 * pad_x, bbox.height + (1 + extra_top) * pad_y
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.008,rounding_size=0.02',
                                  transform=fig.transFigure, fill=False, lw=1.5, edgecolor='grey', zorder=2))
    tx, ty = x0 + w / 2, y0 + h + 0.005
    fig.add_artist(FancyBboxPatch((tx - mask_frac * w / 2, ty - 0.014), mask_frac * w, 0.03,
                                  boxstyle='round,pad=0.0,rounding_size=0.01', transform=fig.transFigure,
                                  facecolor='white', edgecolor='none', zorder=3))
    fig.text(tx, ty, title, ha='center', va='center', fontsize=18, weight='bold', zorder=4)


fig.canvas.draw()
rend = fig.canvas.get_renderer()
tight = lambda a: a.get_tightbbox(rend).transformed(fig.transFigure.inverted())
bb = Bbox.union([tight(fig.axes[k]) for k in range(len(cols))])
add_block_frame(fig, bb, '(a) Covariate densities', mask_frac=0.35, extra_top=2.2)
fig.legend(handles=[Patch(facecolor=COLORS['treated'], alpha=0.6, label='CpG island at the gene start (A = 1)'),
                    Patch(facecolor=COLORS['untreated'], alpha=0.6, label='no CpG island (A = 0)')],
           loc='center', bbox_to_anchor=(bb.x0 + bb.width / 2, bb.y1 + 0.04), ncol=2, frameon=False)
add_block_frame(fig, tight(fig.axes[len(cols)]), '(b) Observed outcome profiles', mask_frac=0.9, extra_top=2.2, pad_x=0.012)
bbc = tight(ax_f)
add_block_frame(fig, bbc, '(c) Average effect', mask_frac=0.6)
# the 3-D axes box is shorter than the 2-D ones: frame (d) spans the 3-D axes and its colorbar
# horizontally and the same height as frame (c)
bs, bc, bf = tight(ax_s), tight(cbar.ax), tight(ax_f)
bbe = tight(ax_c)
# frame (d): from just right of frame (c) to the colorbar, same height as frame (c)
add_block_frame(fig, Bbox([[bbc.x1 + 0.05, bf.y0], [min(bc.x1, bbe.x0 - 0.05), bf.y1]]), '(d) Heterogeneous effect', mask_frac=0.7)
add_block_frame(fig, bbe, '(e) Surface cuts', mask_frac=0.55)
add_block_frame(fig, tight(ax_d), '(f) Contrasts', mask_frac=0.45)

fig.savefig('results/figure_encode_main.pdf', bbox_inches='tight')
fig.savefig('results/figure_encode_main.png', dpi=130, bbox_inches='tight')
print('wrote results/figure_encode_main.pdf/.png')
