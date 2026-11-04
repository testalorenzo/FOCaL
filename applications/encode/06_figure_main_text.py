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

Layout: everything is placed in inches (see "Layout"). Two rows of frames; the gap G between any
two neighbouring frames is the same, horizontally and vertically; (a) and (b) share top and bottom.

Reads data/processed/encode.npz and results/full_results.npz (weights and residual curves saved
by 03_fit.py). Can be launched from any folder.
Output: results/figure_encode_main.pdf/.png
"""

import os
import sys
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import seaborn as sns

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                                    # relative paths below work from any launch folder
sys.path.insert(0, str(HERE.parent))              # focal_apps.py lives in applications/
from focal_apps import ball_envelope  # noqa: E402

ALPHA = 0.05
COLORS = {'treated': '#D62728', 'untreated': '#1F77B4', 'fate': 'blue', 'total': '#52514e',
          'cuts': ['#a99ff0', '#6250d6', '#2e2380'], 'contrasts': ['#eb6834', '#1baf7a']}
POS_LABEL_1 = 'position relative to the gene start (bp)'
POS_LABEL_2 = 'position relative to\nthe gene start (bp)'      # for the narrower panels
POS_TICKS = [-2000, -1000, 0, 1000, 2000]
POS_TICKS_NARROW = [-2000, 0, 2000]                      # bottom-row panels
FS = dict(tick=19, label=22, title=22, legend=18, note=17.5, cov_title=19, block=26)
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 20, 'axes.titlesize': FS['title'], 'axes.labelsize': FS['label'],
                     'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['legend']})
os.makedirs('results', exist_ok=True)

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
W_cut, E = r['cut_weights'], r['residuals']
labels = {'log1p_TPM': 'gene activity, log(1+TPM)', 'log10_length': 'gene length, log10 bp',
          'n_transcripts': 'number of gene versions', 'log10_dist_tss': 'distance to next gene start, log10 bp',
          'strand_plus': 'reading direction'}
level = ['close', 'typical', 'far']


def l2_set(est, w, E):
    """Radius, envelope of the 95% L2 ball and whether zero belongs to it."""
    radius, lo, hi, _ = ball_envelope(est, w, E, ALPHA, dt, seed=0)
    return radius, lo, hi, np.sqrt(dt * (est ** 2).sum()) <= radius


def gene_axis(ax, xlabel=POS_LABEL_2, ticks=POS_TICKS_NARROW):
    """Zero line, gene-start marker and position label."""
    ax.axhline(0, color='k', lw=1.2, ls=':')
    ax.axvline(0, color='k', lw=1.0, ls=':')
    ax.set_xlabel(xlabel)
    ax.set_xticks(ticks)


# ---------------------------------------------------------------------------
# Layout (inches). Row 1 = (a) + (b); row 2 = (c) (d) (e) (f).
#   G is the single gap between any two neighbouring frames (horizontal and vertical).
#   (a) and (b) have the same bottom and top; (b) starts G to the right of (a) and takes the rest of the width.
# ---------------------------------------------------------------------------
W = 34.0                                          # figure width
M, G = 0.35, 0.60                                 # outer margin, gap between frames
H1, H2 = 6.8, 8.6                                 # heights of the top row and of the bottom row of frames
TOP = 0.45                                        # room above the top frames for their titles
H = M + H2 + G + H1 + TOP
U = W - 2 * M                                     # usable width

y2, y1 = M, M + H2 + G
w_a = 0.56 * (U - G)                              # share of the top row given to (a)
frames = {'a': (M, y1, w_a, H1), 'b': (M + w_a + G, y1, U - G - w_a, H1)}
ratios = [1.15, 1.25, 1.15, 1.15]                    # relative widths of (c), (d), (e), (f)
widths = np.array(ratios) / sum(ratios) * (U - 3 * G)
x = M
for key, w in zip('cdef', widths):
    frames[key] = (x, y2, w, H2)
    x += w + G

fig = plt.figure(figsize=(W, H))


def add_ax(x0, y0, w, h, **kw):
    """Axes from a rectangle given in inches."""
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


def inner(key, left, right, bottom, top):
    """Rectangle (inches) inside frame `key`, with the given paddings for ticks, labels and titles."""
    x0, y0, w, h = frames[key]
    return x0 + left, y0 + bottom, w - left - right, h - bottom - top


def legend_row(key, handles, labels_, ncol=2):
    """Legend centred in the strip at the top of a frame."""
    x0, y0, w, h = frames[key]
    fig.legend(handles=handles, labels=labels_, loc='center', ncol=ncol, frameon=False,
               bbox_to_anchor=((x0 + w / 2) / W, (y0 + h - 0.62) / H), fontsize=FS['legend'] + 1)


# (a) covariate densities -----------------------------------------------------------------------
xa, ya, wa_in, ha_in = inner('a', 1.7, 0.4, 0.95, 1.8)
gap_a = 1.5
aw = (wa_in - (len(cols) - 1) * gap_a) / len(cols)
for k, c in enumerate(cols):
    ax = add_ax(xa + k * (aw + gap_a), ya, aw, ha_in)
    xk = X[:, k]
    if c == 'strand_plus':
        ax.bar([0, 1], [xk[A == 0].mean(), xk[A == 1].mean()], color=[COLORS['untreated'], COLORS['treated']], alpha=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['no island', 'island'])
        ax.set_ylabel('share read\nleft to right', fontsize=FS['label'] - 2)
    else:
        sns.kdeplot(xk[A == 1], fill=True, color=COLORS['treated'], alpha=0.6, common_norm=False, ax=ax)
        sns.kdeplot(xk[A == 0], fill=True, color=COLORS['untreated'], alpha=0.6, common_norm=False, ax=ax)
        ax.set_ylabel('density' if k == 0 else '')
    ax.set_title('\n'.join(textwrap.wrap(labels[c], 22)), fontsize=FS['cov_title'], pad=10)
    ax.set_xlabel('')
legend_row('a', [Patch(facecolor=COLORS['treated'], alpha=0.6), Patch(facecolor=COLORS['untreated'], alpha=0.6)],
           ['CpG island at the gene start (A = 1)', 'no CpG island (A = 0)'])

# (b) observed profiles ---------------------------------------------------------------------------
xb, yb, wb_in, hb_in = inner('b', 1.9, 0.5, 0.95, 1.8)
ax = add_ax(xb, yb, wb_in, hb_in)
for a, c, lab in ((0, COLORS['untreated'], 'no CpG island'), (1, COLORS['treated'], 'CpG island')):
    ax.fill_between(pos, np.quantile(Y[A == a], 0.25, axis=0), np.quantile(Y[A == a], 0.75, axis=0),
                    color=c, alpha=0.25, lw=0)
    ax.plot(pos, Y[A == a].mean(axis=0), color=c, lw=2.8, label=f'{lab} (n = {(A == a).sum():,})')
ax.axvline(0, color='k', lw=1.0, ls=':')
ax.set_xlabel(POS_LABEL_1); ax.set_ylabel(r'H3K4me3, $\log_2(1+\mathrm{fc})$')
ax.set_xticks(POS_TICKS)
ax.annotate('gene body', xy=(1900, 0.1), xytext=(600, 0.1), fontsize=FS['note'], va='center',
            arrowprops=dict(arrowstyle='->', color='k', lw=1.4))
ax.set_title('mean and interquartile band', fontsize=FS['cov_title'], pad=10)
h_b, l_b = ax.get_legend_handles_labels()
legend_row('b', h_b, l_b)

# (c) FATE with the L2 global confidence set -------------------------------------------------------
ax_f = add_ax(*inner('c', 1.8, 0.5, 1.5, 1.05))
radius, lo, hi, zero_in = l2_set(fate, np.full(n, 1.0 / n), D - fate)
ax_f.fill_between(pos, lo, hi, color=COLORS['fate'], alpha=0.25, lw=0, label='95% global confidence set')
ax_f.plot(pos, fate, color=COLORS['fate'], lw=3.0, label='FATE')
gene_axis(ax_f)
ax_f.set_ylabel(r'$\hat\tau$')
ax_f.set_title('FATE', pad=14)
ax_f.set_ylim(-0.25, 1.75)
ax_f.legend(frameon=False, loc='upper left')
cmp_ = r'$\leq$' if zero_in else '>'
ax_f.text(0.5, 0.03, f'$\\|\\hat\\tau\\|_{{L_2}}$ = {np.sqrt(dt * (fate ** 2).sum()):.2f} {cmp_} radius = {radius:.3f}\n'
          f'zero {"in" if zero_in else "not in"} the set', transform=ax_f.transAxes, ha='center', va='bottom',
          fontsize=FS['note'])

# (d) F-CATE surface (3-D, set up as in the simulation figure) ---------------------------------------
fx, fy, fw, fh = frames['d']
cb_w, cb_pad = 0.34, 1.5                                         # colorbar width; room at the right of the frame
rx, ry, rw, rh = inner('d', 0.5, cb_pad + 0.9, 0.3, 1.05)         # rectangle of the 3-D axes
ax_s = add_ax(rx, ry + 0.4, rw, rh, projection='3d')            # 0.4 in higher than the frame's inner rectangle
S, sv = r['surface'], r['surface_values']
T, V = np.meshgrid(pos, sv)
lim = np.abs(S).max()
surf = ax_s.plot_surface(T, V, S, cmap=cm.coolwarm, vmin=-lim, vmax=lim, linewidth=0, antialiased=True, alpha=0.95)
ax_s.set_zlim(np.floor(S.min() * 2) / 2, np.ceil(S.max() * 2) / 2)
ax_s.set_xlabel('position (bp)', labelpad=14, fontsize=FS['label'] - 1)
ax_s.set_ylabel('log10 distance (bp)', labelpad=14, fontsize=FS['label'] - 1)
ax_s.set_zlabel(r'$\hat\tau$', labelpad=10, fontsize=FS['label'] + 2)
ax_s.set_xticks([-2000, 0, 2000])
for axis in 'yz':
    ax_s.locator_params(axis=axis, nbins=4)
ax_s.tick_params(axis='both', which='major', labelsize=FS['tick'] - 2, pad=3)
ax_s.tick_params(axis='z', which='major', labelsize=FS['tick'] - 2, pad=8)
ax_s.set_box_aspect((1.5, 1.25, 1.35), zoom=1.15)                  # long position axis, as in the simulation figure
ax_s.view_init(elev=26, azim=-58)
fig.text((fx + fw / 2) / W, (fy + fh - 0.8) / H, 'F-CATE(distance to next gene start)', ha='center', va='center',
         fontsize=FS['title'])
cax = add_ax(fx + fw - cb_pad + 0.4, fy + fh / 2 - 1.9, cb_w, 3.8)
cbar = fig.colorbar(surf, cax=cax)
cbar.ax.tick_params(labelsize=FS['tick'] - 2)
cbar.outline.set_visible(False)

# (e) surface cuts with their L2 global sets -------------------------------------------------------
ax_c = add_ax(*inner('e', 1.8, 0.5, 1.5, 1.05))
zero_verdicts = []
for k, c in enumerate(COLORS['cuts']):
    est = r['cut_estimate'][k]
    radius, lo, hi, zero_in = l2_set(est, W_cut[k], E)
    zero_verdicts.append(zero_in)
    ax_c.fill_between(pos, lo, hi, color=c, alpha=0.18, lw=0)
    ax_c.plot(pos, est, color=c, lw=3.0,
              label=f'{level[k]}: {10 ** qvals[k] - 1:,.0f} bp (q{(10, 50, 90)[k]})')
ax_c.text(0.04, 0.03, 'zero not in the 95% global set\nat any cut' if not any(zero_verdicts) else
          f'zero in the set at {sum(zero_verdicts)} cut(s)', transform=ax_c.transAxes, va='bottom', fontsize=FS['note'])
gene_axis(ax_c)
ax_c.set_ylabel(r'$\hat\tau$')
ax_c.set_title('F-CATE at three distances', pad=14)
ax_c.legend(frameon=False, loc='upper right')
ax_c.set_ylim(-0.6, 3.0)

# (f) differences between distances with their L2 global sets -------------------------------------
ax_d = add_ax(*inner('f', 1.9, 0.5, 1.5, 1.05))
for (k, l), c in zip([(0, 1), (0, 2)], COLORS['contrasts']):
    est = r['cut_estimate'][k] - r['cut_estimate'][l]
    radius, lo, hi, zero_in = l2_set(est, W_cut[k] - W_cut[l], E)
    ax_d.fill_between(pos, lo, hi, color=c, alpha=0.2, lw=0)
    ax_d.plot(pos, est, color=c, lw=3.0,
              label=f'{level[k]} $-$ {level[l]} (zero {"in" if zero_in else "not in"} the set)')
gene_axis(ax_d)
ax_d.set_ylabel(r'difference in $\hat\tau$')
ax_d.set_title('differences between distances', pad=14)
ax_d.legend(frameon=False, loc='upper left')
ax_d.set_ylim(-0.8, 1.6)

# ---------------------------------------------------------------------------
# Framed blocks with titles, as in the paper's application figures (drawn in inches on an overlay)
# ---------------------------------------------------------------------------
ov = fig.add_axes([0, 0, 1, 1], zorder=-1)
ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off')
rend = fig.canvas.get_renderer()


def block_frame(key, title):
    """Rounded frame with the title centred on its top edge."""
    x0, y0, w, h = frames[key]
    ov.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.35', fill=False,
                                lw=1.8, edgecolor='grey', zorder=2))
    t = ov.text(x0 + w / 2, y0 + h, title, ha='center', va='center', fontsize=FS['block'], weight='bold', zorder=4)
    bb = t.get_window_extent(rend).transformed(ov.transData.inverted())
    ov.add_patch(Rectangle((bb.x0 - 0.2, bb.y0), bb.width + 0.4, bb.height, facecolor='white', edgecolor='none', zorder=3))


for key, title in (('a', '(a) Covariate densities'), ('b', '(b) Observed outcome profiles'),
                   ('c', '(c) Average effect'), ('d', '(d) Heterogeneous effect'),
                   ('e', '(e) Surface cuts'), ('f', '(f) Contrasts')):
    block_frame(key, title)

fig.savefig('results/figure_encode_main.pdf', bbox_inches='tight')
fig.savefig('results/figure_encode_main.png', dpi=130, bbox_inches='tight')
print('wrote results/figure_encode_main.pdf/.png')