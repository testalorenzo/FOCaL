"""
04_figure.py -- SHARE figures with the new results (hypertension: main text; high cholesterol: supplement).

Layout and style of the original paper figure, with the new results and no annotations:
  (a) covariate densities by treatment group across the top (treatment = condition at the first interview);
  (b) CASP and (c) mobility index side by side: top, the FATE with its 95% global confidence set (the L2
      ball of Proposition 4.6 drawn as the pointwise envelope of the simulated curves that belong to it);
      bottom, the 3-D F-CATE surfaces over (time, age) for men and women, with one colour and vertical scale
      per outcome shared by the two genders and the colorbar; other covariates at their medians; point estimates.
Reads data/processed/share.npz and results/results_<cond>_<outcome>.npz. Run from the share/ folder.
Output: results/figure_share_<cond>.pdf/.png
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
from scipy.stats import gaussian_kde

sys.path.insert(0, '..')
from focal_apps import ball_envelope  # noqa: E402

ALPHA = 0.05
C = {'treated': '#D62728', 'untreated': '#1F77B4', 'fate': 'blue', 'ink': '#0b0b0b', 'ink2': '#52514e',
     'grid': '#e6e5e1', 'neg': '#6250d6', 'pos': '#eb6834', 'mid': '#f0efec'}
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})
LABELS = {'age': 'Age (years)', 'education_years': 'Education (years)', 'n_children': 'Number of children',
          'female': 'Gender', 'vaccinated_childhood': 'Childhood vaccinations', 'ever_smoked': 'Ever smoked'}
OUT = {'casp': ('CASP', 'higher = better quality of life'), 'mobility': ('Mobility index', 'higher = more impairment')}
T_LABEL = 'Months since first interview'

d = np.load('data/processed/share.npz', allow_pickle=True)
X, cols, months = d['X'], list(d['X_cols']), d['months']



def block_frame(fig, bbox, title, mask_frac=0.3, pad_x=0.012, pad_y=0.03, extra_top=1.4):
    """Rounded frame around a block of axes with the title centred on its top edge, as in the original figure."""
    x0, y0 = bbox.x0 - pad_x, bbox.y0 - pad_y
    w, h = bbox.width + 2 * pad_x, bbox.height + (1 + extra_top) * pad_y
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.008,rounding_size=0.02',
                                  transform=fig.transFigure, fill=False, lw=1.5, edgecolor='grey', zorder=2))
    tx, ty = x0 + w / 2, y0 + h + 0.004
    fig.add_artist(FancyBboxPatch((tx - mask_frac * w / 2, ty - 0.012), mask_frac * w, 0.026,
                                  boxstyle='round,pad=0.0,rounding_size=0.01', transform=fig.transFigure,
                                  facecolor='white', edgecolor='none', zorder=3))
    fig.text(tx, ty, title, ha='center', va='center', fontsize=15, weight='bold', zorder=4)


ORIG_TITLES = {'age': 'age', 'education_years': 'education', 'n_children': 'number_children',
               'female': 'gender', 'vaccinated_childhood': 'vaccinations', 'ever_smoked': 'smoke'}


def density_panel(ax, x, A, name):
    """Covariate panel in the style of the original figure (kde / bars / proportions)."""
    if len(np.unique(x)) <= 2:
        ax.bar([0, 1], [x[A == 0].mean(), x[A == 1].mean()], width=0.6, color=[C['untreated'], C['treated']], alpha=0.75)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Untreated', 'Treated']); ax.set_ylim(0, 1); ax.set_ylabel('Proportion')
    elif name == 'n_children':
        vals = np.arange(0, 8)
        for a, cc, off in ((0, C['untreated'], -0.2), (1, C['treated'], 0.2)):
            ax.bar(vals + off, [(x[A == a] == v).mean() for v in vals], width=0.4, color=cc, alpha=0.75)
        ax.set_xticks(vals); ax.set_ylabel('Proportion')
    else:
        grid = np.linspace(x.min() - 3, x.max() + 3, 200)
        for a, cc in ((1, C['treated']), (0, C['untreated'])):
            kde = gaussian_kde(x[A == a])(grid)
            ax.fill_between(grid, kde, color=cc, alpha=0.4, lw=0)
            ax.plot(grid, kde, color=cc, lw=1.5)
        ax.set_ylabel('Density')
    ax.set_title(ORIG_TITLES[name], fontsize=13)


def fate_panel(ax, res):
    """FATE with its 95% global confidence set (envelope of the simulated curves in the L2 ball)."""
    dt = float(res['dt']); n = int(res['n'])
    D = res['pseudo_diff']; fate = D.mean(axis=0)
    _, lo, hi, _ = ball_envelope(fate, np.full(n, 1.0 / n), D - fate, ALPHA, dt, seed=0)
    ax.fill_between(months, lo, hi, color=C['fate'], alpha=0.28, lw=0)
    ax.plot(months, fate, color=C['fate'], lw=2.2)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xlim(0, 191); ax.set_xlabel('Time'); ax.set_ylabel(r'$\hat\tau$'); ax.set_title('FATE', fontsize=13)


def surface_panels(fig, axes, res):
    """3-D F-CATE surfaces for men and women, one colour and vertical scale per outcome, as in the original."""
    S = {'M': res['surface_men'], 'F': res['surface_women']}
    ages = res['surface_ages']
    zlo = min(v.min() for v in S.values()); zhi = max(v.max() for v in S.values())
    T, V = np.meshgrid(months, ages)
    surf = None
    for ax, (sex, Sv) in zip(axes, S.items()):
        surf = ax.plot_surface(T, V, Sv, cmap=cm.coolwarm, vmin=zlo, vmax=zhi, linewidth=0, antialiased=True)
        ax.set_zlim(zlo, zhi); ax.set_xlim(0, 191); ax.set_ylim(ages.min(), ages.max())
        ax.set_xlabel('Time', labelpad=4); ax.set_ylabel('Age', labelpad=4); ax.set_zlabel(r'$\hat\tau$', labelpad=2)
        ax.set_title(f'FCATE(age, gender={sex})', fontsize=13, pad=0)
        ax.view_init(elev=28, azim=-55)
        ax.tick_params(labelsize=9.5, pad=1)
        ax.set_box_aspect((1.3, 1, 0.7), zoom=0.92)
    cb = fig.colorbar(surf, ax=list(axes), fraction=0.03, pad=0.1, aspect=18, shrink=0.55)
    cb.set_label(r'$\hat\tau$')
    return cb


def make_figure(cond, title_cond):
    """Organisation of the original figure: (a) covariate densities across the top; (b) CASP and (c) mobility
    side by side, each with the FATE above the two surfaces (men, women) and a shared colorbar."""
    A = d[f'A_{cond}'].astype(int)
    fig = plt.figure(figsize=(20, 14.5))
    outer = gridspec.GridSpec(2, 1, height_ratios=[0.45, 1.1], hspace=0.42)
    top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.4)
    dens = [fig.add_subplot(top[0, k]) for k in range(6)]
    for ax, k in zip(dens, range(6)):
        density_panel(ax, X[:, k], A, cols[k])

    blocks = {}
    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.25)
    for b, outcome in enumerate(('casp', 'mobility')):
        res = np.load(f'results/results_{cond}_{outcome}.npz', allow_pickle=True)
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=bottom[0, b], height_ratios=[0.7, 1.0],
                                                 hspace=0.4, wspace=0.3)
        ax_f = fig.add_subplot(inner[0, :])
        fate_panel(ax_f, res)
        pos = ax_f.get_position()
        ax_f.set_position([pos.x0 + 0.1 * pos.width, pos.y0, 0.8 * pos.width, pos.height])
        ax_s = [fig.add_subplot(inner[1, 0], projection='3d'), fig.add_subplot(inner[1, 1], projection='3d')]
        cb = surface_panels(fig, ax_s, res)
        blocks[outcome] = (ax_f, ax_s, cb)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tight = lambda a: a.get_tightbbox(rend).transformed(fig.transFigure.inverted())
    bb = Bbox.union([tight(ax) for ax in dens])
    # centre the bottom blocks under block (a): shift every bottom axis (and colorbar) horizontally
    bottom_axes = [a for ax_f, ax_s, cb in blocks.values() for a in [ax_f] + ax_s + [cb.ax]]
    bb_bottom = Bbox.union([tight(a) for a in bottom_axes])
    dx = (bb.x0 + bb.x1) / 2 - (bb_bottom.x0 + bb_bottom.x1) / 2
    for a in bottom_axes:
        p = a.get_position(); a.set_position([p.x0 + dx, p.y0, p.width, p.height])
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    block_frame(fig, bb, '(a) Covariate Densities', mask_frac=0.3, extra_top=2.0)
    fig.legend(handles=[Patch(facecolor=C['treated'], alpha=0.75, label='Treated'),
                        Patch(facecolor=C['untreated'], alpha=0.75, label='Untreated')],
               loc='center', bbox_to_anchor=(bb.x0 + bb.width / 2, bb.y1 + 0.045), ncol=2, frameon=False, fontsize=12)
    for letter, outcome in (('b', 'casp'), ('c', 'mobility')):
        ax_f, ax_s, cb = blocks[outcome]
        boxes = [tight(a) for a in [ax_f] + ax_s + [cb.ax]]
        bb = Bbox.union(boxes)
        block_frame(fig, bb, f'({letter}) {OUT[outcome][0]} — {title_cond}', mask_frac=0.55, pad_x=0.012, extra_top=1.2)
    fig.savefig(f'results/figure_share_{cond}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'results/figure_share_{cond}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


make_figure('hypertension', 'Hypertension')
make_figure('high_cholesterol', 'High cholesterol')
print('wrote results/figure_share_*.pdf/.png')


# ---------------------------------------------------------------------------
# Supplement: observed outcome trajectories by baseline treatment group
# ---------------------------------------------------------------------------
def outcome_figure(cond, title_cond):
    A = d[f'A_{cond}'].astype(int)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, (outcome, key) in zip(axes, (('casp', 'Y_casp'), ('mobility', 'Y_mobility'))):
        Y = d[key]
        for a, cc in ((0, C['untreated']), (1, C['treated'])):
            idx = rng.choice(np.where(A == a)[0], min(150, (A == a).sum()), replace=False)
            ax.plot(months, Y[idx].T, color=cc, lw=0.4, alpha=0.25)
        for a, cc, lab in ((0, C['untreated'], f'untreated (n = {(A == 0).sum()})'), (1, C['treated'], f'treated: {title_cond.lower()} at first interview (n = {A.sum()})')):
            ax.plot(months, Y[A == a].mean(axis=0), color=cc, lw=2.6, label=f'{lab}, mean')
        ax.set_xlim(0, 191); ax.set_xlabel(T_LABEL); ax.set_ylabel(OUT[outcome][0])
        ax.set_title(f'{OUT[outcome][0]} ({OUT[outcome][1]})', loc='left')
        ax.legend(frameon=False, loc='upper left' if outcome == 'mobility' else 'lower left', fontsize=10)
    fig.text(0.01, 0.98, 'Individual trajectories (150 random subjects per group) and group means', fontsize=10.5, color=C['ink2'], va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f'results/figure_share_outcomes_{cond}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'results/figure_share_outcomes_{cond}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


outcome_figure('hypertension', 'Hypertension')
print('wrote results/figure_share_outcomes_hypertension.pdf')
