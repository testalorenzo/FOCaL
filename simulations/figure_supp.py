#
# Supplementary figures for the simulation study.
#
#   figure_S_samplesize.pdf : error, L2-set coverage and L2 radius as functions of n
#                             (designs n=500, 1000, 2000 from the variations run and
#                             the base design n=5000)
#   figure_S_variations.pdf : error and coverage across the one-factor design variations
#   figure_S_diagnostics.pdf: nuisance errors, runtime and L2 radius (base design)
#   figure_S_pointwise.pdf  : FOCaL (functional nuisances) against the scalar DR-learner applied
#                             pointwise and basis-wise, with gradient-boosting nuisances
#                             (variation 'ML nuisances'): error, runtime, L2 radius, L2 coverage
#
# Inference is reported for the L2 confidence sets only (the grid bands stored in the
# coverage CSVs are not plotted).
#
# Requires results/metrics_base.csv, coverage_base.csv, metrics_variations.csv,
# coverage_variations.csv written by simulations.py.
#
# Usage: python figure_supp.py [--out results]
#

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # FOCaL.py at the repository root
from simulations import SCENARIOS  # noqa: E402

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

METHOD_COLORS = {'FOCaL': 'C0', 'Outcome regression': 'C2', 'FoS-interaction': 'C3', 'IPW': 'C4',
                 'Pointwise DR': 'C1', 'Basis-wise DR': 'C5'}
SCEN_STYLE = {'Correct specification': '-', 'Regression misspecified': '--',
              'Propensity misspecified': '-.', 'Fully misspecified': ':'}
SCEN_SHORT = {'Correct specification': 'correct', 'Regression misspecified': 'regression miss.',
              'Propensity misspecified': 'propensity miss.', 'Fully misspecified': 'both miss.'}
VARIATION_ORDER = ['base', 'overlap x0.5', 'overlap x2', 'p=10', 'nu=1.5', 'separate arms',
                   'heavy tails', 'sparse noisy', 'centered effect', 'localized effect',
                   'sign-changing effect', 'nonlinear modifier', 'nonlinear modifier + splines',
                   'nonlinear modifier + splines (n=2000)']
VARIATION_LABEL = {'base': 'base', 'overlap x0.5': 'overlap\n×0.5', 'overlap x2': 'overlap\n×2', 'p=10': 'p = 10',
                   'nu=1.5': 'ν = 1.5', 'separate arms': 'separate\narms', 'heavy tails': 'heavy\ntails',
                   'sparse noisy': 'sparse\nnoisy', 'centered effect': 'centered\neffect',
                   'localized effect': 'localized\neffect', 'sign-changing effect': 'sign-\nchanging',
                   'nonlinear modifier': 'nonlin.\nmod.', 'nonlinear modifier + splines': 'nonlin.\n+ spl.',
                   'nonlinear modifier + splines (n=2000)': 'nonlin.\n+ spl.\nn = 2000'}


def load(out):
    rd = lambda p: pd.read_csv(p if os.path.exists(p) else p + '.gz')   # tables are gzipped in the repository
    mb, cb = rd(f'{out}/metrics_base.csv'), rd(f'{out}/coverage_base.csv')
    mv, cv = rd(f'{out}/metrics_variations.csv'), rd(f'{out}/coverage_variations.csv')
    for df in (mb, cb, mv, cv):   # name used in the paper
        df['method'] = df['method'].replace({'Plug-in': 'Outcome regression'})
    mb['design'], cb['design'] = 'base', 'base'
    m, c = pd.concat([mb, mv], ignore_index=True), pd.concat([cb, cv], ignore_index=True)
    n_of = {'n=500': 500, 'n=1000': 1000, 'n=2000': 2000, 'base': 5000}
    m['n'], c['n'] = m.design.map(n_of), c.design.map(n_of)
    c['kind'] = np.where(c.profile == 'FATE', 'FATE', 'F-CATE')
    return m, c


# ---------------------------------------------------------------------------
# Figure S1: sample size
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Framed blocks with centred titles, as in the application figures
# ---------------------------------------------------------------------------
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402


def block_frame(fig, bbox, title, pad_x=0.012, pad_y=0.012, extra_top=1.6):
    """Rounded frame around a block of axes with the title centred on its top edge."""
    rend = fig.canvas.get_renderer()
    x0, y0 = bbox.x0 - pad_x, bbox.y0 - pad_y
    w, h = bbox.width + 2 * pad_x, bbox.height + (1 + extra_top) * pad_y
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.006,rounding_size=0.015',
                                  transform=fig.transFigure, fill=False, lw=1.5, edgecolor='grey', zorder=2))
    tx, ty = x0 + w / 2, y0 + h + 0.003
    t = fig.text(tx, ty, title, ha='center', va='center', fontsize=14, weight='bold', zorder=4)
    tb = t.get_window_extent(rend).transformed(fig.transFigure.inverted())
    fig.add_artist(FancyBboxPatch((tb.x0 - 0.006, ty - 0.009), tb.width + 0.012, 0.018,
                                  boxstyle='round,pad=0.0,rounding_size=0.006', transform=fig.transFigure,
                                  facecolor='white', edgecolor='none', zorder=3))


def frame_all(fig, axes, legend_ncol=3, legend_dy=-0.2, h_pad=6.0, w_pad=7.0):
    """
    Turn each axis title into a frame title, move each legend below its axis, lay the figure out
    and draw one frame per axis (twin axes and colorbars sharing the position are included).
    """
    axes = [a for a in np.ravel(axes)]
    titles = []
    for ax in axes:
        titles.append(ax.get_title(loc='left') or ax.get_title())
        ax.set_title('', loc='left'); ax.set_title('', loc='center')
        leg = ax.get_legend()
        if leg is not None:
            handles = getattr(leg, 'legend_handles', None) or leg.legendHandles
            labels = [t.get_text() for t in leg.get_texts()]
            size = leg.get_texts()[0].get_fontsize() if leg.get_texts() else None
            leg.remove()
            ax.legend(handles, labels, frameon=False, ncol=min(len(labels), legend_ncol), loc='upper center',
                      bbox_to_anchor=(0.5, legend_dy), fontsize=size)
    fig.tight_layout(h_pad=h_pad, w_pad=w_pad)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tight = lambda a: a.get_tightbbox(rend).transformed(fig.transFigure.inverted())
    boxes = []
    for ax in axes:
        pos = ax.get_position()
        twins = [a for a in fig.axes if a is not ax and a.get_position().overlaps(pos) and a not in axes]
        boxes.append(Bbox.union([tight(ax)] + [tight(a) for a in twins]))
    # frames on the same row share the same vertical extent
    rows = [round(ax.get_position().y0, 2) for ax in axes]
    for ax, title, bb, r in zip(axes, titles, boxes, rows):
        same = [b_ for b_, r_ in zip(boxes, rows) if r_ == r]
        y0, y1 = min(b_.y0 for b_ in same), max(b_.y1 for b_ in same)
        block_frame(fig, Bbox([[bb.x0, y0], [bb.x1, y1]]), title)


def figure_samplesize(m, c, out):
    ms, cs = m[m.n.notna()], c[(c.n.notna()) & (c.method == 'FOCaL')]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))

    # (a) ARMSE versus n, log-log, with an n^{-1/2} reference
    ax = axes[0]
    # the outcome regression and the FoS regression coincide under correct specification (same linear model):
    # draw the former thick underneath and the latter thin with cross markers on top so that both stay visible
    style = {'FOCaL': dict(lw=2, marker='o', ms=7), 'Outcome regression': dict(lw=3.5, marker='o', ms=9, alpha=0.8),
             'FoS-interaction': dict(lw=1.2, marker='x', ms=7)}
    for method in ['FOCaL', 'Outcome regression', 'FoS-interaction']:
        for scen in ['Correct specification', 'Regression misspecified']:
            d = ms[(ms.method == method) & (ms.scenario == scen)].groupby('n').ARMSE.mean()
            ax.plot(d.index, d.values, ls=SCEN_STYLE[scen], color=METHOD_COLORS[method],
                    label=f'{method}, {SCEN_SHORT[scen]}', **style[method])
    ref = ms[(ms.method == 'FOCaL') & (ms.scenario == 'Regression misspecified')].groupby('n').ARMSE.mean()
    ax.plot(ref.index, ref.iloc[0] * np.sqrt(ref.index[0] / ref.index), color='grey', lw=1, ls='-',
            alpha=0.6, label=r'$n^{-1/2}$ reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$n$')
    ax.set_ylabel('ARMSE')
    ax.set_title('(a) Estimation error versus $n$', loc='left')
    ax.legend(frameon=False, fontsize=9, ncol=1)

    # (b) coverage of the L2 set versus n, F-CATE profiles (averaged) and FATE
    ax = axes[1]
    for scen, color in [('Correct specification', 'black'), ('Regression misspecified', 'C0')]:
        for kind, mk, ls in [('F-CATE', 'o', '-'), ('FATE', 's', '--')]:
            d = cs[(cs.scenario == scen) & (cs.kind == kind)].groupby('n')['cover_L2'].mean()
            ax.plot(d.index, d.values, marker=mk, ls=ls, color=color,
                    label=f'{kind}, {SCEN_SHORT[scen]}')
    ax.axhline(0.95, color='grey', lw=1, ls='-', alpha=0.6)
    ax.set_xscale('log')
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$\\mathbb{L}_2$-set coverage (nominal 0.95)')
    ax.set_title('(b) Coverage versus $n$', loc='left')
    ax.legend(frameon=False, fontsize=9, ncol=1, loc='lower right')

    # (c) L2 radius versus n, correct specification
    ax = axes[2]
    d = cs[cs.scenario == 'Correct specification'].groupby(['kind', 'n'])['radius'].mean()
    for kind, mk, ls in [('F-CATE', 'o', '-'), ('FATE', 's', '--')]:
        ax.plot(d[kind].index, d[kind].values, marker=mk, ls=ls, color='black', label=f'{kind}')
    ref = d['F-CATE']
    ax.plot(ref.index, ref.iloc[0] * np.sqrt(ref.index[0] / ref.index), color='grey', lw=1, alpha=0.6,
            label=r'$n^{-1/2}$ reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$\\mathbb{L}_2$ radius $\\hat c_\\alpha r_n$')
    ax.set_title('(c) Radius of the $\\mathbb{L}_2$ sets versus $n$', loc='left')
    ax.legend(frameon=False, fontsize=9)

    frame_all(fig, axes, legend_ncol=2, legend_dy=-0.2)
    fig.savefig(f'{out}/figure_S_samplesize.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/figure_S_samplesize.png', dpi=110, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Figure S2: design variations
# ---------------------------------------------------------------------------
def figure_variations(m, c, out):
    """Figure S2: ARMSE (left) and coverage (right) across the design variations, one row per scenario."""
    global VARIATION_ORDER
    VARIATION_ORDER = [v for v in VARIATION_ORDER if v in set(m.design)]   # skip designs not run
    mv = m[m.design.isin(VARIATION_ORDER)]
    cv = c[(c.design.isin(VARIATION_ORDER)) & (c.method == 'FOCaL')]
    methods = ['FOCaL', 'Outcome regression', 'FoS-interaction']  # with linear nuisances the pointwise DR-learner coincides with FOCaL: not shown
    fig, axes = plt.subplots(4, 2, figsize=(24, 23))
    letters = iter('abcdefgh')
    for r, scen in enumerate(SCENARIOS):
        # left: ARMSE by design
        ax = axes[r, 0]
        d = mv[(mv.scenario == scen) & (mv.method.isin(methods))]
        sns.boxplot(data=d, x='design', y='ARMSE', hue='method', order=VARIATION_ORDER, hue_order=methods,
                    showfliers=False, width=0.75, ax=ax, palette=[METHOD_COLORS[k] for k in methods])
        ax.set_yscale('log')
        ax.set_xlabel('')
        ax.set_xticks(range(len(VARIATION_ORDER)))
        ax.set_xticklabels([VARIATION_LABEL[v] for v in VARIATION_ORDER])
        ax.set_title(f'({next(letters)}) ARMSE, {SCEN_SHORT[scen]}', loc='left')
        ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))

        # right: coverage of the L2 set by design
        ax = axes[r, 1]
        d = cv[cv.scenario == scen]
        bars = d.groupby(['design', 'kind'])['cover_L2'].mean().reset_index()
        kinds = {'F-CATE': 'F-CATE profiles (mean over $q_{0.10}, q_{0.50}, q_{0.90}$)', 'FATE': 'FATE'}
        bars['kind'] = bars['kind'].map(kinds)
        sns.barplot(data=bars, x='design', y='cover_L2', hue='kind', order=VARIATION_ORDER,
                    hue_order=list(kinds.values()), ax=ax, palette='crest')
        # sparse design: L2-set coverage of the reconstruction-level estimand (stars, one per bar)
        if 'cover_L2_rec' in d.columns and 'sparse noisy' in VARIATION_ORDER:
            rec = d[d.design == 'sparse noisy'].groupby('kind')['cover_L2_rec'].mean()
            i = VARIATION_ORDER.index('sparse noisy')
            for j, kind in enumerate(['F-CATE', 'FATE']):
                ax.plot(i - 0.2 + 0.4 * j, rec[kind], marker='*', mfc='gold', mec='black', ms=13, ls='None',
                        label='$\\mathbb{L}_2$ set, effect on reconstructed curves' if j == 0 else None)
        ax.axhline(0.95, color='black', ls='--', lw=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('')
        ax.set_xticks(range(len(VARIATION_ORDER)))
        ax.set_xticklabels([VARIATION_LABEL[v] for v in VARIATION_ORDER])
        ax.set_ylabel('$\\mathbb{L}_2$-set coverage')
        ax.set_title(f'({next(letters)}) Coverage of the $\\mathbb{{L}}_2$ sets, {SCEN_SHORT[scen]}', loc='left')
        ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10)
    frame_all(fig, axes, legend_ncol=3, legend_dy=-0.28, h_pad=8.0)
    fig.savefig(f'{out}/figure_S_variations.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/figure_S_variations.png', dpi=110, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Figure S3: nuisance diagnostics, runtime and size of the confidence sets (base design)
# ---------------------------------------------------------------------------
def figure_diagnostics(m, c, out):
    mb, cb = m[m.design == 'base'], c[(c.design == 'base') & (c.method == 'FOCaL')]
    order = list(SCENARIOS)
    short = [SCEN_SHORT[s].replace(' ', '\n') for s in order]
    fig, axes = plt.subplots(1, 4, figsize=(27, 5.6))

    # (a) nuisance quality: outcome-regression L2 error and propensity absolute error (cross-fitted)
    f = mb[mb.method == 'FOCaL'].groupby('scenario')[['mu_rmse', 'pi_mae']].mean().loc[order]
    ax = axes[0]
    ax.bar(np.arange(4) - 0.2, f.mu_rmse, width=0.4, color='C0', label=r'$\hat\mu^{(a)}$: mean $\mathbb{L}_2$ error')
    ax.set_ylabel(r'$\mathbb{L}_2$ error of $\hat\mu^{(a)}$', color='C0')
    ax2 = ax.twinx()
    ax2.bar(np.arange(4) + 0.2, f.pi_mae, width=0.4, color='C3', label=r'$\hat\pi$: mean absolute error')
    ax2.set_ylabel(r'absolute error of $\hat\pi$', color='C3')
    ax.set_xticks(range(4))
    ax.set_xticklabels(short)
    ax.set_title('(a) Nuisance estimation error', loc='left')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper left', fontsize=10)

    # (b) runtime per replication: fitting by method, inference for FOCaL
    rt = mb.groupby(['scenario', 'method'])[['runtime_fit', 'runtime_inference']].mean().reset_index()
    ax = axes[1]
    methods = ['FOCaL', 'Outcome regression', 'FoS-interaction', 'IPW']
    w = 0.8 / (len(methods) + 1)
    for k, meth in enumerate(methods):
        d = rt[rt.method == meth].set_index('scenario').loc[order]
        ax.bar(np.arange(4) - 0.4 + w * (k + 0.5), d.runtime_fit, width=w, color=METHOD_COLORS[meth], label=f'{meth}: fit')
    d = rt[rt.method == 'FOCaL'].set_index('scenario').loc[order]
    ax.bar(np.arange(4) - 0.4 + w * (len(methods) + 0.5), d.runtime_inference, width=w, color='grey',
           label='FOCaL: inference')
    ax.set_xticks(range(4))
    ax.set_xticklabels(short)
    ax.set_ylabel('seconds per replication')
    ax.set_title('(b) Runtime', loc='left')
    ax.legend(frameon=False, fontsize=9, ncol=2)

    # (c) mean L2 radius by scenario and target
    sz = cb.groupby(['scenario', 'profile'])['radius'].mean().unstack().loc[order]
    ax = axes[2]
    cols = [(f'X1 q{int(100 * q)}', f'$X_1 = q_{{{q:.2f}}}$') for q in (0.1, 0.5, 0.9)] + [('FATE', 'FATE')]
    w = 0.8 / len(cols)
    for k, (col, lab) in enumerate(cols):
        ax.bar(np.arange(4) - 0.4 + w * (k + 0.5), sz[col], width=w, label=lab, color=plt.cm.Greys(0.35 + 0.2 * k))
    ax.set_yscale('log')
    ax.set_xticks(range(4))
    ax.set_xticklabels(short)
    ax.set_ylabel('$\\mathbb{L}_2$ radius (log scale)')
    ax.set_title('(c) Radius of the $\\mathbb{L}_2$ sets', loc='left')
    ax.legend(frameon=False, fontsize=9)

    # (d) distribution of the L2 radius across replications, by target, correct specification
    ax = axes[3]
    d = cb[cb.scenario == 'Correct specification']
    labels = {f'X1 q{int(100 * q)}': f'$X_1 = q_{{{q:.2f}}}$' for q in (0.1, 0.5, 0.9)}
    labels['FATE'] = 'FATE'
    sns.boxplot(data=d.assign(target=d.profile.map(labels)), x='target', y='radius',
                order=list(labels.values()), showfliers=False, width=0.5, ax=ax, color='lightsteelblue')
    ax.set_xlabel('')
    ax.set_ylabel('$\\mathbb{L}_2$ radius')
    ax.set_title('(d) Radius across replications, correct specification', loc='left')

    frame_all(fig, axes, legend_ncol=2, legend_dy=-0.3)
    fig.savefig(f'{out}/figure_S_diagnostics.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/figure_S_diagnostics.png', dpi=110, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Figure S4: functional vs pointwise vs basis-wise DR-learner with boosting nuisances
# ---------------------------------------------------------------------------
def figure_pointwise(m, c, out):
    """Variation 'ML nuisances' (n = 2000, gradient boosting): the three DR-learners side by side."""
    d = m[m.design == 'ML nuisances']
    if d.empty:
        return
    order = list(SCENARIOS)
    short = [SCEN_SHORT[k] for k in order]
    methods = ['FOCaL', 'Pointwise DR', 'Basis-wise DR', 'Outcome regression']
    fig, axes = plt.subplots(1, 4, figsize=(27, 5.4))
    # (a) ARMSE
    ax = axes[0]
    sns.boxplot(data=d[d.method.isin(methods)], x='scenario', y='ARMSE', hue='method', order=order, hue_order=methods,
                showfliers=False, width=0.75, ax=ax, palette=[METHOD_COLORS[k] for k in methods])
    ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_xlabel('')
    ax.set_title('(a) ARMSE with boosting nuisances', loc='left'); ax.legend(frameon=False, fontsize=9)
    # (b) runtime of the fit
    ax = axes[1]
    rt = d.groupby(['scenario', 'method'])['runtime_fit'].mean().unstack().loc[order]
    w = 0.8 / 3
    for k, meth in enumerate(['FOCaL', 'Pointwise DR', 'Basis-wise DR']):
        ax.bar(np.arange(4) - 0.4 + w * (k + 0.5), rt[meth], width=w, color=METHOD_COLORS[meth], label=meth)
    ax.set_yscale('log'); ax.set_xticks(range(4)); ax.set_xticklabels(short)
    ax.set_ylabel('seconds per replication (log scale)')
    ax.set_title('(b) Fitting time: 10 vs 100 boosters per arm and fold', loc='left'); ax.legend(frameon=False, fontsize=9)
    # (c)-(d) L2 sets of FOCaL and of the pointwise DR-learner (the basis-wise DR-learner has no L2 set)
    cc = c[(c.design == 'ML nuisances') & (c.method.isin(['FOCaL', 'Pointwise DR']))]
    w = 0.8 / 2
    for ax, col, ylab, title in [(axes[2], 'radius', '$\\mathbb{L}_2$ radius',
                                  '(c) Radius of the $\\mathbb{L}_2$ sets at $X_1 = q_{0.50}$'),
                                 (axes[3], 'cover_L2', '$\\mathbb{L}_2$-set coverage',
                                  '(d) $\\mathbb{L}_2$-set coverage (mean over the four targets)')]:
        d_c = cc[cc.profile == 'X1 q50'] if col == 'radius' else cc
        for k, meth in enumerate(['FOCaL', 'Pointwise DR']):
            v = d_c[d_c.method == meth].groupby('scenario')[col].mean().loc[order]
            ax.bar(np.arange(4) - 0.4 + w * (k + 0.5), v.values, width=w, color=METHOD_COLORS[meth], label=meth)
        ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_ylabel(ylab)
        ax.set_title(title, loc='left'); ax.legend(frameon=False, fontsize=9)
    axes[3].axhline(0.95, color='black', ls='--', lw=1); axes[3].set_ylim(0, 1.05)
    axes[3].legend(frameon=False, fontsize=9, loc='lower right')
    frame_all(fig, axes, legend_ncol=2, legend_dy=-0.3)
    fig.savefig(f'{out}/figure_S_pointwise.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/figure_S_pointwise.png', dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m, c = load(args.out)
    figure_samplesize(m, c, args.out)
    figure_variations(m, c, args.out)
    figure_diagnostics(m, c, args.out)
    figure_pointwise(m, c, args.out)
    print('saved', f'{args.out}/figure_S_samplesize.pdf', f'{args.out}/figure_S_variations.pdf', f'{args.out}/figure_S_diagnostics.pdf')
