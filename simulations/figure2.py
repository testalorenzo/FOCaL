#
# Figure 2 of the manuscript (simulation study) and the Supplementary figure with
# the true F-CATE surface.
#
# Panels (a)-(d) use one replication of the base design (seed 72); panels (e)-(f)
# summarize the Monte Carlo results written by simulations.py in results/.
#
# Usage: python figure2.py [--seed 72] [--n_trials_label 1000]
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
from matplotlib import cm, gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # FOCaL.py at the repository root
from simulations import BASE, SCENARIOS, PROFILE_QUANTILES, draw_coefficients, generate_data, fit_focal, true_fcate  # noqa: E402

# scenario styling (same markers/colors as the previous version of the figure)
COLORS = {'Correct specification': 'red', 'Fully misspecified': 'orange',
          'Regression misspecified': 'green', 'Propensity misspecified': 'blue'}
MARKERS = {'Correct specification': 'o', 'Fully misspecified': '^',
           'Regression misspecified': 's', 'Propensity misspecified': 'D'}
SHIFT = {'Correct specification': 0, 'Fully misspecified': 0,
         'Regression misspecified': -2, 'Propensity misspecified': 2}
N_MARKERS = 20

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})


def marker_curve(ax, t, y, label, **kw):
    """Plot a curve by evenly spaced markers (offset per scenario to avoid overlap)."""
    idx = np.clip(np.linspace(0, len(t) - 1, N_MARKERS, dtype=int) + SHIFT[label], 0, len(t) - 1)
    ax.plot(t[idx], y[idx], linestyle='None', marker=MARKERS[label], markersize=6,
            color=COLORS[label], alpha=0.7, label=label, **kw)


def one_replication(cfg, seed):
    """Fit FOCaL under the four scenarios on one data set; return everything the panels need."""
    coef = draw_coefficients(cfg)
    data = generate_data(cfg, coef, seed=seed)
    fits = {name: fit_focal(data, SCENARIOS[name], cfg, seed=seed, fpca=BASE['fpca_nuisance'])[0] for name in SCENARIOS}
    return coef, data, fits


def read_results(path):
    """Read a result table stored as .csv or, in the repository, as .csv.gz."""
    return pd.read_csv(path if os.path.exists(path) else path + '.gz')


def panel_outcomes(ax, data, fits, coef, arm):
    """(a) Observed mean, true mean and DR-recovered mean of the potential outcome in one arm."""
    t = coef['t']
    A = data['A']
    obs = data['Y'][A == arm].mean(axis=0)
    truth = (data['mu1'] if arm == 1 else data['mu0']).mean(axis=0)
    ax.plot(t, obs, color='red' if arm == 1 else 'blue', lw=2, label=f'Observed {"treated" if arm else "untreated"} mean')
    ax.plot(t, truth, color='black', lw=2, alpha=0.5, label='Truth', zorder=15)
    for name, focal in fits.items():
        pseudo = focal._row_order(focal.pseudo1 if arm == 1 else focal.pseudo0)
        marker_curve(ax, t, pseudo.mean(axis=0), name, zorder=10)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\hat{\mathcal{Y}}$')


def panel_densities(fig, gs_cell, data):
    """(b) Densities of the original covariates by treatment group."""
    sub = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_cell, wspace=0.35)
    axes = []
    for j in range(4):
        ax = fig.add_subplot(sub[0, j])
        sns.kdeplot(x=data['X'][data['A'] == 1, j], fill=True, color='red', label='Treated', ax=ax)
        sns.kdeplot(x=data['X'][data['A'] == 0, j], fill=True, color='blue', label='Untreated', ax=ax)
        ax.set_xlabel(f'$X_{j + 1}$')
        ax.set_ylabel('Density' if j == 0 else '')
        axes.append(ax)
    axes[1].legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.18, -0.22))
    return axes


def panel_fate(ax, fits, coef, cfg):
    """(c) Estimated FATE (sample mean of the pseudo-outcome differences) versus the truth."""
    t = coef['t']
    rng = np.random.default_rng(1)
    fate_true = true_fcate(rng.normal(size=(200000, 4 + cfg['p_noise'])), coef, cfg).mean(axis=0)
    ax.plot(t, fate_true, color='black', lw=2, alpha=0.5, label='Truth', zorder=15)
    for name, focal in fits.items():
        marker_curve(ax, t, focal.get_FATE(), name, zorder=10)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\widehat{\mathrm{FATE}}$')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18))


def panel_surface(ax, focal, coef, cfg, truth=False, zlim=None):
    """(d) F-CATE surface along X_1 and time (estimated, or true if truth=True)."""
    x1 = np.linspace(-3, 3, 100)
    X_grid = np.zeros((len(x1), 4 + cfg['p_noise']))
    X_grid[:, 0] = x1
    surf_vals = true_fcate(X_grid, coef, cfg) if truth else focal.predict(X_grid)
    T, X1 = np.meshgrid(coef['t'], x1)
    if zlim is None:
        zlim = (np.floor(surf_vals.min() / 10) * 10, np.ceil(surf_vals.max() / 10) * 10)
    surf = ax.plot_surface(T, X1, surf_vals, cmap=cm.coolwarm, vmin=zlim[0], vmax=zlim[1], linewidth=0, alpha=0.9)
    ax.set_zlim(*zlim)
    ax.set_xlabel('Time')
    ax.set_ylabel('$X_1$')
    ax.set_zlabel(r'$\theta^\star$' if truth else r'$\hat{\theta}$')
    ax.set_box_aspect(None, zoom=1.05)
    return surf, zlim


def block_frame(fig, bbox, title, mask_frac=0.3, pad_x=0.012, pad_y=0.012, extra_top=1.6):
    """Rounded frame around a block of axes with the title centred on its top edge (application-figure style)."""
    x0, y0 = bbox.x0 - pad_x, bbox.y0 - pad_y
    w, h = bbox.width + 2 * pad_x, bbox.height + (1 + extra_top) * pad_y
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.006,rounding_size=0.015',
                                  transform=fig.transFigure, fill=False, lw=1.5, edgecolor='grey', zorder=2))
    tx, ty = x0 + w / 2, y0 + h + 0.003
    fig.add_artist(FancyBboxPatch((tx - mask_frac * w / 2, ty - 0.008), mask_frac * w, 0.018,
                                  boxstyle='round,pad=0.0,rounding_size=0.006', transform=fig.transFigure,
                                  facecolor='white', edgecolor='none', zorder=3))
    fig.text(tx, ty, title, ha='center', va='center', fontsize=15, weight='bold', zorder=4)


def panel_armse(ax, metrics):
    """(e) ARMSE over the Monte Carlo replications, by scenario and method (log scale)."""
    order = list(SCENARIOS)
    # Pointwise DR has the same point estimate as FOCaL: not shown; FoS-interaction is shown in the supplement only
    methods = ['FOCaL', 'Outcome regression', 'IPW']
    sns.boxplot(data=metrics, x='scenario', y='ARMSE', hue='method', order=order, hue_order=methods,
                showfliers=False, width=0.7, ax=ax, palette='muted')
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace(' ', '\n') for s in order])
    ax.legend(frameon=False, title='', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18))


def panel_coverage(ax, coverage, alpha):
    """(f) Coverage of the 95% L2 confidence set (FOCaL) by scenario and profile."""
    order = list(SCENARIOS)
    profiles = [f'X1 q{int(100 * q)}' for q in PROFILE_QUANTILES] + ['FATE']
    cov = coverage[coverage.method == 'FOCaL'].groupby(['scenario', 'profile'])['cover_L2'].mean().reset_index()
    # readable legend labels: which target the confidence set is for
    labels = {f'X1 q{int(100 * q)}': f'F-CATE at $X_1 = q_{{{q:.2f}}}$ (other $X_j = 0$)' for q in PROFILE_QUANTILES}
    labels['FATE'] = 'FATE'
    cov['target'] = cov['profile'].map(labels)
    sns.barplot(data=cov, x='scenario', y='cover_L2', hue='target', order=order,
                hue_order=[labels[p] for p in profiles], ax=ax, palette='crest')
    ax.axhline(1 - alpha, color='black', ls='--', lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Empirical coverage of the true curve')
    ax.set_xlabel('')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace(' ', '\n') for s in order])
    ax.legend(frameon=False, ncol=2, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.18))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()
    cfg = dict(BASE)

    coef, data, fits = one_replication(cfg, args.seed)
    have_results = all(os.path.exists(f'{args.out}/{f}.csv') or os.path.exists(f'{args.out}/{f}.csv.gz') for f in ('metrics_base', 'coverage_base'))
    if have_results:
        metrics = read_results(f'{args.out}/metrics_base.csv')
        metrics['method'] = metrics['method'].replace({'Plug-in': 'Outcome regression'})   # name used in the paper
        coverage = read_results(f'{args.out}/coverage_base.csv')

    fig = plt.figure(figsize=(20, 25))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 0.8, 1, 1.3], hspace=1.0, wspace=0.3)

    # (a) outcome prediction
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    panel_outcomes(ax_a1, data, fits, coef, arm=1)
    panel_outcomes(ax_a2, data, fits, coef, arm=0)
    handles, labels = ax_a1.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    handles.insert(1, h2[0])
    labels.insert(1, l2[0])
    ax_a1.legend(handles, labels, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(1.05, -0.18))
    ax_a1.set_title('Treated', loc='center')
    ax_a2.set_title('Untreated', loc='center')

    # (b) covariate densities
    axes_b = panel_densities(fig, gs[1, :], data)

    # (c) FATE over time
    ax_c = fig.add_subplot(gs[2, 0])
    panel_fate(ax_c, fits, coef, cfg)

    # (e) ARMSE boxplots
    ax_e = fig.add_subplot(gs[2, 1])
    if have_results:
        panel_armse(ax_e, metrics)

    # (d) estimated F-CATE surface (correct specification)
    ax_d = fig.add_subplot(gs[3, 0], projection='3d')
    surf, zlim = panel_surface(ax_d, fits['Correct specification'], coef, cfg)
    cbar_d = fig.colorbar(surf, ax=ax_d, shrink=0.5, aspect=12, pad=0.1, label=r'$\hat{\theta}$')

    # (f) coverage
    ax_f = fig.add_subplot(gs[3, 1])
    if have_results:
        panel_coverage(ax_f, coverage, cfg['alpha'])

    # framed blocks with centred titles, as in the application figures
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tight = lambda a: a.get_tightbbox(rend).transformed(fig.transFigure.inverted())
    # close the gap between row (b) and rows (c)-(f): move the lower axes up so that the gap equals the (a)-(b) gap
    gap_ab = tight(ax_a1).y0 - max(tight(a).y1 for a in axes_b)
    gap_bc = min(tight(a).y1 for a in axes_b) - max(tight(ax_c).y1, tight(ax_e).y1)
    gap_bc = min(tight(a).y0 for a in axes_b) - max(tight(ax_c).y1, tight(ax_e).y1)
    shift = gap_bc - gap_ab
    for a in (ax_c, ax_e, ax_d, cbar_d.ax, ax_f):
        p = a.get_position(); a.set_position([p.x0, p.y0 + shift, p.width, p.height])
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    block_frame(fig, Bbox.union([tight(ax_a1), tight(ax_a2)]), '(a) Outcome prediction', mask_frac=0.25)
    block_frame(fig, Bbox.union([tight(a) for a in axes_b]), '(b) Covariate densities', mask_frac=0.25)
    block_frame(fig, tight(ax_c), '(c) Estimated FATE over time', mask_frac=0.55)
    block_frame(fig, tight(ax_e), '(e) ARMSE under different model specifications', mask_frac=0.8)
    bc, bf = tight(ax_c), tight(ax_f)
    bd = Bbox([[bc.x0, bf.y0], [bc.x1, bf.y1]])   # same width as (c), same height as (f)
    block_frame(fig, bd, '(d) Estimated F-CATE surface along $X_1$ and time', mask_frac=0.8)
    block_frame(fig, tight(ax_f), '(f) Coverage of the 95% $\\mathbb{L}_2$ confidence sets', mask_frac=0.8)

    os.makedirs(args.out, exist_ok=True)
    fig.savefig(f'{args.out}/figure2.pdf', bbox_inches='tight')
    fig.savefig(f'{args.out}/figure2.png', dpi=110, bbox_inches='tight')

    # Supplementary figure: true F-CATE surface on the same vertical scale
    fig_s = plt.figure(figsize=(8, 6))
    ax_s = fig_s.add_subplot(111, projection='3d')
    surf_s, _ = panel_surface(ax_s, None, coef, cfg, truth=True, zlim=zlim)
    fig_s.colorbar(surf_s, ax=ax_s, shrink=0.5, aspect=12, pad=0.1, label=r'$\theta^\star$')
    fig_s.savefig(f'{args.out}/figure_true_surface.pdf', bbox_inches='tight')
    print('saved', f'{args.out}/figure2.pdf', f'{args.out}/figure_true_surface.pdf')


if __name__ == '__main__':
    main()
