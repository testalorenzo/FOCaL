#
# Figure 2 of the manuscript (simulation study) and the Supplementary figure with
# the true F-CATE surface.
#
# Panels (a)-(d) use one replication of the base design (seed 72); panels (e)-(f)
# summarize the Monte Carlo results written by simulations.py in results/.
#
# Usage: python figure2.py [--seed 72] [--out DIR]
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

# colors of the methods in panel (e)
METHOD_COLORS = {'FOCaL': '#C52C2C', 'Outcome regression': '#3A7DC0', 'IPW': '#57AF41'}

GRID_KW = dict(alpha=0.35, lw=0.8)

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 21, 'axes.labelsize': 19,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})


def add_grid(ax):
    """Light grid behind the data."""
    ax.set_axisbelow(True)
    ax.grid(True, which='major', **GRID_KW)


def marker_curve(ax, t, y, label, **kw):
    """Plot a curve by evenly spaced markers (offset per scenario to avoid overlap)."""
    idx = np.clip(np.linspace(0, len(t) - 1, N_MARKERS, dtype=int) + SHIFT[label], 0, len(t) - 1)
    ax.plot(t[idx], y[idx], linestyle='None', marker=MARKERS[label], markersize=8,
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
    ax.plot(t, obs, color='red' if arm == 1 else 'blue', lw=2.5, label=f'Observed {"treated" if arm else "untreated"} mean')
    ax.plot(t, truth, color='black', lw=2.5, alpha=0.5, label='Truth', zorder=15)
    for name, focal in fits.items():
        pseudo = focal._row_order(focal.pseudo1 if arm == 1 else focal.pseudo0)
        y_mean = pseudo.mean(axis=0)
        # solid, semi-transparent line through the whole curve, in the color of the markers (as in panel c)
        ax.plot(t, y_mean, color=COLORS[name], lw=2, alpha=0.4, zorder=8)
        marker_curve(ax, t, y_mean, name, zorder=10)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\hat{\mathcal{Y}}$')
    add_grid(ax)


def panel_densities(fig, gs_cell, data):
    """(b) Densities of the original covariates by treatment group."""
    sub = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_cell, wspace=0.45)
    axes = []
    for j in range(4):
        ax = fig.add_subplot(sub[0, j])
        sns.kdeplot(x=data['X'][data['A'] == 1, j], fill=True, color='red', label='Treated', ax=ax)
        sns.kdeplot(x=data['X'][data['A'] == 0, j], fill=True, color='blue', label='Untreated', ax=ax)
        ax.set_xlabel(f'$X_{j + 1}$')
        ax.set_ylabel('Density' if j == 0 else '')
        add_grid(ax)
        axes.append(ax)
    axes[1].legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.18, -0.28))
    return axes


def panel_fate(ax, fits, coef, cfg):
    """(c) Estimated FATE (sample mean of the pseudo-outcome differences) versus the truth."""
    t = coef['t']
    rng = np.random.default_rng(1)
    fate_true = true_fcate(rng.normal(size=(200000, 4 + cfg['p_noise'])), coef, cfg).mean(axis=0)
    ax.plot(t, fate_true, color='black', lw=2.5, alpha=0.5, label='Truth', zorder=15)
    for name, focal in fits.items():
        # solid, semi-transparent line through the whole curve, in the color of the markers
        ax.plot(t, focal.get_FATE(), color=COLORS[name], lw=2, alpha=0.4, zorder=8)
        marker_curve(ax, t, focal.get_FATE(), name, zorder=10)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\widehat{\mathrm{FATE}}$', labelpad=-3)
    add_grid(ax)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))


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
    ax.set_xlabel('Time', labelpad=12)
    ax.set_ylabel('$X_1$', labelpad=12)
    ax.set_zlabel(r'$\theta^\star$' if truth else r'$\hat{\theta}$', labelpad=5)
    ax.tick_params(axis='both', which='major', pad=4)
    # wider, larger surface: stretch the time axis and zoom into the box
    ax.set_box_aspect((8.5, 8.5, 6.0), zoom=1.2)
    return surf, zlim


def block_frame(fig, bbox, title, mask_frac=0.3, pad_x=0.012, pad_y=0.012, extra_top=1.6):
    """Rounded frame around a block of axes with the title centred on its top edge (application-figure style)."""
    x0, y0 = bbox.x0 - pad_x, bbox.y0 - pad_y
    w, h = bbox.width + 2 * pad_x, bbox.height + (1 + extra_top) * pad_y
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.006,rounding_size=0.015',
                                  transform=fig.transFigure, fill=False, lw=1.8, edgecolor='grey', zorder=2))
    tx, ty = x0 + w / 2, y0 + h + 0.003
    fig.add_artist(FancyBboxPatch((tx - mask_frac * w / 2, ty - 0.009), mask_frac * w, 0.020,
                                  boxstyle='round,pad=0.0,rounding_size=0.006', transform=fig.transFigure,
                                  facecolor='white', edgecolor='none', zorder=3))
    fig.text(tx, ty, title, ha='center', va='center', fontsize=21, weight='bold', zorder=4)


def panel_armse(ax, metrics):
    """(e) ARMSE over the Monte Carlo replications, by scenario and method (log scale)."""
    order = list(SCENARIOS)
    # Pointwise DR has the same point estimate as FOCaL: not shown; FoS-interaction is shown in the supplement only
    methods = ['FOCaL', 'Outcome regression', 'IPW']
    sns.boxplot(data=metrics, x='scenario', y='ARMSE', hue='method', order=order, hue_order=methods,
                showfliers=False, width=0.7, linewidth=1.6, ax=ax, palette=METHOD_COLORS)
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace(' ', '\n') for s in order])
    add_grid(ax)
    ax.grid(True, which='minor', axis='y', alpha=0.15, lw=0.6)
    ax.legend(frameon=False, title='', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3))


def panel_coverage(ax, coverage, alpha):
    """(f) Coverage of the 95% L2 confidence set (FOCaL) by scenario and target."""
    order = list(SCENARIOS)
    profiles = [f'X1 q{int(100 * q)}' for q in PROFILE_QUANTILES] + ['FATE']
    cov = coverage[coverage.method == 'FOCaL'].groupby(['scenario', 'profile'])['cover_L2'].mean().reset_index()
    n_trials = coverage['trial'].nunique()
    # readable legend labels: which target the confidence set is for
    labels = {f'X1 q{int(100 * q)}': f'F-CATE at $X_1 = q_{{{q:.2f}}}$' for q in PROFILE_QUANTILES}
    labels['FATE'] = 'FATE'
    cov['target'] = cov['profile'].map(labels)
    sns.barplot(data=cov, x='scenario', y='cover_L2', hue='target', order=order,
                hue_order=[labels[p] for p in profiles], ax=ax,
                palette=sns.color_palette('viridis', len(profiles)),
                edgecolor='white', linewidth=1.5, saturation=1, width=0.8)
    # nominal level with a band for the Monte Carlo error of an empirical coverage
    mc = 1.96 * np.sqrt(alpha * (1 - alpha) / n_trials)
    ax.axhspan(1 - alpha - mc, 1 - alpha + mc, color='black', alpha=0.10, zorder=0, lw=0)
    ax.axhline(1 - alpha, color='black', ls='--', lw=1.8, zorder=1, label=f'Nominal {100 * (1 - alpha):.0f}%')
    for cont in ax.containers:
        ax.bar_label(cont, fmt='%.2f', padding=4, fontsize=13, rotation=90)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Empirical coverage')
    ax.set_xlabel('')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace(' ', '\n') for s in order])
    add_grid(ax)
    ax.grid(False, axis='x')
    sns.despine(ax=ax)
    ax.legend(frameon=False, ncol=2, fontsize=15, loc='upper center', bbox_to_anchor=(0.5, -0.2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--out', type=str, default=None, help='results directory (default: <script dir>/results)')
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    args.out = os.path.abspath(args.out) if args.out else os.path.join(here, 'results')
    os.makedirs(args.out, exist_ok=True)      # fail early, before the expensive fit
    cfg = dict(BASE)

    coef, data, fits = one_replication(cfg, args.seed)
    have_results = all(os.path.exists(f'{args.out}/{f}.csv') or os.path.exists(f'{args.out}/{f}.csv.gz') for f in ('metrics_base', 'coverage_base'))
    if have_results:
        metrics = read_results(f'{args.out}/metrics_base.csv')
        metrics['method'] = metrics['method'].replace({'Plug-in': 'Outcome regression'})   # name used in the paper
        coverage = read_results(f'{args.out}/coverage_base.csv')
    else:
        print(f'WARNING: no metrics_base/coverage_base in {args.out}; panels (e)-(f) will be empty')

    fig = plt.figure(figsize=(22, 28))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 0.8, 1, 1.3], hspace=1.1, wspace=0.3)

    # (a) outcome prediction
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    panel_outcomes(ax_a1, data, fits, coef, arm=1)
    panel_outcomes(ax_a2, data, fits, coef, arm=0)
    handles, labels = ax_a1.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    handles.insert(1, h2[0])
    labels.insert(1, l2[0])
    ax_a1.legend(handles, labels, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(1.05, -0.2))
    ax_a1.set_title('Treated', loc='center')
    ax_a2.set_title('Untreated', loc='center')

    # (b) covariate densities
    axes_b = panel_densities(fig, gs[1, :], data)
    dx, dy = -0.005, 0.0     # in figure fractions
    for a in axes_b:
        p = a.get_position()
        a.set_position([p.x0 + dx, p.y0 + dy, p.width, p.height])

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
    cbar_d = fig.colorbar(surf, ax=ax_d, shrink=0.5, aspect=12, pad=0.28)
    cbar_d.set_label(r'$\hat{\theta}$', labelpad=-15)    
    p = ax_d.get_position()
    ax_d.set_position([p.x0 - 0.065, p.y0 - 0.045, p.width * 1.60, p.height * 1.4])



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
    gap_bc = min(tight(a).y0 for a in axes_b) - max(tight(ax_c).y1, tight(ax_e).y1)
    shift = gap_bc - gap_ab
    for a in (ax_c, ax_e, ax_d, cbar_d.ax, ax_f):
        p = a.get_position(); a.set_position([p.x0, p.y0 + shift, p.width, p.height])
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    ba1, ba2 = tight(ax_a1), tight(ax_a2)
    bb = Bbox.union([tight(a) for a in axes_b])
    bc, be, bf = tight(ax_c), tight(ax_e), tight(ax_f)
    ya = Bbox.union([ba1, ba2])
    y0, y1 = min(bc.y0, be.y0), max(bc.y1, be.y1)       # shared bottom and top for row (c)/(e)
    x_left = min(ba1.x0, bb.x0, bc.x0)                  # common left edge of (a), (b), (c), (d)
    x_right = max(ba2.x1, bb.x1, be.x1, bf.x1)          # common right edge of (a), (b), (e), (f)
    xe_left = min(be.x0, bf.x0)                         # common left edge of (e) and (f)

    block_frame(fig, Bbox([[x_left, ya.y0], [x_right, ya.y1]]), '(a) Outcome prediction', mask_frac=0.25)
    block_frame(fig, Bbox([[x_left, bb.y0], [x_right, bb.y1]]), '(b) Covariate densities', mask_frac=0.25)
    block_frame(fig, Bbox([[x_left, y0], [bc.x1, y1]]), '(c) Estimated FATE over time', mask_frac=0.6)
    block_frame(fig, Bbox([[xe_left, y0], [x_right, y1]]), '(e) ARMSE under different model specifications', mask_frac=0.9)
    bd = Bbox([[x_left, bf.y0], [bc.x1, bf.y1]])         # same width as (c), same height as (f)
    block_frame(fig, bd, '(d) Estimated F-CATE surface along $X_1$ and time', mask_frac=0.9)
    block_frame(fig, Bbox([[xe_left, bf.y0], [x_right, bf.y1]]), '(f) Coverage of the 95% $\\mathbb{L}_2$ confidence sets', mask_frac=0.9)
    fig.savefig(f'{args.out}/figure2.pdf', bbox_inches='tight')
    fig.savefig(f'{args.out}/figure2.png', dpi=110, bbox_inches='tight')

    # Supplementary figure: true F-CATE surface on the same vertical scale
    fig_s = plt.figure(figsize=(9, 7))
    ax_s = fig_s.add_subplot(111, projection='3d')
    surf_s, _ = panel_surface(ax_s, None, coef, cfg, truth=True, zlim=zlim)
    fig_s.colorbar(surf_s, ax=ax_s, shrink=0.5, aspect=12, pad=0.08, label=r'$\theta^\star$')
    fig_s.savefig(f'{args.out}/figure_true_surface.pdf', bbox_inches='tight')
    print('saved', f'{args.out}/figure2.pdf', f'{args.out}/figure_true_surface.pdf')


if __name__ == '__main__':
    main()