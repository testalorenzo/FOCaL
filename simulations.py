#
# Simple working example of FOCaL 
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from tqdm import tqdm
from matplotlib.patches import FancyBboxPatch


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern

from FOCaL import FOCaL


def generate_data(n, n_time=100, random_state=None):
    """
    Generate functional outcome data with true regression and propensity score.

    Returns:
        Z: Nonlinear covariates DataFrame (x1–x4)
        X: Original covariates DataFrame (z1–z4)
        A: Treatment assignment Series
        Y: Functional outcomes DataFrame (n x n_time)
        betas: Dict of functional coefficients β_j(t)
        mu_true: True expected outcome (without noise) matrix (n x n_time)
        true_pi: True propensity scores vector (n,)
    """
    rng = np.random.default_rng(random_state)

    # --- Covariates ---
    x = rng.normal(0, 1, size=(n, 4))
    X = pd.DataFrame(z, columns=['x1', 'x2', 'x3', 'x4'])

    z1 = np.exp(x[:, 0] / 2)
    z2 = x[:, 1] / (1 + np.exp(x[:, 0])) + 10
    z3 = (x[:, 0] * x[:, 2] / 25 + 0.6) ** 3
    z4 = (x[:, 1] + x[:, 3] + 20) ** 2
    Z = pd.DataFrame({'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4})

    # --- True propensity ---
    true_propensity_linear = -x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2] - 0.1 * x[:, 3]
    true_pi = 1 / (1 + np.exp(-true_propensity_linear))
    A = rng.binomial(1, true_pi)

    # --- Time grid ---
    t_grid = np.linspace(0, 1, n_time).reshape(-1, 1)

    # --- Functional coefficients (GP + shift by scalar mean) ---
    coef_means = {"beta0": 210, "beta1": 27.4, "beta2": 13.7, "beta3": 13.7, "beta4": 13.7, "beta5": 10.0}
    betas = {}
    for name, mean_val in coef_means.items():
        if mean_val == 10.0:
            sd = 10
        else:
            sd = 2
        l = 0.25
        nu = 5.5
        cov = sd ** 2 * Matern(length_scale=l, nu=nu)(t_grid.reshape(-1, 1))
        gp_sample = rng.multivariate_normal(mean=np.zeros(n_time), cov=cov)
        betas[name] = mean_val + gp_sample

    # --- True regression (without noise) ---
    mu_true = np.zeros((n, n_time))
    for i in range(n):
        mu_true[i, :] = (
            betas["beta0"]
            + betas["beta1"] * x[i, 0]
            + betas["beta2"] * x[i, 1]
            + betas["beta3"] * x[i, 2]
            + betas["beta4"] * x[i, 3])

    # --- Observed functional outcomes ---
    Y0 = mu_true + rng.multivariate_normal(np.zeros(n_time), cov)
    Y1 = Y0 + np.outer(x[:, 0] + 1, betas["beta5"])
    Y = A[:, np.newaxis]*Y1 + (1-A[:, np.newaxis])*Y0

    return X, Z, A, Y, betas, mu_true, true_pi, Y1, Y0



# --- Simulation settings --- ARMSE
n_trials = 1000
n = 5000
n_time = 100

# Storage for RMSE
rmse_full_spec = []
rmse_full_mis = []
rmse_reg_mis = []
rmse_pi_mis = []
rmse_oracle = []

# --- Helper function ---
def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

# --- Simulation loop ---
for trial in tqdm(range(n_trials)):
    # Generate data
    X, Z, A, Y, betas, mu_true, true_pi, Y1, Y0 = generate_data(n, n_time=n_time, random_state=trial)

    # Define base learners
    base_mu = LinearRegression()
    base_pi = LogisticRegression(max_iter=1000)
    base_final = LinearRegression()
    
    
    
    
    # --- Oracle estimator ---

    # Oracle predictions (fCATE_hat^oracle)
    true = Y1 - Y0   # shape (n, n_time)

    
    # 1) Full model specification
    focal = FOCaL(clone(base_mu), clone(base_pi), clone(base_final), fpca=5)
    focal.fit(X_mu=X.values, X_pi=X.values, A=A, Y=Y, n_folds=5)
    pred_full = focal.get_FATE(average=False)
    rmse_full_spec.append(rmse(pred_full, true))

    # 2) Full misspecification
    focal = FOCaL(clone(base_mu), clone(base_pi), clone(base_final), fpca=5)
    focal.fit(X_mu=Z.values, X_pi=Z.values, A=A, Y=Y, n_folds=5)
    pred_wrong = focal.get_FATE(average=False)
    rmse_full_mis.append(rmse(pred_wrong, true))

    # 3) Regression misspecified
    focal = FOCaL(clone(base_mu), clone(base_pi), clone(base_final), fpca=5)
    focal.fit(X_mu=Z.values, X_pi=X.values, A=A, Y=Y, n_folds=5)
    pred_reg = focal.get_FATE(average=False)
    rmse_reg_mis.append(rmse(pred_reg, true))

    # 4) Propensity misspecified
    focal = FOCaL(clone(base_mu), clone(base_pi), clone(base_final), fpca=5)
    focal.fit(X_mu=X.values, X_pi=Z.values, A=A, Y=Y, n_folds=5)
    pred_prop = focal.get_FATE(average=False)
    rmse_pi_mis.append(rmse(pred_prop, true))

# --- Prepare DataFrame for Seaborn ---
df = pd.DataFrame({
    'ARMSE': rmse_full_spec + rmse_full_mis + rmse_reg_mis + rmse_pi_mis,
    'Condition': (['Full Specification']*n_trials +
                  ['Full Misspecified']*n_trials +
                  ['Regression Misspecified']*n_trials +
                  ['Propensity Misspecified']*n_trials)
})

sns.set(style="whitegrid", context="notebook", palette="muted")


# Generate functional data
X, Z, A, Y, betas, mu_true, true_pi, Y1, Y0 = generate_data(5000, n_time=100, random_state=72)

# Define nuisances
nuisance_mu = clone(LinearRegression())
nuisance_pi = clone(LogisticRegression(max_iter=1000))
final_regression = clone(LinearRegression())

# Fit FOCaL
focal = FOCaL(nuisance_mu, nuisance_pi, final_regression, fpca=5)
focal.fit(Z.values, Z.values, A, Y, n_folds=5)

# Predict FCATE for new X
pred = focal.predict(Z.values)


#############
# Final plot
#############



colors = {
    "No misspecification": "red",
    "Full misspecification": "orange",
    "Regression misspecified": "green",
    "Propensity score misspecified": "blue"
}

colors_1 = {
    "Full Specification_0": "red",
    "Full Misspecified_0": "orange",
    "Regression Misspecified_0": "green",
    "Propensity Misspecified_0": "blue",
    "Full Specification_1": "red",
    "Full Misspecified_1": "orange",
    "Regression Misspecified_1": "green",
    "Propensity Misspecified_1": "blue",
    "True": "purple"
}




markers = {
    "Full Specification": "o",          # big dots
    "Full Misspecified": "^",        # triangles
    "Regression Misspecified": "s",       # squares
    "Propensity Misspecified": "D"   #diamonds
}

shift_map = {
    "Full Specification": 0,
    "Full Misspecified": 0,       # baseline
    "Propensity Misspecified": 2, # one index later
    "Regression Misspecified": -2, # one index earlier
    "Oracle": -3
}


# Increase all fonts
plt.rcParams.update({
    "font.size": 14,            # default font size
    "axes.titlesize": 16,       # axis titles
    "axes.labelsize": 14,       # x and y labels
    "xtick.labelsize": 14,      # x tick labels
    "ytick.labelsize": 14,      # y tick labels
    "legend.fontsize": 12.8,      # legend text
    "figure.titlesize": 18,     # figure suptitle
})



# Fit FOCaL
focal = FOCaL(nuisance_mu, nuisance_pi, final_regression, fpca=5)
focal.fit(Z.values, Z.values, A, Y, n_folds=5)

# Predict FCATE for new X
pred = focal.predict(Z.values)

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import gridspec, cm
import numpy as np

# --- Defensive shape fixes ---
pred_local = np.array(pred)
if pred_local.shape[0] != Z.shape[0] and pred_local.shape[1] == Z.shape[0]:
    pred_local = pred_local.T

fate = focal.get_FATE(average=False)
fate = np.array(fate)
if fate.shape[0] != Z.shape[0] and fate.shape[1] == Z.shape[0]:
    fate = fate.T

# --- Layout ---
fig = plt.figure(figsize=(22, 20))
gs = gridspec.GridSpec(
    nrows=4, ncols=4, figure=fig,
    height_ratios=[2, 2, 2, 7],  # first row taller
    width_ratios=[1, 1, 1, 1],
    hspace=1.2, wspace=0.55
)

# ---------------------------
# Row 1: Treated vs Untreated
# ---------------------------
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])


# --- Treated ---
schemes_1["Oracle"] = Y1

ax1.plot(
    time,
    treated_mean_obs,
    color="red",
    linewidth=2,
    label="Observed Treated mean"
)

n_markers = 20  # number of markers along the curve

for label, Y_pred in schemes_1.items():
    # keep True or labels ending with _1
    if not (label.endswith("_1") or label == "Oracle"):
        continue

    clean_label = label.replace("_1", "")
    y_mean = np.mean(Y_pred, axis=0)

    # evenly spaced marker indices along the time axis
    marker_idx = np.linspace(0, len(time) - 1, n_markers, dtype=int)

    # apply relative shift for overlapping curves (skip True if you want)
    shift = shift_map.get(clean_label, 0)
    marker_idx = np.clip(marker_idx + shift, 0, len(time)-1)

    if clean_label == "Oracle":
        # plot as a dashed line instead of markers
        ax1.plot(
            time,
            y_mean,
            linewidth=2.0,
            color="black",   # or any color you like
            label=clean_label,
            zorder=15,
            alpha=0.5
        )
    else:
        # plot markers for other schemes
        if clean_label=="Full Misspecified":
            ax1.plot(
                time[marker_idx],
                y_mean[marker_idx],
                marker=markers[clean_label],
                markersize=6,
                color=colors_1[label],
                alpha=0.5,
                label=clean_label,
                zorder=10
            )
        else:
            ax1.plot(
            time[marker_idx],
            y_mean[marker_idx],
            linestyle="None",
            marker=markers[clean_label],
            markersize=6,
            color=colors_1[label],
            alpha=0.5,
            label=clean_label,
            zorder=10
        )




ax1.set_xlabel("Time", labelpad=-130)
ax1.set_ylabel(r"$\hat{Y}$")
ax1.set_title("Treated outcomes")

lines, labels = ax1.get_legend_handles_labels()

# Insert a dummy entry after the first item
lines.insert(1, lines[0])  # reuse first line for spacing
labels.insert(1, "Observed untreated mean")  # dummy text

# Create the legend
leg = ax1.legend(
    lines,
    labels,
    loc="lower left",
    bbox_to_anchor=(-0.05, -0.85),
    frameon=False,
    ncol=7
)

# Make the dummy entry invisible
# 1️⃣ Text color
leg.get_texts()[1].set_color("white")

# 2️⃣ Line/marker color
dummy_handle = leg.get_lines()[1]   # second line in legend
dummy_handle.set_color("white")          # line color
dummy_handle.set_markerfacecolor("white")  # marker fill
dummy_handle.set_markeredgecolor("white")  # marker edge




# --- Untreated ---
schemes_1["Oracle"] = Y0

# Observed untreated mean
# Plot observed untreated mean and capture the line object
obs_line, = ax2.plot(
    time,
    untreated_mean_obs,
    color="blue",
    linewidth=2,
    label="Observed Untreated mean"
)

n_markers = 20  # number of markers along the curve

for label, Y_pred in schemes_1.items():
    # keep labels ending with "_0" or True
    if not (label.endswith("_0") or label == "Oracle"):
        continue

    clean_label = label.replace("_0", "")
    y_mean = np.mean(Y_pred, axis=0)

    # evenly spaced marker indices along the time axis
    marker_idx = np.linspace(0, len(time) - 1, n_markers, dtype=int)

    # apply relative shift for overlapping curves (skip True if you want)
    shift = shift_map.get(clean_label, 0)
    marker_idx = np.clip(marker_idx + shift, 0, len(time)-1)

    if clean_label == "Oracle":
        # plot as a dashed line, hide from legend
        ax2.plot(
            time,
            y_mean,
            linewidth=2.0,
            color="black",
            label="_nolegend_",
            zorder=15,
            alpha=0.5
        )
    else:
        # plot markers for other schemes, hide from legend
        if clean_label=="Full Misspecified":
            ax2.plot(
            time[marker_idx],
            y_mean[marker_idx],
            marker=markers[clean_label],
            markersize=6,
            color=colors_1[label],
            alpha=0.5,
            label="_nolegend_",
            zorder=10)
        else:
            ax2.plot(
            time[marker_idx],
            y_mean[marker_idx],
            linestyle="None",
            marker=markers[clean_label],
            markersize=6,
            color=colors_1[label],
            alpha=0.5,
            label="_nolegend_",
            zorder=10)

# Legend with only the observed line
ax2.legend(
    handles=[obs_line],
    loc="lower left",
    bbox_to_anchor=(-0.95, -0.85),
    frameon=False
)

ax2.set_xlabel("Time", labelpad=-130)
ax2.set_ylabel(r"$\hat{Y}$")
ax2.set_title("Untreated outcomes")




# ---------------------------
# Row 2: Four Density plots
# ---------------------------

gs_row2 = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs[1, :], wspace=0.3
)

for a in range(4):
    ax = fig.add_subplot(gs_row2[0, a])
    cov = Z.iloc[:, a]

    treated_idx = A == 1
    untreated_idx = A == 0

    sns.kdeplot(
        x=cov[treated_idx], weights=true_pi[treated_idx],
        fill=True, color="red", label="Treated",
        common_norm=False, ax=ax
    )
    sns.kdeplot(
        x=cov[untreated_idx], weights=true_pi[untreated_idx],
        fill=True, color="blue", label="Untreated",
        common_norm=False, ax=ax
    )

    ax.set_xlabel(f"X{a}")
    if a == 0:
        ax.set_ylabel("Weighted density")
        ax.legend(loc="upper right",
                  bbox_to_anchor=(2.95, 1.4),
                frameon=False,
                ncol=2)
    else:
        ax.set_ylabel("")
    pos = ax.get_position()
    ax.set_position([
            pos.x0,
            pos.y0 - 0.3*pos.height,
            pos.width,
            pos.height * 1.4
        ])


# ---------------------------
# Row 3: Estimated effect over time (left)
# ---------------------------
# Create the subplot
ax_line = fig.add_subplot(gs[2, 0:2])

# Schemes for this panel
schemes = {
    "Full Specification": pred_full,
    "Full Misspecified": pred_wrong,
    "Regression Misspecified": pred_reg,
    "Propensity Misspecified": pred_prop,
    "Oracle": Y1 - Y0
}

# Number of markers along each curve
n_markers = 20


# Loop over each scheme
for label, Y_pred in schemes.items():
    # If Y_pred is (n_obs, n_time), take the mean across n_obs
    y_arr = np.atleast_2d(Y_pred)       # ensures shape (n_obs, n_time)
    y_mean = np.mean(y_arr, axis=0)     # now guaranteed shape (n_time,)



    # Evenly spaced marker indices along the time axis
    marker_idx = np.linspace(0, len(time) - 1, n_markers, dtype=int)

    # Apply shift for overlapping curves
    shift = shift_map.get(label, 0)
    marker_idx = np.clip(marker_idx + shift, 0, len(time) - 1)

    if label == "Oracle":
        # Plot as dashed line
        ax_line.plot(
            time,
            y_mean,
            linewidth=2.0,
            color="black",
            alpha=0.5,
            zorder=15,
            label=label
        )
    else:
        if label=="Full Misspecified":
            # Plot markers only
            ax_line.plot(
                time[marker_idx],
                y_mean[marker_idx],
                marker=markers.get(label, "o"),  # fallback marker
                markersize=6,
                color=colors_1.get(label+"_1", "C0"),  # fallback color
                alpha=0.5,
                zorder=10,
                label=label
            )
        else:
            # Plot markers only
            ax_line.plot(
                time[marker_idx],
                y_mean[marker_idx],
                linestyle="None",
                marker=markers.get(label, "o"),  # fallback marker
                markersize=6,
                color=colors_1.get(label+"_1", "C0"),  # fallback color
                alpha=0.5,
                zorder=10,
                label=label
            )

        
# Labels and legend
ax_line.set_xlabel("Time", labelpad=-140)
ax_line.set_ylabel(r"$\hat{\tau}$")
ax_line.legend(
    loc="lower left",
    bbox_to_anchor=(-0.05, -0.89),
    frameon=False,
    ncol=3
)




# ---------------------------
# Row 3-4: Boxplots (right)
# ---------------------------
ax_box = fig.add_subplot(gs[2:4, 2:4])

sns.set(style="whitegrid", context="notebook")

# Explicit palette in order of appearance
palette = ["red", "orange", "green", "blue"]

sns.boxplot(
    data=df,
    x="Condition",
    y="ARMSE",
    ax=ax_box,
    showfliers=False,
    width=0.4,
    palette=palette
)

ax_box.set_xlabel("Condition", labelpad=15.5)
ax_box.set_ylabel("ARMSE", labelpad=-1)
ax_box.tick_params(axis="x", rotation=20)

# Position tweak
pos = ax_box.get_position()
ax_box.set_position([
    pos.x0,
    pos.y0 + 0.05 * pos.height,
    pos.width,
    pos.height
])





# ---------------------------
# Row 4: 3D FCATE Surface along Z0 and Time (left)
# ---------------------------
ax_surf = fig.add_subplot(gs[3, 0:2], projection='3d')

# Grid for Z0
z0_vals = np.linspace(Z.iloc[:, 0].min(), Z.iloc[:, 0].max(), 100)

# Build full Z grid: vary Z0, keep other Z columns at mean
Z_grid = np.repeat(Z.mean(axis=0).values[None, :], len(z0_vals), axis=0)
Z_grid[:, 0] = z0_vals

# Predict FCATE
tau_hat = focal.predict(Z_grid)  # shape (n_points, n_time)

# Meshgrid for plotting
T, Z0_mesh = np.meshgrid(np.arange(Y.shape[1]), z0_vals)

Z_LIM = (-45, 85)   # same vertical scale for all plots

surf = ax_surf.plot_surface(
    T, Z0_mesh, tau_hat,
    cmap=cm.coolwarm,
    vmin=Z_LIM[0],   # color scale min
    vmax=Z_LIM[1],   # color scale max
    linewidth=0,
    alpha=0.9
)

ax_surf.set_zlim(*Z_LIM)  # enforce vertical axis


ax_surf.set_box_aspect(None, zoom=1.2)

ax_surf.set_xlabel("Time")
ax_surf.set_ylabel("X0")
ax_surf.set_zlabel(r"$\hat{\tau}$")

fig.colorbar(surf, ax=ax_surf, shrink=0.5, aspect=10, pad=0.13, label=r"$\hat{\tau}$")


# ============================================================
# Fancy frames + masked titles
# ============================================================
def add_block_frame(fig, gs_cell, title, mask_frac=0.36):
    bbox = gs_cell.get_position(fig)
    pad_x, pad_y = 0.023, 0.045

    x0 = bbox.x0 - pad_x
    y0 = bbox.y0 - pad_y
    w  = bbox.width + 1.5 * pad_x
    h  = bbox.height + 1.78 * pad_y

    frame = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        transform=fig.transFigure,
        fill=False, lw=1.5, edgecolor="grey", zorder=2
    )
    fig.add_artist(frame)

    tx, ty = x0 + w/2, y0 + h + 0.01
    fig.text(tx, ty, title, ha="center", va="center",
             fontsize=16, weight="bold", zorder=4)

    mask = FancyBboxPatch(
        (tx - mask_frac*w/2, ty - 0.014),
        mask_frac*w, 0.015,
        boxstyle="round,pad=0.0,rounding_size=0.01",
        transform=fig.transFigure,
        facecolor="white", edgecolor="none", zorder=3
    )
    fig.add_artist(mask)
    
def center_axis(ax, shrink=0.94):
    box = ax.get_position()
    cx = box.x0 + box.width / 2
    cy = box.y0 + box.height / 2
    new_w = box.width * shrink
    new_h = box.height * shrink
    ax.set_position([
        cx - new_w / 2,
        cy - new_h / 2,
        new_w,
        new_h
    ])


add_block_frame(fig, gs[0, :],
                "(a) Outcome Prediction",
                mask_frac=0.2)   # narrower

add_block_frame(fig, gs[1, :],
                "(b) Covariate Densities",
                mask_frac=0.21)   # wider

add_block_frame(fig, gs[2, :2],
                "(c) Estimated FATE over time",
                mask_frac=0.52)

add_block_frame(fig, gs[3, :2],
                "(d) Estimated FCATE surface along X0 and Time",
                mask_frac=0.72)

add_block_frame(fig, gs[2:, 2:],
                "(e) ARMSE under Different Model Specifications",
                mask_frac=0.72)

for ax in fig.axes:
    if not isinstance(ax, Axes3D):
        center_axis(ax, shrink=0.96)



