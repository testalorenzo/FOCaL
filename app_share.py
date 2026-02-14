#
# SHARE Application
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import numpy as np
from numpy import zeros_like, array, vstack
from copy import deepcopy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.base import clone
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from FOCaL import FOCaL

import pyreadr

# Load the RData file
result = pyreadr.read_r('./data/data_for_fasten.Rdata')

# Show the keys (i.e., variable names stored in the RData file)
print(result.keys())


Amat = result['Amat']
names_var = result['names_var']
descriptions_var = result['descriptions_var']
type_var = result['type_var']


# Amat: (num_variables, num_samples, num_timepoints)
import numpy as np

num_vars = Amat.shape[0]
num_samples = Amat.shape[1]
num_timepoints = Amat.shape[2]

scalar_indices = []
functional_indices = []

for i in range(num_vars):
    data_i = Amat[i]  # shape: (samples, timepoints)
    variation = np.std(data_i, axis=1)
    if np.allclose(variation, 0, atol=1e-6):
        scalar_indices.append(i)
    else:
        functional_indices.append(i)


# First Test: Effect of Hypertension on CASP (0) and Mobility Index (9)
'''
To replicate other results, substitute A and Y accordingly. Note that the number of observations
can change due the restrictions imposed on treatment (specifically, row_mask = (np.all(W1 == 0, axis=1)) | (np.all(W1 == 1, axis=1)))
'''

Y1 = Amat[0]
Y2 = Amat[9]
W1 = Amat[16]
W2 = Amat[17]
X = Amat[26]
education = Amat[27]
first_test = Amat[28]
gender = Amat[34]
vaccinations = Amat[35]
smoke = Amat[36]
W1 = np.array(W1)
W2 = np.array(W2)
X = np.array(X)
education = np.array(education)
first_test = np.array(first_test)
gender = np.array(gender)
vaccinations = np.array(vaccinations)
smoke = np.array(smoke)
W1 = W1.round().astype(int)
W2 = W2.round().astype(int)
# rows that are all 0s or all 1s
row_mask = (np.all(W1 == 0, axis=1)) | (np.all(W1 == 1, axis=1))
idx = np.where(row_mask)[0]
Y1 = Amat[0, row_mask]
Y2 = Amat[9, row_mask]
age = X[row_mask, 0]
W1 = W1[row_mask, 0]
education     = education[row_mask, 0]
first_test    = first_test[row_mask, 0]
gender        = gender[row_mask, 0]
vaccinations  = vaccinations[row_mask, 0]
smoke         = smoke[row_mask, 0]
df = pd.DataFrame({
    "age": age,
    "education": education,
    "first_test": first_test,
    "gender": gender,
    "vaccinations": vaccinations,
    "smoke": smoke
})


base_mu = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="logistic",
    solver="adam",          # explicitly Adam
    alpha=1e-3,             # L2 regularization
    learning_rate="adaptive",
    max_iter=50000,
    early_stopping=False,
    random_state=0
)

base_pi = LogisticRegression(max_iter=1000)

base_final = MLPRegressor(
    hidden_layer_sizes=(5,),
    activation="logistic",
    solver="adam",          # explicitly Adam
    alpha=1e-3,             # L2 regularization
    learning_rate="adaptive",
    max_iter=4000,
    early_stopping=False,
    random_state=0
)


# --- Fit FOCaL on your real data ---
focala = FOCaL(clone(base_mu), clone(base_pi), clone(base_final), fpca=5)
focala.fit(X_mu=df.values, X_pi=df.values, A=W1, Y=Y1, n_folds=5)

focalb = FOCaL(clone(base_mu), clone(base_pi), clone(base_final), fpca=5)
focalb.fit(X_mu=df.values, X_pi=df.values, A=W1, Y=Y2, n_folds=5)


# store predicted pseudo-factual outcomes
Y_treat = focala.predict_Y(np.ones(Y1.shape[0]))
Y_control = focala.predict_Y(np.zeros(Y1.shape[0]))

Y_treat_1 = focalb.predict_Y(np.ones(Y1.shape[0]))
Y_control_1 = focalb.predict_Y(np.zeros(Y1.shape[0]))

# --- Prepare observed means ---
time = np.arange(Y1.shape[1])

treated_mean_obs = Y1[W1 == 1].mean(axis=0)
untreated_mean_obs = Y1[W1 == 0].mean(axis=0)

treated_mean_obs_1 = Y2[W1 == 1].mean(axis=0)
untreated_mean_obs_1 = Y2[W1 == 0].mean(axis=0)

# --- Single combined plot ---
plt.figure(figsize=(13, 6))

# Observed
plt.plot(time, treated_mean_obs, color="red", linewidth=2, label="Observed Treated Mean")
plt.plot(time, untreated_mean_obs, color="blue", linewidth=2, label="Observed Untreated Mean")

# Predicted
plt.plot(time, Y_treat.mean(axis=0), color="yellow", linestyle="--", linewidth=2,
         label="Predicted Untreated Mean (trimmed)")
plt.plot(time, Y_control.mean(axis=0), color="brown", linestyle="--", linewidth=2,
         label="Predicted Untreated Mean (trimmed)")

plt.xlabel("Time")
plt.ylabel("Y")
plt.title("Observed vs Predicted Mean Outcomes (Treated & Untreated)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# --- Single combined plot ---
plt.figure(figsize=(13, 6))

# Observed
plt.plot(time, treated_mean_obs_1, color="red", linewidth=2, label="Observed Treated Mean")
plt.plot(time, untreated_mean_obs_1, color="blue", linewidth=2, label="Observed Untreated Mean")

# Predicted
plt.plot(time, Y_treat_1.mean(axis=0), color="yellow", linestyle="--", linewidth=2,
         label="Predicted Untreated Mean (trimmed)")
plt.plot(time, Y_control_1.mean(axis=0), color="brown", linestyle="--", linewidth=2,
         label="Predicted Untreated Mean (trimmed)")

plt.xlabel("Time")
plt.ylabel("Y")
plt.title("Observed vs Predicted Mean Outcomes (Treated & Untreated)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

###############
# Final Figure
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ----------------------------
# Utilities
# ----------------------------
def is_binary(x):
    vals = np.unique(x.dropna())
    return set(vals).issubset({0, 1})

# ----------------------------
# Styling
# ----------------------------
COLORS = {
    "treated": "#D62728",
    "untreated": "#1F77B4"
}
ALPHA = 0.6
sns.set_style("whitegrid")


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



# ----------------------------
# Layout
# ----------------------------
fig = plt.figure(figsize=(22, 18))

gs = gridspec.GridSpec(
    nrows=4,
    ncols=12,
    hspace=1.3,
    wspace=12.5,
    height_ratios=[2, 1, 2, 2]
)

# ============================================================
# Row 1 — Covariate densities / bars
# ============================================================
treated = W1 == 1
untreated = W1 == 0

for i, cov_name in enumerate(df.columns[:6]):
    ax = fig.add_subplot(gs[0, 2*i:2*i+2])
    cov = df[cov_name]

    if cov_name == "first_test":
        cov_name = "number_children"
        levels = np.sort(cov.dropna().unique())
        x = np.arange(len(levels))
        width = 0.35

        ax.bar(x - width/2,
               [np.mean(cov[untreated] == l) for l in levels],
               width, color=COLORS["untreated"], alpha=ALPHA)

        ax.bar(x + width/2,
               [np.mean(cov[treated] == l) for l in levels],
               width, color=COLORS["treated"], alpha=ALPHA)

        ax.set_xticks(x)
        ax.set_xticklabels([int(round(l)) for l in levels])
        ax.set_ylabel("Proportion")

    elif is_binary(cov):
        ax.bar(["Untreated", "Treated"],
               [cov[untreated].mean(), cov[treated].mean()],
               color=[COLORS["untreated"], COLORS["treated"]],
               alpha=ALPHA)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")

    else:
        sns.kdeplot(cov[treated], fill=True,
                    color=COLORS["treated"], alpha=ALPHA,
                    common_norm=False, ax=ax, label='Treated')
        sns.kdeplot(cov[untreated], fill=True,
                    color=COLORS["untreated"], alpha=ALPHA,
                    common_norm=False, ax=ax, label='Untreated')
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(
            loc="center left",
            bbox_to_anchor=(3.53, 1.3),
            frameon=False,
            ncol=2)
    


    ax.set_title(cov_name)
    ax.set_xlabel("")
    
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + 0.13 * pos.width,
        pos.y0,
        pos.width * 1.15,
        pos.height
    ])


# ============================================================
# Inner grids: CASP (left) / Mobility (right)
# ============================================================
gs_casp = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[1:3, :6],
    height_ratios=[1.2, 5.3], hspace=0.35, wspace=0.5
)

gs_mob = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[1:3, 6:],
    height_ratios=[1.2, 5.3], hspace=0.35, wspace=0.5
)

# ============================================================
# Row 2 — FATE
# ============================================================
time = np.arange(Y1.shape[1])

ax_casp_fate = fig.add_subplot(gs_casp[0])
ax_mob_fate  = fig.add_subplot(gs_mob[0])

ax_casp_fate.plot(time, focala.get_FATE(average=True),
                  lw=2, color="blue", alpha=0.7)
ax_mob_fate.plot(time, focalb.get_FATE(average=True),
                 lw=2, color="blue", alpha=0.7)

ax_casp_fate.set_title("FATE", x=0.47)
ax_mob_fate.set_title("FATE", x=0.47)
ax_casp_fate.set_ylabel(r"$\hat{\tau}$")
ax_mob_fate.set_ylabel(r"$\hat{\tau}$")
ax_casp_fate.set_xlabel("Time")
ax_mob_fate.set_xlabel("Time")

pos = ax_casp_fate.get_position()
ax_casp_fate.set_position([
    pos.x0 + 0.1 * pos.width,
    pos.y0 - 0.005,
    pos.width * 0.85,
    pos.height * 1.3
])

pos = ax_mob_fate.get_position()
ax_mob_fate.set_position([
    pos.x0 + 0.1 * pos.width,
    pos.y0 - 0.005,
    pos.width * 0.85,
    pos.height * 1.3
])

ite_casp = focala.get_FATE(average=False)
ite_mob = focalb.get_FATE(average=False)

drfos_cov = np.cov(ite_casp, rowvar=False) / ite_casp.shape[0]

# Simulate 95% confidence bands from Gaussian Process centerd in drfos and with covariance matrix cov_dr_fos
drfos_sim = np.random.multivariate_normal(np.mean(ite_casp, axis=0), drfos_cov, 10000)
lower_bound = np.quantile(drfos_sim, 0.025, axis=0)
upper_bound = np.quantile(drfos_sim, 0.975, axis=0)

ax_casp_fate.fill_between(range(192), lower_bound, upper_bound, alpha=0.35)


drfos_cov2 = np.cov(ite_mob, rowvar=False) / ite_mob.shape[0]

# Simulate 95% confidence bands from Gaussian Process centerd in drfos and with covariance matrix cov_dr_fos
drfos_sim2 = np.random.multivariate_normal(np.mean(ite_mob, axis=0), drfos_cov2, 10000)
lower_bound2 = np.quantile(drfos_sim2, 0.025, axis=0)
upper_bound2 = np.quantile(drfos_sim2, 0.975, axis=0)

ax_mob_fate.fill_between(range(192), lower_bound2, upper_bound2, alpha=0.35)



# ============================================================
# Row 3 — FCATE surfaces (EXACT SAME SYNTAX AS YOUR ORIGINAL)
# ============================================================
age_vals = np.linspace(df["age"].min(), df["age"].max(), 100)
Z_ref = df.mean(axis=0).values
gender_vals = (0, 1)

models = {"CASP": focala, "Mobility": focalb}

# ---- precompute common z-limits ----
zlims = {}
for label, model in models.items():
    zmin, zmax = np.inf, -np.inf
    for gender_val in gender_vals:
        Z_grid = np.repeat(Z_ref[None, :], len(age_vals), axis=0)
        Z_grid[:, df.columns.get_loc("age")] = age_vals
        Z_grid[:, df.columns.get_loc("gender")] = gender_val
        Z_grid[:, df.columns.get_loc("education")] = np.mean(education)
        Z_grid[:, df.columns.get_loc("vaccinations")] = 1
        Z_grid[:, df.columns.get_loc("smoke")] = 0
        Z_grid[:, df.columns.get_loc("first_test")] = np.mean(first_test)

        tau_hat = model.predict(Z_grid)
        zmin = min(zmin, tau_hat.min())
        zmax = max(zmax, tau_hat.max())

    zlims[label] = (zmin, zmax)

time = np.arange(Y1.shape[1])
T_mesh, AGE_mesh = np.meshgrid(time, age_vals)

gs_casp_fc = gridspec.GridSpecFromSubplotSpec(1, 2, gs_casp[1], wspace=0.4)
gs_mob_fc  = gridspec.GridSpecFromSubplotSpec(1, 2, gs_mob[1],  wspace=0.4)

# ---- CASP ----
axes_casp = []
surf_casp = None

for j, gender_val in enumerate(gender_vals):
    ax = fig.add_subplot(gs_casp_fc[j], projection="3d")
    axes_casp.append(ax)

    Z_grid = np.repeat(Z_ref[None, :], len(age_vals), axis=0)
    Z_grid[:, df.columns.get_loc("age")] = age_vals
    Z_grid[:, df.columns.get_loc("gender")] = gender_val
    Z_grid[:, df.columns.get_loc("education")] = np.mean(education)
    Z_grid[:, df.columns.get_loc("vaccinations")] = 1
    Z_grid[:, df.columns.get_loc("smoke")] = 0
    Z_grid[:, df.columns.get_loc("first_test")] = np.mean(first_test)

    tau_hat = focala.predict(Z_grid)

    surf = ax.plot_surface(
        T_mesh, AGE_mesh, tau_hat,
        cmap=cm.coolwarm, linewidth=0, alpha=0.9,
        vmin=zlims["CASP"][0], vmax=zlims["CASP"][1]
    )

    if j == 0:
        surf_casp = surf

    ax.set_title(f"FCATE(age, gender={'M' if gender_val==0 else 'F'})")
    ax.set_xlabel("Time", labelpad=6)
    ax.set_ylabel("Age", labelpad=10)
    ax.set_zlabel(r"$\hat{\tau}$", labelpad=4)
    
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + 0.1 * pos.width,
        pos.y0,
        pos.width * 1.2,
        pos.height
    ])

fig.colorbar(
    surf_casp,
    ax=axes_casp,
    shrink=0.6,
    pad=0.07
)





# ---- Mobility ----
axes_mob = []
surf_mob = None

for j, gender_val in enumerate(gender_vals):
    ax = fig.add_subplot(gs_mob_fc[j], projection="3d")
    axes_mob.append(ax)

    Z_grid = np.repeat(Z_ref[None, :], len(age_vals), axis=0)
    Z_grid[:, df.columns.get_loc("age")] = age_vals
    Z_grid[:, df.columns.get_loc("gender")] = gender_val
    Z_grid[:, df.columns.get_loc("education")] = np.mean(education)
    Z_grid[:, df.columns.get_loc("vaccinations")] = 1
    Z_grid[:, df.columns.get_loc("smoke")] = 0
    Z_grid[:, df.columns.get_loc("first_test")] = np.mean(first_test)

    tau_hat = focalb.predict(Z_grid)

    surf = ax.plot_surface(
        T_mesh, AGE_mesh, tau_hat,
        cmap=cm.coolwarm, linewidth=0, alpha=0.9,
        vmin=zlims["Mobility"][0], vmax=zlims["Mobility"][1]
    )

    if j == 0:
        surf_mob = surf

    ax.set_title(f"FCATE(age, gender={'M' if gender_val==0 else 'F'})")
    ax.set_xlabel("Time", labelpad=6)
    ax.set_ylabel("Age", labelpad=10)
    ax.set_zlabel(r"$\hat{\tau}$", labelpad=4)
    
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + 0.1 * pos.width,
        pos.y0,
        pos.width * 1.2,
        pos.height
    ])

fig.colorbar(
    surf_mob,
    ax=axes_mob,
    shrink=0.6,
    pad=0.07
)




# ============================================================
# Fancy frames + masked titles
# ============================================================
def add_block_frame(fig, gs_cell, title, mask_frac=0.36):
    bbox = gs_cell.get_position(fig)
    pad_x, pad_y = 0.02, 0.03

    x0 = bbox.x0 - pad_x
    y0 = bbox.y0 - pad_y
    w  = bbox.width + 1.85 * pad_x
    h  = bbox.height + 2.95 * pad_y

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
        mask_frac*w, 0.035,
        boxstyle="round,pad=0.0,rounding_size=0.01",
        transform=fig.transFigure,
        facecolor="white", edgecolor="none", zorder=3
    )
    fig.add_artist(mask)

add_block_frame(fig, gs[0, :],
                "(a) Covariate Densities",
                mask_frac=0.2)

add_block_frame(fig, gs[1:3, :6],
                "(b) CASP — Hypertension",
                mask_frac=0.42)

add_block_frame(fig, gs[1:3, 6:],
                "(c) Mobility Index — Hypertension",
                mask_frac=0.52)

plt.show()
