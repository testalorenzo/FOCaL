#
# COVID-19 Application
#

# Rdata can be downloaded from https://github.com/tobiaboschi/fdaCOVID2/raw/refs/heads/main/fdaCOVID2git.Rdata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from copy import deepcopy
from numpy import zeros_like, array, vstack
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa
import re
import warnings

import FOCaL

if __name__ == "__main__":

    plot = True

    # Choose wave
    wave = 1

    # Load data
    varnames = np.loadtxt('./data/varnames.txt', dtype='str')
    varnames = [name.replace('"', '') for name in varnames]
    if wave == 1:
        X = np.loadtxt('./data/A1.txt')
        Y = np.loadtxt('./data/b1.txt')
    else:
        X = np.loadtxt('./data/A2.txt')
        Y = np.loadtxt('./data/b2.txt')

    n = X.shape[0]
    p = X.shape[1]
    t = Y.shape[1]

    # Create DataFrame
    data = pd.DataFrame(X, columns=varnames)
    # data = data.drop(['area_before'], axis=1)

    # sns.histplot(data, x='adults_per_family_doctor_2019')
    # plt.show()

    # Create treatment
    data['A'] = (data.adults_per_family_doctor_2019 >= np.median(data.adults_per_family_doctor_2019)).astype(int)
    A = data['A'].to_numpy()

    X = data.drop(columns=['adults_per_family_doctor_2019', 'A'])
    cols = X.columns
    X = X.to_numpy()

    # Plot Y color-coded by treatment
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(Y[A==1, :].T, color='blue', alpha=0.5)
        plt.plot(Y[A==0, :].T, color='red', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Compute and use only fPCAs of Y
    n_components = 1
    pca = PCA(n_components=n_components)
    Y_model = pca.fit_transform(Y)
    #Y_model = Y_model.ravel()
    #Y = pca.inverse_transform(Y_model)

    # Compare two representations
    if plot:
        plt.figure(figsize=(8, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Y')
        plt.plot(Y.T)
        plt.subplot(1, 2, 2)
        plt.title('PCA Transformed Y')
        plt.plot(pca.inverse_transform(Y_model).T)
        plt.show()

    # Explained variance plot
    explained_variance = pca.explained_variance_ratio_
    if plot:
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    # Initialize FOCaL model
    focal = FOCaL.FOCaL(
        nuisance_mu= RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
        nuisance_pi= RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
        final_regression= DecisionTreeRegressor(random_state=42, max_depth=3), # RandomForestRegressor(random_state=42),
        fpca = n_components if n_components > 0 else False
    )

    # Fit the model
    focal.fit(X, A, Y, n_folds=2, random_state=42, DR=False)

    # Get the estimated functional average treatment effect
    fate = focal.get_FATE()
    if plot:
        # Plot the estimated functional average treatment effect
        plt.figure(figsize=(10, 6))
        plt.plot(range(t), fate, label='FOCaL')
        plt.plot(range(t), Y[A==1].mean(axis=0) - Y[A==0].mean(axis=0), label='Difference in Means')
        plt.axhline(0, color='red', linestyle='--')
        plt.legend()
        plt.title('Estimated Functional Average Treatment Effect')
        plt.xlabel('Time Points')
        plt.ylabel('FATE')
        plt.tight_layout()
        plt.show()

    # Predict the treatment effect
    fcate = focal.predict(X)
    
    #plt.matshow(fcate, cmap='coolwarm', aspect='auto')
    #plt.colorbar(label='Estimated Treatment Effect')
    if plot:
        plt.plot(fcate.T, alpha=0.7)
        plt.title('Estimated Treatment Effect by FOCaL')
        plt.xlabel('Time Points')
        plt.ylabel('Samples')
        plt.tight_layout()
        plt.show()

    # Predict marginal treatment effect wrt to each variable
    n_eval = 50
    for var in cols:
        var_range = np.linspace(data[var].min(), data[var].max(), n_eval)
        X_pred = np.tile(X.mean(axis=0), (n_eval, 1))
        X_pred[:, cols==var] = var_range.reshape(-1, 1)
        fcate_pred = focal.predict(X_pred)
        if plot:
            # Plot 3d surface perc_over_65 vs time vs treatment effect
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            X_grid, Y_grid = np.meshgrid(np.arange(t), var_range)
            Z_grid = fcate_pred
            ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm', edgecolor='none')
            ax.set_ylabel(var)
            ax.set_xlabel('Time')
            ax.set_zlabel('Estimated Treatment Effect')
            ax.set_title('Estimated Treatment Effect vs ' + var)
            plt.tight_layout()
            plt.show()

    # Plot tree
    if plot:
        for i in range(2):
            plot_tree(focal.FCATE[i], filled=True, feature_names=cols, fontsize=5, max_depth=3)
            plt.show()
    
            leaf_nodes = []
            n_nodes = focal.FCATE[i].tree_.node_count
            for j in range(n_nodes):
                if focal.FCATE[i].tree_.children_left[j] == focal.FCATE[i].tree_.children_right[j]:
                    leaf_nodes.append(j)

            leaf_values = focal.FCATE[i].tree_.value[leaf_nodes, 0, 0] 
            predictions = focal.FCATE_basis[i].inverse_transform(leaf_values.reshape(-1, focal.fpca))

            plt.figure(figsize=(10, 6))
            plt.plot(predictions.T, alpha=0.7)
            plt.title(f'Predictions from Leaf Nodes of Tree {i+1}')
            plt.xlabel('Time Points')
            plt.ylabel('Predicted Treatment Effect')
            plt.tight_layout()
            plt.show()            



def load_and_fit_wave(wave, n_components=1, random_state=42):
    """
    Load data for a single wave, define treatment, and fit FOCaL.
    Each wave is treated as a fully independent causal experiment.
    """

    # ----------------------------
    # Load variable names
    # ----------------------------
    varnames = np.loadtxt("varnames.txt", dtype=str)
    varnames = [v.replace('"', '') for v in varnames]

    # ----------------------------
    # Load wave-specific data
    # ----------------------------
    if wave == 1:
        X_raw = np.loadtxt("A1.txt")
        Y = np.loadtxt("b1.txt")
    elif wave == 2:
        X_raw = np.loadtxt("A2.txt")
        Y = np.loadtxt("b2.txt")
    else:
        raise ValueError("wave must be 1 or 2")

    # ----------------------------
    # Build DataFrame
    # ----------------------------
    data = pd.DataFrame(X_raw, columns=varnames)

    # ----------------------------
    # Define treatment
    # ----------------------------
    data["A"] = (
        data["adults_per_family_doctor_2019"]
        >= data["adults_per_family_doctor_2019"].mean()
    ).astype(int)

    A = data["A"].to_numpy()

    # ----------------------------
    # Covariate matrix
    # ----------------------------
    X = data.drop(
        columns=["adults_per_family_doctor_2019", "A"]
    )
    cols = X.columns
    X = X.to_numpy()
    
    print(data['pm10_2019'].min())
    print(data['pm10_2019'].max())
    

    # ----------------------------
    # Fit FOCaL
    # ----------------------------
    
    base_mu = MLPRegressor(
        hidden_layer_sizes=(20, 20, 20),
        activation="relu",
        solver="adam",          # explicitly Adam
        alpha=1e-3,             # L2 regularization
        learning_rate="adaptive",
        max_iter=20000,
        early_stopping=False,
        random_state=0
    )


    base_final = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10),
        activation="relu",
        solver="adam",          # explicitly Adam
        alpha=1e-3,             # L2 regularization
        learning_rate="adaptive",
        max_iter=5000,
        early_stopping=False,
        random_state=0
    )

    
    focal = FOCaL(
        nuisance_mu=base_mu,
        nuisance_pi=LogisticRegression(max_iter=1000),
        final_regression=base_final,
        fpca=5
    )

    focal.fit(
        X, X, A, Y,
        n_folds=5,
        random_state=random_state,
        DR=True
    )

    # ----------------------------
    # Sanity checks
    # ----------------------------
    print(f"\nWave {wave}")
    print("n:", X.shape[0])
    print("treated share:", A.mean())
    print("Y shape:", Y.shape)

    return {
        "wave": wave,
        "data": data,
        "X": X,
        "A": A,
        "Y": Y,
        "cols": cols,
        "focal": focal
    }


# ----------------------------
# Run for both waves
# ----------------------------
res_wave2 = load_and_fit_wave(wave=2)
res_wave1 = load_and_fit_wave(wave=1)


warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------
# Styling
# ----------------------------
COLORS = {
    "treated": "#D62728",
    "untreated": "#1F77B4"
}
ALPHA = 0.6
sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.8,
    "figure.titlesize": 18,
})

def strip_year(name):
    return re.sub(r"^average_|_\d{4}$", "", name)

# ============================================================
# Helper: plot one wave column
# ============================================================
def plot_wave_column(fig, gs_cell, res, wave_label):

    df     = res["data"]
    X      = res["X"]
    Y      = res["Y"]
    A      = res["A"]
    cols   = res["cols"]
    focal  = res["focal"]

    time = np.arange(Y.shape[1])
    m = Y.shape[1]

    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs_cell,
        height_ratios=[1.0, 7],
        hspace=0.6
    )

    # ============================================================
    # FATE
    # ============================================================
    ax_fate = fig.add_subplot(gs_inner[0,:])

    tau_i = focal.get_FATE(average=False)
    
    # Trim extremes
    norms = np.linalg.norm(tau_i, axis=1)
    keep = (norms >= np.quantile(norms, 0.01)) & \
           (norms <= np.quantile(norms, 0.99))

    tau_trimmed = tau_i[keep]
    tau_fate = tau_trimmed.mean(axis=0)

    # Covariance for CI from trimmed data
    drfos_cov = np.cov(tau_trimmed, rowvar=False) / tau_trimmed.shape[0]

    # Simulate 95% confidence bands
    drfos_sim = np.random.multivariate_normal(
        np.mean(tau_trimmed, axis=0),
        drfos_cov,
        10000
    )
    lower_bound = np.quantile(drfos_sim, 0.025, axis=0)
    upper_bound = np.quantile(drfos_sim, 0.975, axis=0)
    
    

    # Plot CI
    ax_fate.fill_between(range(len(time)), lower_bound, upper_bound, alpha=0.35)

    # Plot FATE
    ax_fate.plot(time, tau_fate, lw=2, color="blue", alpha=0.7)
    ax_fate.axhline(0, ls=":", lw=1)

    # Vertical axis limits
    ax_fate.set_ylim(-1, 3)

    ax_fate.set_title("FATE")
    ax_fate.title.set_position((0.5, 1.0))
    ax_fate.set_xlabel("Time")
    ax_fate.set_ylabel(r"$\hat{\tau}$")

    # Adjust position
    pos = ax_fate.get_position()
    ax_fate.set_position([
        pos.x0 + 0.1 * pos.width,
        pos.y0 - 0.02,
        pos.width * 0.9,
        pos.height * 2.3
    ])


    # ============================================================
    # FCATE (single surface)
    # ============================================================
    ax_fc = fig.add_subplot(gs_inner[1,0:2], projection="3d")

    var_name = "area_before"
    var_idx  = list(cols).index(var_name)

    var_vals = np.linspace(
        df[var_name].min(),
        df[var_name].max(),
        100
    )

    X_ref = X.mean(axis=0)
    X_grid = np.repeat(X_ref[None, :], len(var_vals), axis=0)
    X_grid[:, var_idx] = var_vals

    tau_hat = focal.predict(X_grid)

    T_mesh, V_mesh = np.meshgrid(time, var_vals)

    # === FIXED COLOR SCALE ===
    Z_LIM = (-2.5, 8)

    surf = ax_fc.plot_surface(
        T_mesh, V_mesh, tau_hat,
        cmap=cm.coolwarm,
        vmin=Z_LIM[0],
        vmax=Z_LIM[1],
        linewidth=0,
        alpha=0.9
    )

    ax_fc.set_zlim(*Z_LIM)

    ax_fc.set_title(f"FCATE({strip_year(var_name)})")
    ax_fc.title.set_position((0.64, 1.0))
    ax_fc.set_xlabel("Time", labelpad=6)
    ax_fc.set_ylabel(strip_year(var_name), labelpad=10)
    ax_fc.set_zlabel(r"$\hat{\tau}$", labelpad=4)

    pos = ax_fc.get_position()
    ax_fc.set_position([
        pos.x0 + 1 * pos.width,
        pos.y0 + 0.05 * pos.height,
        pos.width,
        pos.height
    ])

    fig.colorbar(surf, ax=ax_fc, shrink=0.6, pad=0.12)
    
    
    # ============================================================
    # FCATE (single lines)
    # ============================================================
    
    
    ax_fate2d = fig.add_subplot(gs_inner[1, 2])

    # Quantiles
    x_col = X[:, var_idx]

    var_vals = np.linspace(x_col.min(), x_col.max(), 100)
    q_low, q_high = np.quantile(x_col, [0.05, 0.95])

    print(q_low)
    print(q_high)

    X_ref = X.mean(axis=0)

    X_low  = X_ref.copy()
    X_high = X_ref.copy()

    X_low[var_idx]  = q_low
    X_high[var_idx] = q_high

    tau_fc_low  = focal.predict(X_low.reshape(1, 6))
    tau_fc_high = focal.predict(X_high.reshape(1, 6))

    # Plot
    ax_fate2d.plot(time, tau_fc_low.reshape(m,-1),  lw=2, color="blue",   label=f"FCATE ({round(q_low, 2)})")
    ax_fate2d.plot(time, tau_fc_high.reshape(m,-1), lw=2, color="red", label=f"FCATE ({round(q_high, 2)})")

    ax_fate2d.axhline(0, ls=":", lw=1)
    ax_fate2d.set_ylim(-3, 10)

    
    ax_fate2d.set_title("FCATE Surface Cuts", y=1.14)
    ax_fate2d.set_xlabel("Time")
    ax_fate2d.set_ylabel(r"$\hat{\tau}$", labelpad=-3)
    ax_fate2d.legend(frameon=False)
    
    
    ######CONFIDENCE INTERVALS
    
    tau_hat = focal.predict(X)
    res_sq = (tau_i - tau_hat)
    V = np.einsum('ni,nj->nij', res_sq, res_sq)
    triu_indices = np.triu_indices(m)
    V_flat = np.array([v[triu_indices] for v in V])

    #X_low

    v_pred_flat = (
        RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0
        )
        .fit(X, V_flat)
        .predict(X_low.reshape(1, -1))
    )


    

    # 6. Reconstruct the m x m matrix
    sigma_hat = np.zeros((m, m))
    sigma_hat[triu_indices] = v_pred_flat
    sigma_hat = sigma_hat + sigma_hat.T - np.diag(sigma_hat.diagonal())
    sigma_hat /= tau_i.shape[0]
    
    # Simulate 95% confidence bands
    drfos_sim = np.random.multivariate_normal(
        tau_fc_low.reshape(m),
        sigma_hat,
        10000
    )
    lower_bound = np.quantile(drfos_sim, 0.05, axis=0)
    upper_bound = np.quantile(drfos_sim, 0.95, axis=0)
    
    

    # Plot CI
    ax_fate2d.fill_between(range(len(time)), lower_bound, upper_bound, alpha=0.35)
    
    
    #X_high
    

    v_pred_flat = (
        RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0
        )
        .fit(X, V_flat)
        .predict(X_high.reshape(1, -1))
    )



    
    

    # 6. Reconstruct the m x m matrix
    sigma_hat = np.zeros((m, m))
    sigma_hat[triu_indices] = v_pred_flat
    sigma_hat = sigma_hat + sigma_hat.T - np.diag(sigma_hat.diagonal())
    sigma_hat /= tau_i.shape[0]
    
    # Simulate 95% confidence bands
    drfos_sim = np.random.multivariate_normal(
        tau_fc_high.reshape(m),
        sigma_hat,
        10000
    )
    lower_bound = np.quantile(drfos_sim, 0.05, axis=0)
    upper_bound = np.quantile(drfos_sim, 0.95, axis=0)
    
    

    # Plot CI
    ax_fate2d.fill_between(range(len(time)), lower_bound, upper_bound, alpha=0.35)
    

    


    # Position tweak
    pos = ax_fate2d.get_position()
    ax_fate2d.set_position([
        pos.x0 + 0.1 * pos.width,
        pos.y0 + 0.03,
        pos.width * 1.1,
        pos.height * 0.75
    ])

    
    
    


# ============================================================
# Build OUTER figure
# ============================================================
fig = plt.figure(figsize=(20, 12))

outer = gridspec.GridSpec(
    nrows=2,
    ncols=1,
    height_ratios=[0.4, 1.0],
    hspace=0.55
)

# ============================================================
# Row 1 — Covariate densities
# ============================================================
gs_dens = gridspec.GridSpecFromSubplotSpec(
    1, 7,
    subplot_spec=outer[0],
    wspace=1.15
)

df = res_wave1["data"]
df22 = res_wave2["data"]
A  = res_wave1["A"]

treated   = A == 1
untreated = A == 0

for i, cov_name in enumerate(covariates[:5]):
    ax = fig.add_subplot(gs_dens[0, i])
    cov = df[cov_name]

    if cov.nunique() <= 2:
        ax.bar(
            ["Untreated", "Treated"],
            [cov[untreated].mean(), cov[treated].mean()],
            color=[COLORS["untreated"], COLORS["treated"]],
            alpha=ALPHA
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion" if i == 0 else "")
    else:
        sns.kdeplot(
            cov[treated],
            fill=True,
            color=COLORS["treated"],
            alpha=ALPHA,
            label="Treated" if i == 0 else None,
            common_norm=False,
            ax=ax
        )
        sns.kdeplot(
            cov[untreated],
            fill=True,
            color=COLORS["untreated"],
            alpha=ALPHA,
            label="Untreated" if i == 0 else None,
            common_norm=False,
            ax=ax
        )
        ax.set_ylabel("Density" if i == 0 else "")
    if i==0:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(4, 1.3),
            frameon=False,
            ncol=2
        )


    ax.set_title(strip_year(cov_name))
    ax.set_xlabel("")
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + 0.2 * pos.width,
        pos.y0 + 0.03,
        pos.width * 1.4,
        pos.height * 0.95
    ])
ax = fig.add_subplot(gs_dens[0, 5])
cov = df["area_before"]

sns.kdeplot(
    cov[treated],
    fill=True,
    color=COLORS["treated"],
    alpha=ALPHA,
    common_norm=False,
    ax=ax
)
sns.kdeplot(
    cov[untreated],
    fill=True,
    color=COLORS["untreated"],
    alpha=ALPHA,
    common_norm=False,
    ax=ax
)

ax.set_title("area_before_wave1")
ax.set_xlabel("")
ax.set_ylabel("")
pos = ax.get_position()
ax.set_position([
        pos.x0 + 0.2 * pos.width,
        pos.y0 + 0.03,
        pos.width * 1.4,
        pos.height * 0.95
    ])
ax = fig.add_subplot(gs_dens[0, 6])
cov = df22["area_before"]

sns.kdeplot(
    cov[treated],
    fill=True,
    color=COLORS["treated"],
    alpha=ALPHA,
    common_norm=False,
    ax=ax
)
sns.kdeplot(
    cov[untreated],
    fill=True,
    color=COLORS["untreated"],
    alpha=ALPHA,
    common_norm=False,
    ax=ax
)

ax.set_title("area_before_wave2")
ax.set_xlabel("")
ax.set_ylabel("")
pos = ax.get_position()
ax.set_position([
        pos.x0 + 0.2 * pos.width,
        pos.y0 + 0.03,
        pos.width * 1.4,
        pos.height * 0.95
    ])


# ============================================================
# Row 2 — Wave 1 | Wave 2
# ============================================================
gs_waves = gridspec.GridSpecFromSubplotSpec(
    1, 2,
    subplot_spec=outer[1],
    wspace=0.28
)

plot_wave_column(fig, gs_waves[0], res_wave1, "Wave 1")
plot_wave_column(fig, gs_waves[1], res_wave2, "Wave 2")

# ============================================================
# Fancy frames
# ============================================================
def add_block_frame(fig, gs_cell, title, mask_frac=0.36):
    bbox = gs_cell.get_position(fig)
    pad_x, pad_y = 0.022, 0.018

    x0 = bbox.x0 - pad_x
    y0 = bbox.y0 - pad_y
    w  = bbox.width + 3 * pad_x
    h  = bbox.height + 6.8 * pad_y

    frame = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        transform=fig.transFigure,
        fill=False, lw=1.5, edgecolor="grey"
    )
    fig.add_artist(frame)

    tx, ty = x0 + w/2, y0 + h + 0.01
    fig.text(tx, ty, title, ha="center", va="center",
             fontsize=16, weight="bold")

    mask = FancyBboxPatch(
        (tx - mask_frac*w/2, ty - 0.014),
        mask_frac*w, 0.015,
        boxstyle="round,pad=0.0,rounding_size=0.01",
        transform=fig.transFigure,
        facecolor="white", edgecolor="none"
    )
    fig.add_artist(mask)

add_block_frame(fig, outer[0], "(a) Covariate densities", mask_frac=0.25)
add_block_frame(fig, gs_waves[0], "(b) Wave 1")
add_block_frame(fig, gs_waves[1], "(c) Wave 2")

plt.show()



