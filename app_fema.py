#
# FEMA application
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.decomposition import PCA

import FOCaL

if __name__ == "__main__":

    plot = True

    # Choose response
    var = "wages"

    # Load data
    X = pd.read_csv('./data/covariatesFEMA.csv', index_col=0)
    A = X['IS_D'].to_numpy()
    X.index = X.FIPS
    X = X.drop(columns=['FIPS', 'Area', 'IS_D'])
    X = pd.get_dummies(X, drop_first=True)
    states = X.filter(like='STATE_').columns
    all = X.columns
    cols = [col for col in all if col not in states]
    data = X.copy()
    X = X.to_numpy()

    if var == "wages":
        Y = pd.read_csv('./data/wages_smooth.csv', index_col=0)
    else:
        Y = pd.read_csv('./data/income_smooth.csv', index_col=0)

    Y = Y.to_numpy()

    n = X.shape[0]
    p = X.shape[1]
    t = Y.shape[1]

    # Plot difference in means
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(t), Y[A==1].mean(axis=0))
        plt.plot(range(t), Y[A==0].mean(axis=0))
        plt.tight_layout()
        plt.show()

    n_components = 7

    # Initialize FOCaL model
    focal = FOCaL.FOCaL(
        nuisance_mu=RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
        nuisance_pi=BalancedRandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1, sampling_strategy='auto', bootstrap=False, replacement=True),
        final_regression=RandomForestRegressor(random_state=42),
        fpca=n_components
    )

    # Fit the model
    focal.fit(X, A, Y, n_folds=2, random_state=42, clip_pi=True, DR=False)

    # Get the estimated functional average treatment effect
    fate = focal.get_FATE()

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
    #fcate = focal.predict(X).reshape(n, n_components)
    # fcate = pca.inverse_transform(fcate)

    # Predict marginal treatment effect wrt to each variable
    n_eval = 30
    for var in cols:
        var_range = np.linspace(data[var].min(), data[var].max(), n_eval)
        X_pred = np.tile(X.mean(axis=0), (n_eval, 1))
        X_pred[:, [x in states for x in all]] = 0 # Set all states to 0
        X_pred[:, all==var] = var_range.reshape(-1, 1)
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

    # Plot avg effect per state
    states_effects = []
    for state in states:
        X_pred = X.mean(axis=0)
        X_pred[[x in states for x in all]] = 0 # Set all states to 0
        X_pred[all==state] = 1
        X_pred = X_pred.reshape(1, -1)
        fcate_pred = focal.predict(X_pred)
        states_effects.append(fcate_pred.ravel())
        
    plt.figure(figsize=(10, 6))
    for state, fcate_pred in zip(states, np.array(states_effects)):
        if abs(fcate_pred.max()) > 100:
            continue
        plt.plot(range(len(fcate_pred)), fcate_pred, label=state)
    plt.title('Estimated Treatment Effect by State')
    plt.xlabel('Time Points')
    plt.ylabel('Estimated Treatment Effect')
    plt.legend()
    plt.tight_layout()
    plt.show()
