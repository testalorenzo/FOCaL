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
