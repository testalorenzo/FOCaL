#
# SHARE Application
#

import pyreadr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

from FOCaL import FOCaL


if __name__ == "__main__":
    
    # Import Rdata
    result = pyreadr.read_r('./data/data_for_fasten.Rdata')
    result.keys()

    data = result['Amat'].to_numpy()
    description = result['descriptions_var']
    types = result['type_var']

    # Effect of hypertension on mobility index adjusted for covariates
    idx_treat = 16 # 17 
    idx_response = 0 # 9
    idx_covariates = [26, 27, 29, 34, 35, 36]

    good_subset = data[idx_treat,:,:].sum(axis=1) == 192 # 419
    control = data[idx_treat,:,:].sum(axis=1) == 0 # 577

    Y_treatment = data[idx_response,good_subset,:]
    Y_control = data[idx_response,control,:]

    X_treatment = data[:,good_subset,0]
    X_control = data[:,control,0]
    X_treatment = X_treatment[idx_covariates,:].T
    X_control = X_control[idx_covariates,:].T

    X = np.concatenate((X_treatment, X_control), axis=0)
    Y = np.concatenate((Y_treatment, Y_control), axis=0)
    A = np.concatenate((np.ones(sum(good_subset)), np.zeros(sum(control))), axis=0)
    
    n = X.shape[0]
    p = X.shape[1]

    # Initialize FOCaL
    focal = FOCaL(
        nuisance_mu=RandomForestRegressor(random_state=42),
        nuisance_pi=RandomForestClassifier(random_state=42),
        final_regression=DecisionTreeRegressor(random_state=42)
    )

    # Fit the model
    focal.fit(X, A, Y, n_folds=2)

    # Get the estimated treatment effect
    for model in focal.FCATE:
        text_representation = tree.export_text(model)
        print(text_representation)