#
# Simple working example of FOCaL 
#

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern

from FOCaL import FOCaL

np.random.seed(42)  # For reproducibility

# Example data
n = 100  # Number of samples
p = 5  # Number of features
t = 100  # Number of time points

grid_t = np.linspace(0, 1, t)  # Time grid

X = np.random.normal(size=(n, p)) # Covariates
gamma = np.random.normal(0, 0.2, size=p) # Coefficients for the linear model
logit = X @ gamma # Linear predictor for the propensity score
pi = 1 / (1 + np.exp(-logit)) # Propensity score
A = np.random.binomial(1, pi) # Treatment assignment

# plt.hist(pi, bins=30, alpha=0.5, color='blue', label='Propensity Score Distribution')
# plt.title('Distribution of Propensity Scores')
# plt.xlabel('Propensity Score')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid()
# plt.show()

sd = 1 # Standard deviation of the error term
l = 0.25 # Length scale for the Matern kernel
nu = 3.5 # Smoothness parameter for the Matern kernel
cov = sd ** 2 * Matern(length_scale=l, nu=nu)(grid_t.reshape(-1, 1)) # Covariance matrix for the error term
Y0 = np.zeros((n, t))  # Initialize functional outcome for control group
Y1 = np.zeros((n, t))  # Initialize functional outcome for treatment group
for j in range(p):
    Y0 += np.outer(X[:, j], np.random.multivariate_normal(np.zeros(t), cov, 1))  # Functional outcome for control group
    Y1 += np.outer(X[:, j], np.random.multivariate_normal(np.zeros(t), cov, 1))  # Functional outcome for treatment group

# Add noise
Y0 += np.random.multivariate_normal(np.zeros(t), cov / 1000, n) # Add noise to control group
Y1 += np.random.multivariate_normal(np.zeros(t), cov / 1000, n) # Add noise to treatment group

Y = np.where(A[:, None] == 1, Y1, Y0) # Functional outcomes based on treatment assignment

# plt.figure()
# plt.plot(grid_t, Y0.T, label='Control Group', color='red')
# plt.plot(grid_t, Y1.T, label='Treatment Group', color='green')
# plt.title('Functional Outcomes for Control and Treatment Groups')
# plt.xlabel('Time')
# plt.ylabel('Functional Outcome')
# plt.legend()
# plt.grid()
# plt.show()  

# Define nuisance models
nuisance_mu = RandomForestRegressor()
nuisance_pi = RandomForestClassifier()
final_regression = RandomForestRegressor()

# Create FOCaL instance
focal = FOCaL(nuisance_mu, nuisance_pi, final_regression)

# Fit the model
focal.fit(X, A, Y, n_folds=5)

# Predict using the fitted model
predictions = focal.predict(X[0,:].reshape(1,-1))  # Predict for a single sample

# Plot the predictions
plt.figure()
plt.plot(grid_t, Y1[0,:] - Y0[0,:], label='Functional Pseudo Outcome', color='orange')
plt.plot(grid_t, predictions.reshape(-1,1), label='Predicted Functional Outcome', color='blue')
plt.title('Predicted Functional Outcome from FOCaL')
plt.xlabel('Time')
plt.ylabel('Functional Outcome')
plt.grid()
plt.show()

focal.mu1_estimates[0].shape