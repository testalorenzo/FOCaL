#
# FOCaL
#

from sklearn.model_selection import KFold
from numpy import zeros_like, array
from copy import deepcopy

class FOCaL:
    def __init__(self, nuisance_mu, nuisance_pi, final_regression):
        """
        Initialize the FOCaL learner.

        Parameters:
        nuisance_mu: A function that estimates the expected outcome given treatment and covariates.
        nuisance_pi: A function that estimates the propensity score given treatment and covariates.
        final_regression: A function that performs the final regression step to estimate the functional outcome.
        """

        self.nuisance_mu = nuisance_mu
        self.nuisance_pi = nuisance_pi
        self.final_regression = final_regression
        self.mu1_estimates = []
        self.mu0_estimates = []
        self.pi_estimates = []
        self.FCATE = []

    def fit(self, X, A, Y, n_folds=5):
        """
        Train the nuisance functions and compute pseudo-outcomes.

        Parameters:
        X: Covariates.
        A: Treatment assignments.
        Y: Functional outcomes.
        n_folds: Number of folds for cross-fitting.

        Returns:
        difference: The estimated treatment effect as a difference of pseudo-outcomes.
        """
        
        kf = KFold(n_splits=n_folds, shuffle=True)

        mu1_estimates = zeros_like(Y)
        mu0_estimates = zeros_like(Y)
        pi_estimates = zeros_like(A)
        FCATEs = []
        # train mu1 on treatment group and mu0 on control group
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            A_train, A_test = A[train_index], A[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            mu1_estimates = self.nuisance_mu.fit(X_train[A_train == 1], Y_train[A_train == 1]).predict(X_test)
            mu0_estimates = self.nuisance_mu.fit(X_train[A_train == 0], Y_train[A_train == 0]).predict(X_test)
            pi_estimates = self.nuisance_pi.fit(X_train, A_train).predict_proba(X_test)[:,1]

            # Clip pi_estimates to avoid division by zero
            pi_estimates = pi_estimates.clip(1e-6, 1 - 1e-6)

            # Store predictions
            self.mu1_estimates.append(mu1_estimates)
            self.mu0_estimates.append(mu0_estimates)
            self.pi_estimates.append(pi_estimates)

            # compute pseudo-outcomes
            pseudo_outcome1 = mu1_estimates + A_test[:,None] * (Y_test - mu1_estimates) / pi_estimates[:,None]
            pseudo_outcome0 = mu0_estimates + (1 - A_test[:,None]) * (Y_test - mu0_estimates) / (1 - pi_estimates[:,None])
            difference = pseudo_outcome1 - pseudo_outcome0

            # Final regression to estimate the functional outcome
            FCATE_split = self.final_regression.fit(X[test_index], difference)
            # Store list of fitted FCATE models
            self.FCATE.append(deepcopy(FCATE_split))

    def predict(self, X):
        """
        Predict the functional outcome for new data.

        Parameters:
        X: New covariates.

        Returns:
        Predicted functional outcomes.
        """
        
        # Compute predictions for each fitted FCATE model and average them
        predictions = array([fcate.predict(X) for fcate in self.FCATE])

        return predictions.mean(axis=0)





