#
# FOCaL
#

from numpy import zeros_like, array, vstack
from copy import deepcopy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA

class FOCaL:
    def __init__(self, nuisance_mu, nuisance_pi, final_regression, fpca=False):
        """
        Initialize the FOCaL learner.

        Parameters:
        nuisance_mu: A function that estimates the expected outcome given treatment and covariates.
        nuisance_pi: A function that estimates the propensity score given treatment and covariates.
        final_regression: A function that performs the final regression step to estimate the functional outcome.
        fpca: Number of components for functional PCA. If False, no PCA is applied.
        """

        self.nuisance_mu = nuisance_mu
        self.nuisance_pi = nuisance_pi
        self.final_regression = final_regression
        self.mu1_estimates = []
        self.mu0_estimates = []
        self.pi_estimates = []
        self.FCATE = []
        self.FCATE_basis = []
        self.FATE = []
        self.fpca = fpca

    def fit(self, X, A, Y, n_folds=5, balanced_folds=True, random_state=12, clip_pi=True, DR=True):
        """
        Train the nuisance functions and compute pseudo-outcomes.

        Parameters:
        X: Covariates.
        A: Treatment assignments.
        Y: Functional outcomes.
        n_folds: Number of folds for cross-fitting.
        random_state: Random seed for reproducibility.
        clip_pi: Whether to clip propensity scores to avoid division by zero.
        DR: Whether to use doubly robust estimation.
        fpca: Number of components for functional PCA. If False, no PCA is applied.

        Returns:
        difference: The estimated treatment effect as a difference of pseudo-outcomes.
        """

        if Y.ndim != 1 and Y.shape[1] == 1:
            Y = Y.ravel()

        if balanced_folds:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            gen_kf = kf.split(X, A)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            gen_kf = kf.split(X)

        mu1_estimates = zeros_like(Y)
        mu0_estimates = zeros_like(Y)
        pi_estimates = zeros_like(A)
        for train_index, test_index in gen_kf:
            X_train, X_test = X[train_index], X[test_index]
            A_train, A_test = A[train_index], A[test_index]

            if self.fpca is not False:
                pca = PCA(n_components=self.fpca)
                Y_train = pca.fit_transform(Y[train_index])
                Y_test = Y[test_index]
            else:
                Y_train, Y_test = Y[train_index], Y[test_index]

            if Y_train.ndim != 1 and Y_train.shape[1] == 1:
                Y_train = Y_train.ravel()

            # train mu1 on treatment group and mu0 on control group
            mu1_estimates = self.nuisance_mu.fit(X_train[A_train == 1], Y_train[A_train == 1]).predict(X_test)
            mu0_estimates = self.nuisance_mu.fit(X_train[A_train == 0], Y_train[A_train == 0]).predict(X_test)
            pi_estimates = self.nuisance_pi.fit(X_train, A_train).predict_proba(X_test)[:,1]

            # Clip pi_estimates to avoid division by zero
            if clip_pi:
                pi_estimates = pi_estimates.clip(1e-6, 1 - 1e-6)

            # compute pseudo-outcomes (reverse PCA if needed)
            if self.fpca is not False:
                mu1_estimates = pca.inverse_transform(mu1_estimates.reshape(-1, self.fpca))
                mu0_estimates = pca.inverse_transform(mu0_estimates.reshape(-1, self.fpca))
            else:
                mu1_estimates = mu1_estimates
                mu0_estimates = mu0_estimates

            # Store predictions
            self.mu1_estimates.append(mu1_estimates)
            self.mu0_estimates.append(mu0_estimates)
            self.pi_estimates.append(pi_estimates)

            # Ensure the shapes are compatible for broadcasting
            if Y.ndim == 1:
                mu1_estimates = mu1_estimates[:, None]
                mu0_estimates = mu0_estimates[:, None]
                Y_test = Y_test[:, None]

            if DR:
                pseudo_outcome1 = mu1_estimates + A_test[:,None] * (Y_test - mu1_estimates) / pi_estimates[:,None]
                pseudo_outcome0 = mu0_estimates + (1 - A_test[:,None]) * (Y_test - mu0_estimates) / (1 - pi_estimates[:,None])
            else:
                pseudo_outcome1 = mu1_estimates
                pseudo_outcome0 = mu0_estimates
            
            difference = pseudo_outcome1 - pseudo_outcome0
            self.FATE.append(difference)

            # Refit fPCA
            if self.fpca is not False:
                pca = PCA(n_components=self.fpca)
                difference_model = pca.fit_transform(difference)
            else:
                difference_model = difference

            if difference_model.ndim != 1 and difference_model.shape[1] == 1:
                difference_model = difference_model.ravel()

            # Final regression to estimate the functional outcome
            FCATE_split = self.final_regression.fit(X_test, difference_model)
            # Store list of fitted FCATE models
            if self.fpca is not False:
                self.FCATE_basis.append(deepcopy(pca))
            self.FCATE.append(deepcopy(FCATE_split))


    def predict(self, x):
        """
        Predict the functional outcome for new data.

        Parameters:
        x: New covariates.

        Returns:
        Predicted functional outcomes.
        """
        
        # Compute predictions for each fitted FCATE model and average them
        predictions = [fcate.predict(x) for fcate in self.FCATE]
        if self.fpca is not False:
            # Inverse transform if PCA was applied
            predictions = array([self.FCATE_basis[i].inverse_transform(predictions[i].reshape(-1, self.fpca)) for i in range(len(predictions))])
        return predictions.mean(axis=0)
    
    def get_FATE(self):
        """
        Get the estimated functional average treatment effect.

        Returns:
        FATE: The estimated functional average treatment effect.
        """
        # Average the estimates across folds
        if self.FATE[0].ndim == 1:
            self.FATE = vstack([x.reshape(x.shape[0],1) for x in self.FATE]).mean(axis=0)
        else:
            self.FATE = vstack(self.FATE).mean(axis=0)

        return self.FATE





