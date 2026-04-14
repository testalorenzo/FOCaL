import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from FOCaL import FOCaL


# FOCaL fit and predict with simple synthetic data:
# Univariate problem (X)
# Random treatment based on X (A)
# Constant effect over time (tau)
# Linear regression and logistic regression as nuisance estimators

n = 200
n_time = 20
rng = np.random.default_rng(0)

X = rng.normal(size=(n, 1))

true_propensity_linear = -X[:, 0]
true_pi = 1 / (1 + np.exp(-true_propensity_linear))
A = rng.binomial(1, true_pi, n)

t = np.linspace(0, 1, n_time)

tau = np.ones(n_time) * 2.0   

Y = np.zeros((n, n_time))
for i in range(n):
    baseline = X[i, 0]        
    noise = rng.normal(0, 0.1, size=n_time)

    Y0 = baseline + noise
    Y1 = baseline + tau + noise

    Y[i] = A[i]*Y1 + (1-A[i])*Y0



mu_model = LinearRegression()
pi_model = LogisticRegression()
final_model = LinearRegression()


# Call the Class FOCaL
focal = FOCaL(mu_model, pi_model, final_model, fpca=2)

# Fit the data
focal.fit(X, X, A, Y, n_folds=2)


# Predict on the same input
tau_hat = focal.predict(X)


# You can also check the results
print("True tau (first 5):", tau[:5])
print("Estimated tau (first individual, first 5):", tau_hat[0, :5])
