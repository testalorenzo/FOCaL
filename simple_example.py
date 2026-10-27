"""
simple_example.py -- FOCaL on simple synthetic data: fit, predict and build a global confidence set.

Univariate covariate X, treatment A assigned with probability depending on X, functional outcome on
a grid of 20 points with a constant treatment effect of 2. Nuisance functions: linear regression
(outcome regressions, fit on functional principal component scores) and logistic regression
(propensity score). Third stage: least squares on (1, x), a linear smoother, so that the L2
confidence sets of the paper are available.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from FOCaL import FOCaL, FixedBasisLS

n, n_time = 500, 20
rng = np.random.default_rng(0)
X = rng.normal(size=(n, 1))
A = rng.binomial(1, 1 / (1 + np.exp(X[:, 0])))            # P(A = 1 | X) = logit^{-1}(-X)
t = np.linspace(0, 1, n_time)
tau = 2.0 * np.ones(n_time)                              # constant effect over time
noise = rng.normal(0, 0.5, size=(n, n_time))
Y0 = X[:, [0]] + np.sin(2 * np.pi * t) + noise           # baseline depends on X and on time
Y = Y0 + A[:, None] * tau

# Stage 1-3: cross-fitted nuisances, doubly robust pseudo-outcomes, third stage on (1, x)
focal = FOCaL(nuisance_mu=LinearRegression(), nuisance_pi=LogisticRegression(),
              final_regression=FixedBasisLS(), fpca=2, fpca_third_stage=False, eps=0.01)
focal.fit(X, X, A, Y, n_folds=5, random_state=0)

# F-CATE at new covariate values and FATE
X_new = np.array([[-1.0], [0.0], [1.0]])
tau_hat = focal.predict(X_new)                           # (3, n_time)
print('true effect (first 5 points):', tau[:5])
print('estimated F-CATE at x = 0 (first 5 points):', tau_hat[1, :5].round(2))
print('estimated FATE (first 5 points):', focal.get_FATE()[:5].round(2))

# 95% global confidence sets (L2 balls): radius, and whether the true curve lies in the set
dt = 1.0 / (n_time - 1)                                  # quadrature weight of the grid on [0, 1]
cs = focal.confidence_set(X_new, alpha=0.05, dt=dt)
for k, x in enumerate(X_new[:, 0]):
    inside = np.sqrt(dt * ((tau_hat[k] - tau) ** 2).sum()) <= cs['radius'][k]
    print(f'x = {x:+.0f}: L2 radius {cs["radius"][k]:.3f}, true curve inside the set: {inside}')
fcs = focal.fate_confidence_set(alpha=0.05, dt=dt)
print(f'FATE: L2 radius {fcs["radius"][0]:.3f}')
