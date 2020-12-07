import numpy as np
import theano.tensor as tt

from numpy.random import multivariate_normal

import pymc3 as pm

# Generate some multivariate normal data:
n_obs = 1000

# Mean values:
mu_r = np.linspace(0, 2, num=4)
n_var = len(mu_r)

# Standard deviations:
stds = np.ones(4) / 2.0

# Correlation matrix of 4 variables:
corr_r = np.array(
    [
        [1.0, 0.75, 0.0, 0.15],
        [0.75, 1.0, -0.06, 0.19],
        [0.0, -0.06, 1.0, -0.04],
        [0.15, 0.19, -0.04, 1.0],
    ]
)
cov_matrix = np.diag(stds).dot(corr_r.dot(np.diag(stds)))

dataset = multivariate_normal(mu_r, cov_matrix, size=n_obs)

with pm.Model() as model:

    mu = pm.Normal("mu", mu=0, sigma=1, shape=n_var)

    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov("chol_cov", n=n_var, eta=1, sd_dist=sd_dist)
    # compute the covariance matrix
    chol = pm.expand_packed_triangular(n_var, packed_chol, lower=True)
    cov = tt.dot(chol, chol.T)

    # Extract the standard deviations etc
    sd = pm.Deterministic("sd", tt.sqrt(tt.diag(cov)))
    corr = tt.diag(sd ** -1).dot(cov.dot(tt.diag(sd ** -1)))
    r = pm.Deterministic("r", corr[np.triu_indices(n_var, k=1)])

    like = pm.MvNormal("likelihood", mu=mu, chol=chol, observed=dataset)


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        trace = pm.sample(n)
    pm.traceplot(
        trace, varnames=["mu", "r"], lines={"mu": mu_r, "r": corr_r[np.triu_indices(n_var, k=1)]}
    )


if __name__ == "__main__":
    run()
