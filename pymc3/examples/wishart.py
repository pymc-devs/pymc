import pymc3 as pm
import numpy as np
import theano
import theano.tensor as T
import scipy.stats
import matplotlib.pyplot as plt

# Covariance matrix we want to recover
covariance = np.matrix([[2, .5, -.5],
                        [.5, 1.,  0.],
                        [-.5, 0., 0.5]])

prec = np.linalg.inv(covariance)

mean = [.5, 1, .2]
data = scipy.stats.multivariate_normal(mean, covariance).rvs(5000)

plt.scatter(data[:, 0], data[:, 1])

with pm.Model() as model:
    S = np.eye(3)
    nu = 5
    mu = pm.Normal('mu', mu=0, sd=1, shape=3)

    # Use the transformed Wishart distribution
    # Under the hood this will do a Cholesky decomposition
    # of S and add two RVs to the sampler: c and z
    prec = pm.WishartBartlett('prec', S, nu)

    # To be able to compare it to truth, convert precision to covariance
    cov = pm.Deterministic('cov', tt.nlinalg.matrix_inverse(prec))

    lp = pm.MvNormal('likelihood', mu=mu, tau=prec, observed=data)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        trace = pm.sample(n, step, start)

    pm.traceplot(trace)

if __name__ == '__main__':
    run()
