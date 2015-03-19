from pymc import *

import theano.tensor as t
from theano.tensor.nlinalg import matrix_inverse as inv
from numpy import array, diag
from numpy.random import multivariate_normal

n_obs = 1000

# Mean values:
mu = array([ 0.0006,  0.0011,  0.0003, -0.0002])
n_var = len(mu)

# Standard deviations:
stds = array([ 0.011,  0.009,  0.003,  0.031])

# Correlation matrix of 4 variables:
corr = array([[ 1.  ,  0.75,  0.  ,  0.15],
              [ 0.75,  1.  , -0.06,  0.19],
              [ 0.  , -0.06,  1.  , -0.04],
              [ 0.15,  0.19, -0.04,  1.  ]])
cov_matrix = diag(stds).dot(corr.dot(diag(stds)))

data = multivariate_normal(mu, cov_matrix, size=n_obs)

with Model() as model:
    
    mu = Normal('mu', mu=0, tau=1 ** -2, shape=n_var)

    # We can specify separate priors for sigma and the correlation matrix:
    sigma = Uniform('sigma', shape=n_var)
    corr_matrix = LKJCorr('corr', n=1, p=n_var)
    cov = t.diag(sigma).dot(corr_matrix.dot(t.diag(sigma)))

    like = MvNormal('likelihood', mu=mu, tau=inv(cov), observed=data)


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        step = NUTS()
        tr = sample(n, step=step)

if __name__ == '__main__':
    run()
