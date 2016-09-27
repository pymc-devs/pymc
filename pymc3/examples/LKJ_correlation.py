from pymc3 import *

import theano.tensor as tt
from theano.tensor.nlinalg import matrix_inverse as inv
from numpy import array, diag, linspace
from numpy.random import multivariate_normal

# Generate some multivariate normal data:
n_obs = 1000

# Mean values:
mu = linspace(0, 2, num=4)
n_var = len(mu)

# Standard deviations:
stds = np.ones(4) / 2.0

# Correlation matrix of 4 variables:
corr = array([[1.,  0.75,  0.,  0.15],
              [0.75,  1., -0.06,  0.19],
              [0., -0.06,  1., -0.04],
              [0.15,  0.19, -0.04,  1.]])
cov_matrix = diag(stds).dot(corr.dot(diag(stds)))

dataset = multivariate_normal(mu, cov_matrix, size=n_obs)


# In order to convert the upper triangular correlation values to a complete
# correlation matrix, we need to construct an index matrix:
n_elem = int(n_var * (n_var - 1) / 2)
tri_index = np.zeros([n_var, n_var], dtype=int)
tri_index[np.triu_indices(n_var, k=1)] = np.arange(n_elem)
tri_index[np.triu_indices(n_var, k=1)[::-1]] = np.arange(n_elem)

with Model() as model:

    mu = Normal('mu', mu=0, sd=1, shape=n_var)

    # We can specify separate priors for sigma and the correlation matrix:
    sigma = Uniform('sigma', shape=n_var)
    corr_triangle = LKJCorr('corr', n=1, p=n_var)
    corr_matrix = corr_triangle[tri_index]
    corr_matrix = tt.fill_diagonal(corr_matrix, 1)

    cov_matrix = tt.diag(sigma).dot(corr_matrix.dot(tt.diag(sigma)))

    like = MvNormal('likelihood', mu=mu, cov=cov_matrix, observed=dataset)


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        start = find_MAP()
        step = NUTS(scaling=start)
        tr = sample(n, step=step, start=start)

if __name__ == '__main__':
    run()
