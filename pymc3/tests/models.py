from pymc3 import Model, Normal, Categorical, Metropolis
import numpy as np
import pymc3 as pm
from itertools import product
import theano.tensor as tt
from theano.compile.ops import as_op


def simple_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        Normal('x', mu, tau=tau, shape=2, testval=tt.ones(2) * .1)

    return model.test_point, model, (mu, tau ** -1)


def simple_categorical():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([0.0, 1.0, 2.0, 3.0])
    with Model() as model:
        Categorical('x', p, shape=3, testval=[1, 2, 3])

    mu = np.dot(p, v)
    var = np.dot(p, (v - mu) ** 2)
    return model.test_point, model, (mu, var)


def multidimensional_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        Normal('x', mu, tau=tau, shape=(3, 2), testval=.1 * tt.ones((3, 2)))

    return model.test_point, model, (mu, tau ** -1)


def simple_arbitrary_det():
    @as_op(itypes=[tt.dscalar], otypes=[tt.dscalar])
    def arbitrary_det(value):
        return value

    with Model() as model:
        a = Normal('a')
        b = arbitrary_det(a)
        Normal('obs', mu=b.astype('float64'), observed=np.array([1, 3, 5]))

    return model.test_point, model


def simple_init():
    start, model, moments = simple_model()
    step = Metropolis(model.vars, np.diag([1.]), model=model)
    return model, start, step, moments


def simple_2model():
    mu = -2.1
    tau = 1.3
    p = .4
    with Model() as model:
        x = pm.Normal('x', mu, tau=tau, testval=.1)
        pm.Deterministic('logx', tt.log(x))
        pm.Bernoulli('y', p)
    return model.test_point, model


def mv_simple():
    mu = np.array([-.1, .5, 1.1])
    p = np.array([
        [2., 0, 0],
        [.05, .1, 0],
        [1., -0.05, 5.5]])
    tau = np.dot(p, p.T)
    with pm.Model() as model:
        pm.MvNormal('x', tt.constant(mu), tau=tt.constant(tau),
                    shape=3, testval=np.array([.1, 1., .8]))
    H = tau
    C = np.linalg.inv(H)
    return model.test_point, model, (mu, C)


def mv_simple_discrete():
    d = 2
    n = 5
    p = np.array([.15, .85])
    with pm.Model() as model:
        pm.Multinomial('x', n, tt.constant(p), shape=d, testval=np.array([1, 4]))
        mu = n * p
        # covariance matrix
        C = np.zeros((d, d))
        for (i, j) in product(range(d), range(d)):
            if i == j:
                C[i, i] = n * p[i] * (1 - p[i])
            else:
                C[i, j] = -n * p[i] * p[j]

    return model.test_point, model, (mu, C)


def mv_prior_simple():
    n = 3
    noise = 0.1
    X = np.linspace(0, 1, n)[:, None]

    K = pm.gp.cov.ExpQuad(1, 1)(X).eval()
    L = np.linalg.cholesky(K)
    K_noise = K + noise * np.eye(n)
    obs = np.array([-0.1, 0.5, 1.1])

    # Posterior mean
    L_noise = np.linalg.cholesky(K_noise)
    alpha = np.linalg.solve(L_noise.T, np.linalg.solve(L_noise, obs))
    mu_post = np.dot(K.T, alpha)

    # Posterior standard deviation
    v = np.linalg.solve(L_noise, K)
    std_post = (K - np.dot(v.T, v)).diagonal() ** 0.5

    with pm.Model() as model:
        x = pm.Flat('x', shape=n)
        x_obs = pm.MvNormal('x_obs', observed=obs, mu=x,
                            cov=noise * np.eye(n), shape=n)

    return model.test_point, model, (K, L, mu_post, std_post, noise)


def non_normal(n=2):
    with pm.Model() as model:
        pm.Beta('x', 3, 3, shape=n, transform=None)
    return model.test_point, model, (np.tile([.5], n), None)


def exponential_beta(n=2):
    with pm.Model() as model:
        pm.Beta('x', 3, 1, shape=n, transform=None)
        pm.Exponential('y', 1, shape=n, transform=None)
    return model.test_point, model, None


def beta_bernoulli(n=2):
    with pm.Model() as model:
        pm.Beta('x', 3, 1, shape=n, transform=None)
        pm.Bernoulli('y', 0.5)
    return model.test_point, model, None
