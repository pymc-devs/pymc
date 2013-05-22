from pymc import Model, Normal, Metropolis
import numpy as np
import pymc as pm


def simple_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        x = Normal('x', mu, tau, shape=2, testval=[.1]*2)

    return model.test_point, model, (mu, tau ** -1)


def simple_init():
    start, model, moments = simple_model()

    step = Metropolis(model.vars, np.diag([1.]), model=model)
    return model, start, step, moments


def simple_2model():
    mu = -2.1
    tau = 1.3
    p = .4
    with Model() as model:
        x = pm.Normal('x', mu, tau, testval=.1)
        y = pm.Bernoulli('y', p)

    return model.test_point, model


def mv_simple():
    mu = np.array([-.1, .5, 1.1])
    p = np.array([
        [2., 0, 0],
        [.05, .1, 0],
        [1., -0.05, 5.5]])

    tau = np.dot(p, p.T)

    with pm.Model() as model:
        x = pm.multivariate.Normal('x', pm.constant(mu), pm.constant(
            tau), shape=3, testval=np.array([.1, 1., .8]))

    H = tau
    C = np.linalg.inv(H)

    return model.test_point, model, (mu, C)
