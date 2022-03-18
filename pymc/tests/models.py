#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from itertools import product

import aesara
import aesara.tensor as at
import numpy as np

from aesara.compile.ops import as_op

import pymc as pm

from pymc import Categorical, Metropolis, Model, Normal
from pymc.aesaraf import floatX_array


def simple_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        Normal("x", mu, tau=tau, size=2, initval=floatX_array([0.1, 0.1]))

    return model.compute_initial_point(), model, (mu, tau**-0.5)


def simple_categorical():
    p = floatX_array([0.1, 0.2, 0.3, 0.4])
    v = floatX_array([0.0, 1.0, 2.0, 3.0])
    with Model() as model:
        Categorical("x", p, size=3, initval=[1, 2, 3])

    mu = np.dot(p, v)
    var = np.dot(p, (v - mu) ** 2)
    return model.compute_initial_point(), model, (mu, var)


def multidimensional_model():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        Normal("x", mu, tau=tau, size=(3, 2), initval=0.1 * np.ones((3, 2)))

    return model.compute_initial_point(), model, (mu, tau**-0.5)


def simple_arbitrary_det():
    scalar_type = at.dscalar if aesara.config.floatX == "float64" else at.fscalar

    @as_op(itypes=[scalar_type], otypes=[scalar_type])
    def arbitrary_det(value):
        return value

    with Model() as model:
        a = Normal("a")
        b = arbitrary_det(a)
        Normal("obs", mu=b.astype("float64"), observed=floatX_array([1, 3, 5]))

    return model.compute_initial_point(), model


def simple_init():
    start, model, moments = simple_model()
    step = Metropolis(model.value_vars, np.diag([1.0]), model=model)
    return model, start, step, moments


def simple_2model():
    mu = -2.1
    tau = 1.3
    p = 0.4
    with Model() as model:
        x = pm.Normal("x", mu, tau=tau, initval=0.1)
        pm.Deterministic("logx", at.log(x))
        pm.Bernoulli("y", p)
    return model.compute_initial_point(), model


def simple_2model_continuous():
    mu = -2.1
    tau = 1.3
    with Model() as model:
        x = pm.Normal("x", mu, tau=tau, initval=0.1)
        pm.Deterministic("logx", at.log(x))
        pm.Beta("y", alpha=1, beta=1, size=2)
    return model.compute_initial_point(), model


def mv_simple():
    mu = floatX_array([-0.1, 0.5, 1.1])
    p = floatX_array([[2.0, 0, 0], [0.05, 0.1, 0], [1.0, -0.05, 5.5]])
    tau = np.dot(p, p.T)
    with pm.Model() as model:
        pm.MvNormal(
            "x",
            at.constant(mu),
            tau=at.constant(tau),
            initval=floatX_array([0.1, 1.0, 0.8]),
        )
    H = tau
    C = np.linalg.inv(H)
    return model.compute_initial_point(), model, (mu, C)


def mv_simple_coarse():
    mu = floatX_array([-0.2, 0.6, 1.2])
    p = floatX_array([[2.0, 0, 0], [0.05, 0.1, 0], [1.0, -0.05, 5.5]])
    tau = np.dot(p, p.T)
    with pm.Model() as model:
        pm.MvNormal(
            "x",
            at.constant(mu),
            tau=at.constant(tau),
            initval=floatX_array([0.1, 1.0, 0.8]),
        )
    H = tau
    C = np.linalg.inv(H)
    return model.compute_initial_point(), model, (mu, C)


def mv_simple_very_coarse():
    mu = floatX_array([-0.3, 0.7, 1.3])
    p = floatX_array([[2.0, 0, 0], [0.05, 0.1, 0], [1.0, -0.05, 5.5]])
    tau = np.dot(p, p.T)
    with pm.Model() as model:
        pm.MvNormal(
            "x",
            at.constant(mu),
            tau=at.constant(tau),
            initval=floatX_array([0.1, 1.0, 0.8]),
        )
    H = tau
    C = np.linalg.inv(H)
    return model.compute_initial_point(), model, (mu, C)


def mv_simple_discrete():
    d = 2
    n = 5
    p = floatX_array([0.15, 0.85])
    with pm.Model() as model:
        pm.Multinomial("x", n, at.constant(p), initval=np.array([1, 4]))
        mu = n * p
        # covariance matrix
        C = np.zeros((d, d))
        for (i, j) in product(range(d), range(d)):
            if i == j:
                C[i, i] = n * p[i] * (1 - p[i])
            else:
                C[i, j] = -n * p[i] * p[j]

    return model.compute_initial_point(), model, (mu, C)


def mv_prior_simple():
    n = 3
    noise = 0.1
    X = np.linspace(0, 1, n)[:, None]

    K = pm.gp.cov.ExpQuad(1, 1)(X).eval()
    L = np.linalg.cholesky(K)
    K_noise = K + noise * np.eye(n)
    obs = floatX_array([-0.1, 0.5, 1.1])

    # Posterior mean
    L_noise = np.linalg.cholesky(K_noise)
    alpha = np.linalg.solve(L_noise.T, np.linalg.solve(L_noise, obs))
    mu_post = np.dot(K.T, alpha)

    # Posterior standard deviation
    v = np.linalg.solve(L_noise, K)
    std_post = (K - np.dot(v.T, v)).diagonal() ** 0.5

    with pm.Model() as model:
        x = pm.Flat("x", size=n)
        x_obs = pm.MvNormal("x_obs", observed=obs, mu=x, cov=noise * np.eye(n))

    return model.compute_initial_point(), model, (K, L, mu_post, std_post, noise)


def non_normal(n=2):
    with pm.Model() as model:
        pm.Beta("x", 3, 3, size=n, transform=None)
    return model.compute_initial_point(), model, (np.tile([0.5], n), None)


def exponential_beta(n=2):
    with pm.Model() as model:
        pm.Beta("x", 3, 1, size=n, transform=None)
        pm.Exponential("y", 1, size=n, transform=None)
    return model.compute_initial_point(), model, None


def beta_bernoulli(n=2):
    with pm.Model() as model:
        pm.Beta("x", 3, 1, size=n, transform=None)
        pm.Bernoulli("y", 0.5)
    return model.compute_initial_point(), model, None


def simple_normal(bounded_prior=False):
    """Simple normal for testing MLE / MAP; probes issue #2482."""
    x0 = 10.0
    sigma = 1.0
    a, b = (9, 12)  # bounds for uniform RV, need non-symmetric to reproduce issue

    with pm.Model(rng_seeder=2482) as model:
        if bounded_prior:
            mu_i = pm.Uniform("mu_i", a, b)
        else:
            mu_i = pm.Flat("mu_i")
        pm.Normal("X_obs", mu=mu_i, sigma=sigma, observed=x0)

    return model.compute_initial_point(), model, None
