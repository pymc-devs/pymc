import pytest
import pickle
import functools
import numpy as np
from theano import theano, tensor as tt
import pymc3 as pm
from pymc3 import Model, Normal
from pymc3.variational import (
    ADVI, FullRankADVI, SVGD,
    Empirical, ASVGD,
    MeanField, fit
)
from pymc3.variational.operators import KL

from pymc3.tests import models
from pymc3.tests.helpers import SeededTest

def test_elbo():
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])

    post_mu = np.array([1.88], dtype=theano.config.floatX)
    post_sd = np.array([1], dtype=theano.config.floatX)
    # Create a model for test
    with Model() as model:
        mu = Normal('mu', mu=mu0, sd=sigma)
        Normal('y', mu=mu, sd=1, observed=y_obs)

    # Create variational gradient tensor
    mean_field = MeanField(model=model)
    elbo = -KL(mean_field)()(10000)

    mean_field.shared_params['mu'].set_value(post_mu)
    mean_field.shared_params['rho'].set_value(np.log(np.exp(post_sd) - 1))

    f = theano.function([], elbo)
    elbo_mc = f()

    # Exact value
    elbo_true = (-0.5 * (
        3 + 3 * post_mu ** 2 - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu +
        y_obs[0] ** 2 + y_obs[1] ** 2 + mu0 ** 2 + 3 * np.log(2 * np.pi)) +
                 0.5 * (np.log(2 * np.pi) + 1))
    np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)


@pytest.fixture(scope='module', params=['mini', 'full'])
def simple_model_data(request):
    n = 1000
    sd0 = 2.
    mu0 = 4.
    sd = 3.
    mu = -5.

    data = sd * np.random.randn(n) + mu
    if request.param == 'mini':
        data = pm.Minibatch(data)
    d = n / sd ** 2 + 1 / sd0 ** 2
    mu_post = (n * np.mean(data) / sd ** 2 + mu0 / sd0 ** 2) / d
    return dict(
        data=data,
        mu_post=mu_post,
        mu0=mu0,
        sd0=sd0,
        sd=sd
    )


@pytest.fixture(scope='module')
def simple_model(simple_model_data):
    with Model():
        mu_ = Normal(
            'mu', mu=simple_model_data['mu0'],
            sd=simple_model_data['sd0'], testval=0)
        Normal('x', mu=mu_, sd=simple_model_data['sd'],
               observed=simple_model_data['data'])


