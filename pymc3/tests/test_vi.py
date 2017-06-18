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


@pytest.fixture('function', autouse=True)
def set_seed():
    np.random.seed(42)
    pm.set_tt_rng(42)


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


@pytest.fixture(
    'module',
    params=[
        dict(mini=True, scale=False),
        dict(mini=False, scale=True),
    ],
    ids=['mini-noscale', 'full-scale']
)
def minibatch_and_scaling(request):
    return request.param


@pytest.fixture('module')
def using_minibatch(minibatch_and_scaling):
    return minibatch_and_scaling['mini']


@pytest.fixture('module')
def scale_cost_to_minibatch(minibatch_and_scaling):
    return minibatch_and_scaling['scale']


@pytest.fixture('module')
def simple_model_data(using_minibatch):
    n = 1000
    sd0 = 2.
    mu0 = 4.
    sd = 3.
    mu = -5.

    data = sd * np.random.randn(n) + mu
    d = n / sd ** 2 + 1 / sd0 ** 2
    mu_post = (n * np.mean(data) / sd ** 2 + mu0 / sd0 ** 2) / d
    if using_minibatch:
        data = pm.Minibatch(data)
    return dict(
        n=n,
        data=data,
        mu_post=mu_post,
        d=d,
        mu0=mu0,
        sd0=sd0,
        sd=sd,
    )


@pytest.fixture(scope='module')
def simple_model(simple_model_data):
    with Model() as model:
        mu_ = Normal(
            'mu', mu=simple_model_data['mu0'],
            sd=simple_model_data['sd0'], testval=0)
        Normal('x', mu=mu_, sd=simple_model_data['sd'],
               observed=simple_model_data['data'],
               total_size=simple_model_data['n'])
    return model


@pytest.fixture(
    scope='function',
    params=[
        dict(cls=ADVI, init=dict()),
        dict(cls=FullRankADVI, init=dict()),
        dict(cls=SVGD, init=dict(n_particles=500)),
        dict(cls=ASVGD, init=dict(temperature=1.5)),
    ],
    ids=lambda d: d['cls'].__name__
)
def inference(request, simple_model, scale_cost_to_minibatch):
    cls = request.param['cls']
    init = request.param['init']
    with simple_model:
        return cls(scale_cost_to_minibatch=scale_cost_to_minibatch, **init)


@pytest.fixture('function')
def fit_kwargs(inference, using_minibatch):
    cb = [pm.callbacks.CheckParametersConvergence(
            every=500,
            diff='relative', tolerance=0.001),
          pm.callbacks.CheckParametersConvergence(
            every=500,
            diff='absolute', tolerance=0.0001)]
    _select = {
        (ADVI, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.02, n_win=50),
            n=5000, callbacks=cb
        ),
        (ADVI, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=12000, callbacks=cb
        ),
        (FullRankADVI, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=6000, callbacks=cb
        ),
        (FullRankADVI, 'mini'): dict(
            obj_optimizer=pm.rmsprop(learning_rate=0.001),
            n=12000, callbacks=cb
        ),
        (SVGD, 'full'): dict(
            obj_optimizer=pm.sgd(learning_rate=0.01),
            n=1000, callbacks=cb
        ),
        (SVGD, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=5),
            n=3000, callbacks=cb
        ),
        (ASVGD, 'full'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=1000, callbacks=cb
        ),
        (ASVGD, 'mini'): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50),
            n=3000, callbacks=cb
        )
    }
    if using_minibatch:
        key = 'mini'
    else:
        key = 'full'
    return _select[(type(inference), key)]


def test_fit(inference,
             fit_kwargs,
             simple_model_data
             ):
    trace = inference.fit(**fit_kwargs).sample(10000)
    mu_post = simple_model_data['mu_post']
    d = simple_model_data['d']
    np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.05)
    np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.1)
