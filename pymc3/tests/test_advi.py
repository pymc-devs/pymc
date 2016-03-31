import numpy as np
from pymc3 import Model, Normal, HalfNormal, DiscreteUniform, Poisson, switch, Exponential
from pymc3.theanof import inputvars
from pymc3.variational.advi import variational_gradient_estimate, advi, advi_minibatch
from theano import function
import theano.tensor as tt

from nose.tools import assert_raises

def test_elbo():
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])
        
    # Create a model for test
    with Model() as model:
        mu = Normal('mu', mu=mu0, sd=sigma)
        y = Normal('y', mu=mu, sd=1, observed=y_obs)

    vars = inputvars(model.vars)

    # Create variational gradient tensor
    grad, elbo, shared, uw = variational_gradient_estimate(
        vars, model, n_mcsamples=10000)

    # Variational posterior parameters
    uw_ = np.array([1.88, np.log(1)])

    # Calculate elbo computed with MonteCarlo
    f = function([uw], elbo)
    elbo_mc = f(uw_)

    # Exact value
    elbo_true = (-0.5 * (
        3 + 3 * uw_[0]**2 - 2 * (y_obs[0] + y_obs[1] + mu0) * uw_[0] +
        y_obs[0]**2 + y_obs[1]**2 + mu0**2 + 3 * np.log(2 * np.pi)) +
        0.5 * (np.log(2 * np.pi) + 1))

    np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)

disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                        2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                        3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], value=-999)
year = np.arange(1851, 1962)

def test_check_discrete():
    with Model() as disaster_model:
        switchpoint = DiscreteUniform('switchpoint', lower=year.min(), upper=year.max(), testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = Exponential('early_rate', 1)
        late_rate = Exponential('late_rate', 1)

        # Allocate appropriate Poisson rates to years before and after current
        rate = switch(switchpoint >= year, early_rate, late_rate)

        disasters = Poisson('disasters', rate, observed=disaster_data)

    # This should raise ValueError
    assert_raises(ValueError, advi, model=disaster_model, n=10)

def test_check_discrete_minibatch():
    disaster_data_t = tt.vector()
    disaster_data_t.tag.test_value = np.zeros(len(disaster_data))

    with Model() as disaster_model:

        switchpoint = DiscreteUniform(
            'switchpoint', lower=year.min(), upper=year.max(), testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = Exponential('early_rate', 1)
        late_rate = Exponential('late_rate', 1)

        # Allocate appropriate Poisson rates to years before and after current
        rate = switch(switchpoint >= year, early_rate, late_rate)

        disasters = Poisson('disasters', rate, observed=disaster_data_t)

    def create_minibatch():
        while True:
            return disaster_data

    # This should raise ValueError
    assert_raises(
        ValueError, advi_minibatch, model=disaster_model, n=10, 
        minibatch_RVs=[disasters], minibatch_tensors=[disaster_data_t], 
        minibatches=[create_minibatch()], verbose=False)
    