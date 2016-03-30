import numpy as np
from pymc3 import Model, Normal
from pymc3.theanof import inputvars
from pymc3.variational.advi import variational_gradient_estimate
from theano import function

def test_advi():
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
        vars, model, n_mcsamples=1000000)

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
