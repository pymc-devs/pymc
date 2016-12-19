import unittest
import numpy as np
import theano
from pymc3 import Model, Normal
from pymc3.variational.replacements import MeanField


class TestMeanField(unittest.TestCase):
    def test_elbo(self):
        mu0 = 1.5
        sigma = 1.0
        y_obs = np.array([1.6, 1.4])

        post_mu = np.array([1.88])
        post_sd = np.array([1])
        # Create a model for test
        with Model() as model:
            mu = Normal('mu', mu=mu0, sd=sigma)
            Normal('y', mu=mu, sd=1, observed=y_obs)

        # Create variational gradient tensor
        mean_field = MeanField(model)
        elbo = mean_field.elbo(samples=10000)

        mean_field.shared_params['mu'].set_value(post_mu)
        mean_field.shared_params['rho'].set_value(np.log(np.exp(post_sd) - 1))

        f = theano.function([], elbo.mean())
        elbo_mc = f()

        # Exact value
        elbo_true = (-0.5 * (
            3 + 3 * post_mu**2 - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu +
            y_obs[0]**2 + y_obs[1]**2 + mu0**2 + 3 * np.log(2 * np.pi)) +
            0.5 * (np.log(2 * np.pi) + 1))
        np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)

if __name__ == '__main__':
    unittest.main()
