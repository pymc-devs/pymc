import unittest
import tqdm
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
from pymc3 import Model, Normal
from pymc3.variational.opvi import (
    KL, MeanField, FullRank, TestFunction
)

from pymc3.tests import models
from pymc3.tests.helpers import SeededTest


class TestApproximates:
    class Base(SeededTest):
        op = KL
        approx = MeanField
        tf = TestFunction
        NITER = 30000

        def test_elbo(self):
            if self.approx is not MeanField:
                raise unittest.SkipTest
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
            mean_field = self.approx(model=model)
            elbo = -self.op(mean_field)(self.tf())(mean_field.random())

            mean_field.shared_params['mu'].set_value(post_mu)
            mean_field.shared_params['rho'].set_value(np.log(np.exp(post_sd) - 1))

            f = theano.function([], elbo)
            elbo_mc = sum(f() for _ in range(10000))/10000

            # Exact value
            elbo_true = (-0.5 * (
                3 + 3 * post_mu**2 - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu +
                y_obs[0]**2 + y_obs[1]**2 + mu0**2 + 3 * np.log(2 * np.pi)) +
                0.5 * (np.log(2 * np.pi) + 1))
            np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)

        def test_vars_view(self):
            _, model, _ = models.multidimensional_model()
            with model:
                app = self.approx()
                posterior = app.random(10)
                x_sampled = app.view(posterior, 'x').eval()
            self.assertEqual(x_sampled.shape, (10,) + model['x'].dshape)

        def test_sample_vp(self):
            n_samples = 100
            xs = np.random.binomial(n=1, p=0.2, size=n_samples)
            with pm.Model():
                p = pm.Beta('p', alpha=1, beta=1)
                pm.Binomial('xs', n=1, p=p, observed=xs)
                app = self.approx()
                trace = app.sample_vp(draws=1, hide_transformed=True)
                self.assertListEqual(trace.varnames, ['p'])
                self.assertEqual(len(trace), 1)
                trace = app.sample_vp(draws=10, hide_transformed=False)
                self.assertListEqual(sorted(trace.varnames), ['p', 'p_logodds_'])
                self.assertEqual(len(trace), 10)

        def test_optimizer(self):
            n = 1000
            sd0 = 2.
            mu0 = 4.
            sd = 3.
            mu = -5.

            data = sd * np.random.randn(n) + mu

            d = n / sd ** 2 + 1 / sd0 ** 2
            mu_post = (n * np.mean(data) / sd ** 2 + mu0 / sd0 ** 2) / d

            with Model():
                mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
                Normal('x', mu=mu_, sd=sd, observed=data)
                pm.Deterministic('mu_sq', mu_**2)
                approx = self.approx()
                obj_f = self.op(approx)(self.tf())
                step = obj_f.step_function(score=False)
                for _ in tqdm.tqdm(range(self.NITER)):
                    step()
                trace = approx.sample_vp(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.1)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)


class TestMeanField(TestApproximates.Base):
    pass


class TestFullRank(TestApproximates.Base):
    approx = FullRank

if __name__ == '__main__':
    unittest.main()
