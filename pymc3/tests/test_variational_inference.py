import unittest
import numpy as np
import theano
import pymc3 as pm
from pymc3 import Model, Normal
from pymc3.variational.inference import (
    KL, MeanField, ADVI, FullRankADVI
)

from pymc3.tests import models
from pymc3.tests.helpers import SeededTest


class TestELBO(SeededTest):
    def test_elbo(self):
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
        elbo = -KL(mean_field)()(mean_field.random())

        mean_field.shared_params['mu'].set_value(post_mu)
        mean_field.shared_params['rho'].set_value(np.log(np.exp(post_sd) - 1))

        f = theano.function([], elbo)
        elbo_mc = sum(f() for _ in range(10000)) / 10000

        # Exact value
        elbo_true = (-0.5 * (
            3 + 3 * post_mu ** 2 - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu +
            y_obs[0] ** 2 + y_obs[1] ** 2 + mu0 ** 2 + 3 * np.log(2 * np.pi)) +
                     0.5 * (np.log(2 * np.pi) + 1))
        np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)


class TestApproximates:
    class Base(SeededTest):
        inference = None
        NITER = 30000

        def test_vars_view(self):
            _, model, _ = models.multidimensional_model()
            with model:
                app = self.inference().approx
                posterior = app.random(10)
                x_sampled = app.view(posterior, 'x').eval()
            self.assertEqual(x_sampled.shape, (10,) + model['x'].dshape)

        def test_sample_vp(self):
            n_samples = 100
            xs = np.random.binomial(n=1, p=0.2, size=n_samples)
            with pm.Model():
                p = pm.Beta('p', alpha=1, beta=1)
                pm.Binomial('xs', n=1, p=p, observed=xs)
                app = self.inference().approx
                trace = app.sample_vp(draws=1, hide_transformed=True)
                self.assertListEqual(trace.varnames, ['p'])
                self.assertEqual(len(trace), 1)
                trace = app.sample_vp(draws=10, hide_transformed=False)
                self.assertListEqual(sorted(trace.varnames), ['p', 'p_logodds_'])
                self.assertEqual(len(trace), 10)

        def test_optimizer_with_full_data(self):
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
                inf = self.inference()
                self.assertEqual(len(inf.hist), 0)
                inf.fit(10)
                self.assertEqual(len(inf.hist), 10)
                self.assertFalse(np.isnan(inf.hist).any())
                approx = inf.fit(self.NITER)
                self.assertEqual(len(inf.hist), self.NITER + 10)
                self.assertFalse(np.isnan(inf.hist).any())
                trace = approx.sample_vp(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.1)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.2)

        def test_optimizer_minibatch_with_generator(self):
            n = 1000
            sd0 = 2.
            mu0 = 4.
            sd = 3.
            mu = -5.

            data = sd * np.random.randn(n) + mu

            d = n / sd**2 + 1 / sd0**2
            mu_post = (n * np.mean(data) / sd**2 + mu0 / sd0**2) / d

            def create_minibatch(data):
                while True:
                    data = np.roll(data, 100, axis=0)
                    yield data[:100]

            minibatches = create_minibatch(data)
            with Model():
                mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
                Normal('x', mu=mu_, sd=sd, observed=minibatches, total_size=n)
                inf1 = self.inference()
                approx = inf1.fit(self.NITER)
                trace1 = approx.sample_vp(10000)
            np.testing.assert_allclose(np.mean(trace1['mu']), mu_post, rtol=0.4)
            np.testing.assert_allclose(np.std(trace1['mu']), np.sqrt(1. / d), rtol=0.4)

        def test_optimizer_minibatch_with_callback(self):
            n = 1000
            sd0 = 2.
            mu0 = 4.
            sd = 3.
            mu = -5.

            data = sd * np.random.randn(n) + mu

            d = n / sd ** 2 + 1 / sd0 ** 2
            mu_post = (n * np.mean(data) / sd ** 2 + mu0 / sd0 ** 2) / d

            def create_minibatch(data):
                while True:
                    data = np.roll(data, 100, axis=0)
                    yield data[:100]

            minibatches = create_minibatch(data)
            with Model():
                data_t = theano.shared(next(minibatches))

                def cb(*_):
                    data_t.set_value(next(minibatches))
                mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
                Normal('x', mu=mu_, sd=sd, observed=data_t, total_size=n)
                inf2 = self.inference()
                approx = inf2.fit(self.NITER, callbacks=[cb])
                trace2 = approx.sample_vp(10000)
            np.testing.assert_allclose(np.mean(trace2['mu']), mu_post, rtol=0.4)
            np.testing.assert_allclose(np.std(trace2['mu']), np.sqrt(1. / d), rtol=0.4)


class TestMeanField(TestApproximates.Base):
    inference = ADVI


class TestFullRank(TestApproximates.Base):
    inference = FullRankADVI

if __name__ == '__main__':
    unittest.main()
