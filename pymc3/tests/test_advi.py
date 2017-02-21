import numpy as np
import pymc3 as pm
from pymc3 import Model, Normal, DiscreteUniform, Poisson, Exponential
from pymc3.theanof import inputvars
from pymc3.variational import advi, advi_minibatch, sample_vp
from pymc3.variational.advi import _calc_elbo, adagrad_optimizer
from pymc3.theanof import CallableTensor
from theano import function, shared
import theano.tensor as tt

from .helpers import SeededTest, TestHandler, Matcher


class TestADVI(SeededTest):
    def setUp(self):
        super(TestADVI, self).setUp()
        self.disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                                 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                                 2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                                                 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                                 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                                 3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                                                 value=-999)
        self.year = np.arange(1851, 1962)

        self.handler = h = TestHandler(Matcher())
        pm._log.addHandler(h)

    def tearDown(self):
        pm._log.removeHandler(self.handler)
        self.handler.close()

    def test_elbo(self):
        mu0 = 1.5
        sigma = 1.0
        y_obs = np.array([1.6, 1.4])

        # Create a model for test
        with Model() as model:
            mu = Normal('mu', mu=mu0, sd=sigma)
            Normal('y', mu=mu, sd=1, observed=y_obs)

        model_vars = inputvars(model.vars)

        # Create variational gradient tensor
        elbo, _ = _calc_elbo(model_vars, model, n_mcsamples=10000, random_seed=self.random_seed)

        # Variational posterior parameters
        uw_ = np.array([1.88, np.log(1)])

        # Calculate elbo computed with MonteCarlo
        uw_shared = shared(uw_, 'uw_shared')
        elbo = CallableTensor(elbo)(uw_shared)
        f = function([], elbo)
        elbo_mc = f()

        # Exact value
        elbo_true = (-0.5 * (
            3 + 3 * uw_[0]**2 - 2 * (y_obs[0] + y_obs[1] + mu0) * uw_[0] +
            y_obs[0]**2 + y_obs[1]**2 + mu0**2 + 3 * np.log(2 * np.pi)) +
            0.5 * (np.log(2 * np.pi) + 1))

        np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)

    def test_check_discrete(self):
        with Model():
            switchpoint = DiscreteUniform(
                'switchpoint', lower=self.year.min(), upper=self.year.max(), testval=1900)

            # Priors for pre- and post-switch rates number of disasters
            early_rate = Exponential('early_rate', 1)
            late_rate = Exponential('late_rate', 1)

            # Allocate appropriate Poisson rates to years before and after current
            rate = tt.switch(switchpoint >= self.year, early_rate, late_rate)
            Poisson('disasters', rate, observed=self.disaster_data)

            # This should raise ValueError
            with self.assertRaises(ValueError):
                advi(n=10)

    def test_check_discrete_minibatch(self):
        disaster_data_t = tt.vector()
        disaster_data_t.tag.test_value = np.zeros(len(self.disaster_data))

        def create_minibatches():
            while True:
                return (self.disaster_data,)

        with Model():
            switchpoint = DiscreteUniform(
                'switchpoint', lower=self.year.min(), upper=self.year.max(), testval=1900)

            # Priors for pre- and post-switch rates number of disasters
            early_rate = Exponential('early_rate', 1)
            late_rate = Exponential('late_rate', 1)

            # Allocate appropriate Poisson rates to years before and after current
            rate = tt.switch(switchpoint >= self.year, early_rate, late_rate)
            disasters = Poisson('disasters', rate, observed=disaster_data_t)

            with self.assertRaises(ValueError):
                advi_minibatch(n=10, minibatch_RVs=[disasters], minibatch_tensors=[disaster_data_t],
                               minibatches=create_minibatches())

    def test_advi(self):
        n = 1000
        sd0 = 2.
        mu0 = 4.
        sd = 3.
        mu = -5.

        data = sd * np.random.randn(n) + mu

        d = n / sd**2 + 1 / sd0**2
        mu_post = (n * np.mean(data) / sd**2 + mu0 / sd0**2) / d

        with Model():
            mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
            Normal('x', mu=mu_, sd=sd, observed=data)
            advi_fit = advi(n=1000, accurate_elbo=False, learning_rate=1e-1)
            np.testing.assert_allclose(advi_fit.means['mu'], mu_post, rtol=0.1)
            trace = sample_vp(advi_fit, 10000)

        np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.4)
        np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)

        h = self.handler
        self.assertTrue(h.matches(msg="converged"))

        # Test for n < 10
        with Model():
            mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
            Normal('x', mu=mu_, sd=sd, observed=data)
            advi_fit = advi(n=5, accurate_elbo=False, learning_rate=1e-1)

        # Check to raise NaN with a large learning coefficient
        with self.assertRaises(FloatingPointError):
            with Model():
                mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
                Normal('x', mu=mu_, sd=sd, observed=data)
                advi_fit = advi(n=1000, accurate_elbo=False, learning_rate=1e10)

    def test_advi_optimizer(self):
        n = 1000
        sd0 = 2.
        mu0 = 4.
        sd = 3.
        mu = -5.

        data = sd * np.random.randn(n) + mu

        d = n / sd**2 + 1 / sd0**2
        mu_post = (n * np.mean(data) / sd**2 + mu0 / sd0**2) / d

        with Model():
            mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
            Normal('x', mu=mu_, sd=sd, observed=data)
            optimizer = adagrad_optimizer(learning_rate=0.1, epsilon=0.1)
            advi_fit = advi(n=1000, optimizer=optimizer)
            np.testing.assert_allclose(advi_fit.means['mu'], mu_post, rtol=0.1)
            trace = sample_vp(advi_fit, 10000)

        np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.4)
        np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)

    def test_advi_minibatch(self):
        n = 1000
        sd0 = 2.
        mu0 = 4.
        sd = 3.
        mu = -5.

        data = sd * np.random.randn(n) + mu

        d = n / sd**2 + 1 / sd0**2
        mu_post = (n * np.mean(data) / sd**2 + mu0 / sd0**2) / d

        data_t = tt.vector()
        data_t.tag.test_value = np.zeros(1,)

        def create_minibatch(data):
            while True:
                data = np.roll(data, 100, axis=0)
                yield (data[:100],)

        minibatches = create_minibatch(data)

        with Model():
            mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
            x = Normal('x', mu=mu_, sd=sd, observed=data_t)
            advi_fit = advi_minibatch(
                n=1000, minibatch_tensors=[data_t],
                minibatch_RVs=[x], minibatches=minibatches,
                total_size=n, learning_rate=1e-1)

            np.testing.assert_allclose(advi_fit.means['mu'], mu_post, rtol=0.1)
            trace = sample_vp(advi_fit, 10000)

        np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.4)
        np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)

        # Test for n < 10
        with Model():
            mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
            x = Normal('x', mu=mu_, sd=sd, observed=data_t)
            advi_fit = advi_minibatch(
                n=5, minibatch_tensors=[data_t],
                minibatch_RVs=[x], minibatches=minibatches,
                total_size=n, learning_rate=1e-1)

        # Check to raise NaN with a large learning coefficient
        with self.assertRaises(FloatingPointError):
            with Model():
                mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
                x = Normal('x', mu=mu_, sd=sd, observed=data_t)
                advi_fit = advi_minibatch(
                    n=1000, minibatch_tensors=[data_t],
                    minibatch_RVs=[x], minibatches=minibatches,
                    total_size=n, learning_rate=1e10)

    def test_advi_minibatch_shared(self):
        n = 1000
        sd0 = 2.
        mu0 = 4.
        sd = 3.
        mu = -5.

        data = sd * np.random.randn(n) + mu

        d = n / sd**2 + 1 / sd0**2
        mu_post = (n * np.mean(data) / sd**2 + mu0 / sd0**2) / d

        data_t = shared(np.zeros(1,))

        def create_minibatches(data):
            while True:
                data = np.roll(data, 100, axis=0)
                yield (data[:100],)

        with Model():
            mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
            x = Normal('x', mu=mu_, sd=sd, observed=data_t)
            advi_fit = advi_minibatch(
                n=1000, minibatch_tensors=[data_t], encoder_params=[],
                minibatch_RVs=[x], minibatches=create_minibatches(data),
                total_size=n, learning_rate=1e-1)
            np.testing.assert_allclose(advi_fit.means['mu'], mu_post, rtol=0.1)
            trace = sample_vp(advi_fit, 10000)

        np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.4)
        np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)

    def test_sample_vp(self):
        n_samples = 100
        xs = np.random.binomial(n=1, p=0.2, size=n_samples)
        with pm.Model():
            p = pm.Beta('p', alpha=1, beta=1)
            pm.Binomial('xs', n=1, p=p, observed=xs)
            v_params = advi(n=1000)
            trace = sample_vp(v_params, draws=1, hide_transformed=True)
            self.assertListEqual(trace.varnames, ['p'])
            trace = sample_vp(v_params, draws=1, hide_transformed=False)
            self.assertListEqual(sorted(trace.varnames), ['p', 'p_logodds_'])
