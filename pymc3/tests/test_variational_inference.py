import pickle
import numpy as np
from theano import theano, tensor as tt
import pymc3 as pm
from pymc3 import Model, Normal
from pymc3.variational import (
    ADVI, FullRankADVI,
    Histogram,
    fit
)
from pymc3.variational.operators import KL
from pymc3.variational.approximations import MeanField

from pymc3.tests import models
from pymc3.tests.helpers import SeededTest
import pytest


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
            assert x_sampled.shape == (10,) + model['x'].dshape

        def test_vars_view_dynamic_size(self):
            _, model, _ = models.multidimensional_model()
            with model:
                app = self.inference().approx
                i = tt.iscalar('i')
                i.tag.test_value = 1
                posterior = app.random(i)
            x_sampled = app.view(posterior, 'x').eval({i: 10})
            assert x_sampled.shape == (10,) + model['x'].dshape
            x_sampled = app.view(posterior, 'x').eval({i: 1})
            assert x_sampled.shape == (1,) + model['x'].dshape

        def test_vars_view_dynamic_size_numpy(self):
            _, model, _ = models.multidimensional_model()
            with model:
                app = self.inference().approx
                i = tt.iscalar('i')
                i.tag.test_value = 1
            x_sampled = app.view(app.random_fn(10), 'x')
            assert x_sampled.shape == (10,) + model['x'].dshape
            x_sampled = app.view(app.random_fn(1), 'x')
            assert x_sampled.shape == (1,) + model['x'].dshape
            x_sampled = app.view(app.random_fn(), 'x')
            assert x_sampled.shape == () + model['x'].dshape

        def test_sample_vp(self):
            n_samples = 100
            xs = np.random.binomial(n=1, p=0.2, size=n_samples)
            with pm.Model():
                p = pm.Beta('p', alpha=1, beta=1)
                pm.Binomial('xs', n=1, p=p, observed=xs)
                app = self.inference().approx
                trace = app.sample_vp(draws=1, hide_transformed=True)
                assert trace.varnames == ['p']
                assert len(trace) == 1
                trace = app.sample_vp(draws=10, hide_transformed=False)
                assert sorted(trace.varnames) == ['p', 'p_logodds_']
                assert len(trace) == 10

        def test_sample_node(self):
            n_samples = 100
            xs = np.random.binomial(n=1, p=0.2, size=n_samples)
            with pm.Model():
                p = pm.Beta('p', alpha=1, beta=1)
                pm.Binomial('xs', n=1, p=p, observed=xs)
                app = self.inference().approx
            app.sample_node(p).eval()   # should be evaluated without errors

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
                assert len(inf.hist) == 0
                inf.fit(10)
                assert len(inf.hist) == 10
                assert not np.isnan(inf.hist).any()
                approx = inf.fit(self.NITER)
                assert len(inf.hist) == self.NITER + 10
                assert not np.isnan(inf.hist).any()
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
                inf = self.inference()
                approx = inf.fit(self.NITER)
                trace = approx.sample_vp(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.4)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)

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
                inf = self.inference()
                approx = inf.fit(self.NITER, callbacks=[cb], obj_n_mc=10)
                trace = approx.sample_vp(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.4)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.4)

        def test_pickling(self):
            with models.multidimensional_model()[1]:
                inference = self.inference()

            inference = pickle.loads(pickle.dumps(inference))
            inference.fit(20)

        def test_aevb(self):
            _, model, _ = models.exponential_beta(n=2)
            x = model.x
            y = model.y
            mu = theano.shared(x.init_value) * 2
            rho = theano.shared(np.zeros_like(x.init_value))
            with model:
                inference = self.inference(local_rv={y: (mu, rho)})
                approx = inference.fit(3, obj_n_mc=2)
                approx.sample_vp(10)
                approx.apply_replacements(
                    y,
                    more_replacements={x: np.asarray([1, 1], dtype=x.dtype)}
                ).eval()

        def test_profile(self):
            with models.multidimensional_model()[1]:
                self.inference().run_profiling(10)

        def test_multiple_replacements(self):
            _, model, _ = models.exponential_beta(n=2)
            x = model.x
            y = model.y
            xy = x*y
            xpy = x+y
            with model:
                mf = self.inference().approx
                xy_, xpy_ = mf.apply_replacements([xy, xpy])
                xy_s, xpy_s = mf.sample_node([xy, xpy])
                xy_.eval()
                xpy_.eval()
                xy_s.eval()
                xpy_s.eval()


class TestMeanField(TestApproximates.Base):
    inference = ADVI

    def test_approximate(self):
        with models.multidimensional_model()[1]:
            meth = ADVI()
            fit(10, method=meth)
            with pytest.raises(KeyError):
                fit(10, method='undefined')
            with pytest.raises(TypeError):
                fit(10, method=1)


class TestFullRank(TestApproximates.Base):
    inference = FullRankADVI

    def test_from_mean_field(self):
        with models.multidimensional_model()[1]:
            advi = ADVI()
            full_rank = FullRankADVI.from_mean_field(advi.approx)
            full_rank.fit(20)

    def test_from_advi(self):
        with models.multidimensional_model()[1]:
            advi = ADVI()
            full_rank = FullRankADVI.from_advi(advi)
            full_rank.fit(20)

    def test_combined(self):
        with models.multidimensional_model()[1]:
            with pytest.raises(ValueError):
                fit(10, method='advi->fullrank_advi', frac=1)
            fit(10, method='advi->fullrank_advi', frac=.5)

    def test_approximate(self):
        with models.multidimensional_model()[1]:
            fit(10, method='fullrank_advi')


class TestHistogram(SeededTest):
    def test_sampling(self):
        with models.multidimensional_model()[1]:
            full_rank = FullRankADVI()
            approx = full_rank.fit(20)
            trace0 = approx.sample_vp(10000)
            histogram = Histogram(trace0)
        trace1 = histogram.sample_vp(100000)
        np.testing.assert_allclose(trace0['x'].mean(0), trace1['x'].mean(0), atol=0.01)
        np.testing.assert_allclose(trace0['x'].var(0), trace1['x'].var(0), atol=0.01)

    def test_aevb_histogram(self):
        _, model, _ = models.exponential_beta(n=2)
        x = model.x
        mu = theano.shared(x.init_value)
        rho = theano.shared(np.zeros_like(x.init_value))
        with model:
            inference = ADVI(local_rv={x: (mu, rho)})
            approx = inference.approx
            trace0 = approx.sample_vp(10000)
            histogram = Histogram(trace0, local_rv={x: (mu, rho)})
            trace1 = histogram.sample_vp(10000)
            histogram.random(no_rand=True)
            histogram.random_fn(no_rand=True)
        np.testing.assert_allclose(trace0['y'].mean(0), trace1['y'].mean(0), atol=0.02)
        np.testing.assert_allclose(trace0['y'].var(0), trace1['y'].var(0), atol=0.02)
        np.testing.assert_allclose(trace0['x'].mean(0), trace1['x'].mean(0), atol=0.02)
        np.testing.assert_allclose(trace0['x'].var(0), trace1['x'].var(0), atol=0.02)

    def test_random_with_transformed(self):
        p = .2
        trials = (np.random.uniform(size=10) < p).astype('int8')
        with pm.Model():
            p = pm.Uniform('p')
            pm.Bernoulli('trials', p, observed=trials)
            trace = pm.sample(1000, step=pm.Metropolis())
            histogram = Histogram(trace)
            histogram.randidx(None).eval()
            histogram.randidx(1).eval()
            histogram.random_fn(no_rand=True)
            histogram.random_fn(no_rand=False)
            histogram.histogram_logp.eval()


if __name__ == '__main__':
    object.main()
