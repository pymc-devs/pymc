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


def _test_aevb(self):
    # add to inference that supports aevb
    with pm.Model() as model:
        x = pm.Normal('x')
        pm.Normal('y', x)
    x = model.x
    y = model.y
    mu = theano.shared(x.init_value) * 2
    rho = theano.shared(np.zeros_like(x.init_value))
    with model:
        inference = self.inference(local_rv={x: (mu, rho)})
        approx = inference.fit(3, obj_n_mc=2, obj_optimizer=self.optimizer)
        approx.sample(10)
        approx.apply_replacements(
            y,
            more_replacements={x: np.asarray([1, 1], dtype=x.dtype)}
        ).eval()


class TestApproximates:
    @pytest.mark.usefixtures('strict_float32')
    class Base(SeededTest):
        inference = None
        NITER = 12000
        optimizer = pm.adagrad_window(learning_rate=0.01, n_win=50)
        conv_cb = property(lambda self: [
            pm.callbacks.CheckParametersConvergence(
                every=500,
                diff='relative', tolerance=0.001),
            pm.callbacks.CheckParametersConvergence(
                every=500,
                diff='absolute', tolerance=0.0001)
        ])

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

        def test_sample(self):
            n_samples = 100
            xs = np.random.binomial(n=1, p=0.2, size=n_samples)
            with pm.Model():
                p = pm.Beta('p', alpha=1, beta=1)
                pm.Binomial('xs', n=1, p=p, observed=xs)
                app = self.inference().approx
                trace = app.sample(draws=1, include_transformed=False)
                assert trace.varnames == ['p']
                assert len(trace) == 1
                trace = app.sample(draws=10, include_transformed=True)
                assert sorted(trace.varnames) == ['p', 'p_logodds__']
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
                inf = self.inference(start={})
                approx = inf.fit(self.NITER,
                                 obj_optimizer=self.optimizer,
                                 callbacks=self.conv_cb,)
                trace = approx.sample(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.05)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.1)

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
                approx = inf.fit(self.NITER * 3, obj_optimizer=self.optimizer,
                                 callbacks=self.conv_cb)
                trace = approx.sample(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.05)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.1)

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
                    yield pm.floatX(data[:100])

            minibatches = create_minibatch(data)
            with Model():
                data_t = theano.shared(next(minibatches))

                def cb(*_):
                    data_t.set_value(next(minibatches))
                mu_ = Normal('mu', mu=mu0, sd=sd0, testval=0)
                Normal('x', mu=mu_, sd=sd, observed=data_t, total_size=n)
                inf = self.inference(scale_cost_to_minibatch=True)
                approx = inf.fit(
                    self.NITER * 3, callbacks=[cb] + self.conv_cb, obj_optimizer=self.optimizer)
                trace = approx.sample(10000)
            np.testing.assert_allclose(np.mean(trace['mu']), mu_post, rtol=0.05)
            np.testing.assert_allclose(np.std(trace['mu']), np.sqrt(1. / d), rtol=0.1)

        def test_n_obj_mc(self):
            n_samples = 100
            xs = np.random.binomial(n=1, p=0.2, size=n_samples)
            with pm.Model():
                p = pm.Beta('p', alpha=1, beta=1)
                pm.Binomial('xs', n=1, p=p, observed=xs)
                inf = self.inference(scale_cost_to_minibatch=True)
                # should just work
                inf.fit(10, obj_n_mc=10, obj_optimizer=self.optimizer)

        def test_pickling(self):
            with models.multidimensional_model()[1]:
                inference = self.inference()

            inference = pickle.loads(pickle.dumps(inference))
            inference.fit(20)

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
    test_aevb = _test_aevb

    def test_length_of_hist(self):
        with models.multidimensional_model()[1]:
            inf = self.inference()
            assert len(inf.hist) == 0
            inf.fit(10)
            assert len(inf.hist) == 10
            assert not np.isnan(inf.hist).any()
            inf.fit(self.NITER, obj_optimizer=self.optimizer)
            assert len(inf.hist) == self.NITER + 10
            assert not np.isnan(inf.hist).any()


class TestFullRank(TestApproximates.Base):
    inference = FullRankADVI
    test_aevb = _test_aevb

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


class TestSVGD(TestApproximates.Base):
    inference = functools.partial(SVGD, n_particles=100)


class TestASVGD(TestApproximates.Base):
    NITER = 5000
    inference = functools.partial(ASVGD, temperature=1.5)
    test_aevb = _test_aevb


class TestEmpirical(SeededTest):
    def test_sampling(self):
        with models.multidimensional_model()[1]:
            full_rank = FullRankADVI()
            approx = full_rank.fit(20)
            trace0 = approx.sample(10000)
            approx = Empirical(trace0)
        trace1 = approx.sample(100000)
        np.testing.assert_allclose(trace0['x'].mean(0), trace1['x'].mean(0), atol=0.01)
        np.testing.assert_allclose(trace0['x'].var(0), trace1['x'].var(0), atol=0.01)

    def test_aevb_empirical(self):
        _, model, _ = models.exponential_beta(n=2)
        x = model.x
        mu = theano.shared(x.init_value)
        rho = theano.shared(np.zeros_like(x.init_value))
        with model:
            inference = ADVI(local_rv={x: (mu, rho)})
            approx = inference.approx
            trace0 = approx.sample(10000)
            approx = Empirical(trace0, local_rv={x: (mu, rho)})
            trace1 = approx.sample(10000)
            approx.random(no_rand=True)
            approx.random_fn(no_rand=True)
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
            approx = Empirical(trace)
            approx.randidx(None).eval()
            approx.randidx(1).eval()
            approx.random_fn(no_rand=True)
            approx.random_fn(no_rand=False)
            approx.histogram_logp.eval()

    def test_init_from_noize(self):
        with models.multidimensional_model()[1]:
            approx = Empirical.from_noise(100)
            assert approx.histogram.eval().shape == (100, 6)

_model = models.simple_model()[1]
with _model:
    pm.Potential('pot', tt.ones((10, 10)))
    _advi = ADVI()
    _fullrank_advi = FullRankADVI()
    _svgd = SVGD()


@pytest.mark.parametrize(
    ['method', 'kwargs', 'error'],
    [
        ('undefined', dict(), KeyError),
        (1, dict(), TypeError),
        (_advi, dict(start={}), None),
        (_fullrank_advi, dict(), None),
        (_svgd, dict(), None),
        ('advi', dict(total_grad_norm_constraint=10), None),
        ('advi->fullrank_advi', dict(frac=.1), None),
        ('advi->fullrank_advi', dict(frac=1), ValueError),
        ('fullrank_advi', dict(), None),
        ('svgd', dict(total_grad_norm_constraint=10), None),
        ('svgd', dict(start={}), None),
        ('asvgd', dict(start={}, total_grad_norm_constraint=10), None),
        ('svgd', dict(local_rv={_model.free_RVs[0]: (0, 1)}), ValueError)
    ]
)
def test_fit(method, kwargs, error):
    with _model:
        if error is not None:
            with pytest.raises(error):
                fit(10, method=method, **kwargs)
        else:
            fit(10, method=method, **kwargs)


@pytest.mark.parametrize(
    'diff',
    [
        'relative',
        'absolute'
    ]
)
@pytest.mark.parametrize(
    'ord',
    [1, 2, np.inf]
)
def test_callbacks_convergence(diff, ord):
    cb = pm.variational.callbacks.CheckParametersConvergence(every=1, diff=diff, ord=ord)

    class _approx:
        params = (theano.shared(np.asarray([1, 2, 3])), )

    approx = _approx()

    with pytest.raises(StopIteration):
        cb(approx, None, 1)
        cb(approx, None, 10)


def test_tracker_callback():
    import time
    tracker = pm.callbacks.Tracker(
        ints=lambda *t: t[-1],
        ints2=lambda ap, h, j: j,
        time=time.time,
    )
    for i in range(10):
        tracker(None, None, i)
    assert 'time' in tracker.hist
    assert 'ints' in tracker.hist
    assert 'ints2' in tracker.hist
    assert (len(tracker['ints'])
            == len(tracker['ints2'])
            == len(tracker['time'])
            == 10)
    assert tracker['ints'] == tracker['ints2'] == list(range(10))
    tracker = pm.callbacks.Tracker(
        bad=lambda t: t  # bad signature
    )
    with pytest.raises(TypeError):
        tracker(None, None, 1)
