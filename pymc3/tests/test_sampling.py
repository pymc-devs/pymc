from itertools import combinations
import numpy as np

try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock

import numpy.testing as npt
import pymc3 as pm
import theano.tensor as tt
from theano import shared
import theano
from .models import simple_init
from .helpers import SeededTest
from scipy import stats
import pytest


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestSample(SeededTest):
    def setup_method(self):
        super().setup_method()
        self.model, self.start, self.step, _ = simple_init()

    def test_sample_does_not_set_seed(self):
        random_numbers = []
        for _ in range(2):
            np.random.seed(1)
            with self.model:
                pm.sample(1, tune=0, chains=1)
                random_numbers.append(np.random.random())
        assert random_numbers[0] == random_numbers[1]

    def test_parallel_sample_does_not_reuse_seed(self):
        cores = 4
        random_numbers = []
        draws = []
        for _ in range(2):
            np.random.seed(1)  # seeds in other processes don't effect main process
            with self.model:
                trace = pm.sample(100, tune=0, cores=cores)
            # numpy thread mentioned race condition.  might as well check none are equal
            for first, second in combinations(range(cores), 2):
                first_chain = trace.get_values('x', chains=first)
                second_chain = trace.get_values('x', chains=second)
                assert not (first_chain == second_chain).all()
            draws.append(trace.get_values('x'))
            random_numbers.append(np.random.random())

        # Make sure future random processes aren't effected by this
        assert random_numbers[0] == random_numbers[1]
        assert (draws[0] == draws[1]).all()

    def test_sample(self):
        test_cores = [1]
        with self.model:
            for cores in test_cores:
                for steps in [1, 10, 300]:
                    pm.sample(steps, tune=0, step=self.step, cores=cores,
                              random_seed=self.random_seed)

    def test_sample_init(self):
        with self.model:
            for init in ('advi', 'advi_map', 'map', 'nuts'):
                pm.sample(init=init, tune=0,
                          n_init=1000, draws=50,
                          random_seed=self.random_seed)

    def test_sample_args(self):
        with self.model:
            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, init=None, foo=1)
            assert "'foo'" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, init=None, step_kwargs={'foo': {}})
            assert 'foo' in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                pm.sample(10, tune=0, init=None, target_accept=0.9)
            assert 'target_accept' in str(excinfo.value)

    def test_iter_sample(self):
        with self.model:
            samps = pm.sampling.iter_sample(draws=5, step=self.step,
                                            start=self.start, tune=0,
                                            random_seed=self.random_seed)
            for i, trace in enumerate(samps):
                assert i == len(trace) - 1, "Trace does not have correct length."

    def test_parallel_start(self):
        with self.model:
            tr = pm.sample(0, tune=5, cores=2,
                           discard_tuned_samples=False,
                           start=[{'x': [10, 10]}, {'x': [-10, -10]}],
                           random_seed=self.random_seed)
        assert tr.get_values('x', chains=0)[0][0] > 0
        assert tr.get_values('x', chains=1)[0][0] < 0

    def test_sample_tune_len(self):
        with self.model:
            trace = pm.sample(draws=100, tune=50, cores=1)
            assert len(trace) == 100
            trace = pm.sample(draws=100, tune=50, cores=1,
                              discard_tuned_samples=False)
            assert len(trace) == 150
            trace = pm.sample(draws=100, tune=50, cores=4)
            assert len(trace) == 100

    @pytest.mark.parametrize(
        'start, error', [
            ([1, 2], TypeError),
            ({'x': 1}, ValueError),
            ({'x': [1, 2, 3]}, ValueError),
            ({'x': np.array([[1, 1], [1, 1]])}, ValueError)
        ]
    )
    def test_sample_start_bad_shape(self, start, error):
        with pytest.raises(error):
            pm.sampling._check_start_shape(self.model, start)

    @pytest.mark.parametrize(
        'start', [
            {'x': np.array([1, 1])},
            {'x': [10, 10]},
            {'x': [-10, -10]},
        ]
    )
    def test_sample_start_good_shape(self, start):
        pm.sampling._check_start_shape(self.model, start)


def test_empty_model():
    with pm.Model():
        pm.Normal('a', observed=1)
        with pytest.raises(ValueError) as error:
            pm.sample()
        error.match('any free variables')


def test_partial_trace_sample():
    with pm.Model() as model:
        a = pm.Normal('a', mu=0, sigma=1)
        b = pm.Normal('b', mu=0, sigma=1)
        trace = pm.sample(trace=[a])


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestNamedSampling(SeededTest):
    def test_shared_named(self):
        G_var = shared(value=np.atleast_2d(1.), broadcastable=(True, False),
                       name="G")

        with pm.Model():
            theta0 = pm.Normal('theta0', mu=np.atleast_2d(0),
                               tau=np.atleast_2d(1e20), shape=(1, 1),
                               testval=np.atleast_2d(0))
            theta = pm.Normal('theta', mu=tt.dot(G_var, theta0),
                              tau=np.atleast_2d(1e20), shape=(1, 1))
            res = theta.random()
            assert np.isclose(res, 0.)

    def test_shared_unnamed(self):
        G_var = shared(value=np.atleast_2d(1.), broadcastable=(True, False))
        with pm.Model():
            theta0 = pm.Normal('theta0', mu=np.atleast_2d(0),
                               tau=np.atleast_2d(1e20), shape=(1, 1),
                               testval=np.atleast_2d(0))
            theta = pm.Normal('theta', mu=tt.dot(G_var, theta0),
                              tau=np.atleast_2d(1e20), shape=(1, 1))
            res = theta.random()
            assert np.isclose(res, 0.)

    def test_constant_named(self):
        G_var = tt.constant(np.atleast_2d(1.), name="G")
        with pm.Model():
            theta0 = pm.Normal('theta0', mu=np.atleast_2d(0),
                               tau=np.atleast_2d(1e20), shape=(1, 1),
                               testval=np.atleast_2d(0))
            theta = pm.Normal('theta', mu=tt.dot(G_var, theta0),
                              tau=np.atleast_2d(1e20), shape=(1, 1))

            res = theta.random()
            assert np.isclose(res, 0.)


class TestChooseBackend:
    def test_choose_backend_none(self):
        with mock.patch('pymc3.sampling.NDArray') as nd:
            pm.sampling._choose_backend(None, 'chain')
        assert nd.called

    def test_choose_backend_list_of_variables(self):
        with mock.patch('pymc3.sampling.NDArray') as nd:
            pm.sampling._choose_backend(['var1', 'var2'], 'chain')
        nd.assert_called_with(vars=['var1', 'var2'])

    def test_choose_backend_invalid(self):
        with pytest.raises(ValueError):
            pm.sampling._choose_backend('invalid', 'chain')

    def test_choose_backend_shortcut(self):
        backend = mock.Mock()
        shortcuts = {'test_backend': {'backend': backend,
                                      'name': None}}
        pm.sampling._choose_backend('test_backend', 'chain', shortcuts=shortcuts)
        assert backend.called


class TestSamplePPC(SeededTest):
    def test_normal_scalar(self):
        with pm.Model() as model:
            mu = pm.Normal('mu', 0., 1.)
            a = pm.Normal('a', mu=mu, sigma=1, observed=0.)
            trace = pm.sample()

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.sample_posterior_predictive(trace, samples=1000, vars=[])
            assert len(ppc) == 0
            ppc = pm.sample_posterior_predictive(trace, samples=1000, vars=[a])
            assert 'a' in ppc
            assert ppc['a'].shape == (1000,)
        _, pval = stats.kstest(ppc['a'],
                               stats.norm(loc=0, scale=np.sqrt(2)).cdf)
        assert pval > 0.001

        with model:
            ppc = pm.sample_posterior_predictive(trace, samples=10, size=5, vars=[a])
            assert ppc['a'].shape == (10, 5)

    def test_normal_vector(self):
        with pm.Model() as model:
            mu = pm.Normal('mu', 0., 1.)
            a = pm.Normal('a', mu=mu, sigma=1,
                          observed=np.array([.5, .2]))
            trace = pm.sample()

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.sample_posterior_predictive(trace, samples=10, vars=[])
            assert len(ppc) == 0
            ppc = pm.sample_posterior_predictive(trace, samples=10, vars=[a])
            assert 'a' in ppc
            assert ppc['a'].shape == (10, 2)

            ppc = pm.sample_posterior_predictive(trace, samples=10, vars=[a], size=4)
            assert 'a' in ppc
            assert ppc['a'].shape == (10, 4, 2)

    def test_vector_observed(self):
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=1)
            a = pm.Normal('a', mu=mu, sigma=1,
                          observed=np.array([0., 1.]))
            trace = pm.sample()

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.sample_posterior_predictive(trace, samples=10, vars=[])
            assert len(ppc) == 0
            ppc = pm.sample_posterior_predictive(trace, samples=10, vars=[a])
            assert 'a' in ppc
            assert ppc['a'].shape == (10, 2)

            ppc = pm.sample_posterior_predictive(trace, samples=10, vars=[a], size=4)
            assert 'a' in ppc
            assert ppc['a'].shape == (10, 4, 2)

    def test_sum_normal(self):
        with pm.Model() as model:
            a = pm.Normal('a', sigma=0.2)
            b = pm.Normal('b', mu=a)
            trace = pm.sample()

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive([model.test_point], samples=10)
            ppc = pm.sample_posterior_predictive(trace, samples=1000, vars=[b])
            assert len(ppc) == 1
            assert ppc['b'].shape == (1000,)
            scale = np.sqrt(1 + 0.2 ** 2)
            _, pval = stats.kstest(ppc['b'], stats.norm(scale=scale).cdf)
            assert pval > 0.001

    def test_model_not_drawable_prior(self):
        data = np.random.poisson(lam=10, size=200)
        model = pm.Model()
        with model:
            mu = pm.HalfFlat('sigma')
            pm.Poisson('foo', mu=mu, observed=data)
            trace = pm.sample(tune=1000)

        with model:
            with pytest.raises(ValueError) as excinfo:
                pm.sample_prior_predictive(50)
            assert "Cannot sample" in str(excinfo.value)
            samples = pm.sample_posterior_predictive(trace, 50)
            assert samples['foo'].shape == (50, 200)


class TestSamplePPCW(SeededTest):
    def test_sample_posterior_predictive_w(self):
        data0 = np.random.normal(0, 1, size=500)

        with pm.Model() as model_0:
            mu = pm.Normal('mu', mu=0, sigma=1)
            y = pm.Normal('y', mu=mu, sigma=1, observed=data0)
            trace_0 = pm.sample()

        with pm.Model() as model_1:
            mu = pm.Normal('mu', mu=0, sigma=1, shape=len(data0))
            y = pm.Normal('y', mu=mu, sigma=1, observed=data0)
            trace_1 = pm.sample()

        traces = [trace_0, trace_0]
        models = [model_0, model_0]
        ppc = pm.sample_posterior_predictive_w(traces, 100, models)
        assert ppc['y'].shape == (100, 500)

        traces = [trace_0, trace_1]
        models = [model_0, model_1]
        ppc = pm.sample_posterior_predictive_w(traces, 100, models)
        assert ppc['y'].shape == (100, 500)


@pytest.mark.parametrize('method', [
    'jitter+adapt_diag', 'adapt_diag', 'advi', 'ADVI+adapt_diag',
    'advi+adapt_diag_grad', 'map', 'advi_map', 'nuts'
])
def test_exec_nuts_init(method):
    with pm.Model() as model:
        pm.Normal('a', mu=0, sigma=1, shape=2)
        pm.HalfNormal('b', sigma=1)
    with model:
        start, _ = pm.init_nuts(init=method, n_init=10)
        assert isinstance(start, list)
        assert len(start) == 1
        assert isinstance(start[0], dict)
        assert 'a' in start[0] and 'b_log__' in start[0]
        start, _ = pm.init_nuts(init=method, n_init=10, chains=2)
        assert isinstance(start, list)
        assert len(start) == 2
        assert isinstance(start[0], dict)
        assert 'a' in start[0] and 'b_log__' in start[0]

class TestSamplePriorPredictive(SeededTest):
    def test_ignores_observed(self):
        observed = np.random.normal(10, 1, size=200)
        with pm.Model():
            # Use a prior that's way off to show we're ignoring the observed variables
            mu = pm.Normal('mu', mu=-100, sigma=1)
            positive_mu = pm.Deterministic('positive_mu', np.abs(mu))
            z = -1 - positive_mu
            pm.Normal('x_obs', mu=z, sigma=1, observed=observed)
            prior = pm.sample_prior_predictive()

        assert (prior['mu'] < 90).all()
        assert (prior['positive_mu'] > 90).all()
        assert (prior['x_obs'] < 90).all()
        assert prior['x_obs'].shape == (500, 200)
        npt.assert_array_almost_equal(prior['positive_mu'], np.abs(prior['mu']), decimal=4)

    def test_respects_shape(self):
        for shape in (2, (2,), (10, 2), (10, 10)):
            with pm.Model():
                mu = pm.Gamma('mu', 3, 1, shape=1)
                goals = pm.Poisson('goals', mu, shape=shape)
                trace = pm.sample_prior_predictive(10)
            if shape == 2:  # want to test shape as an int
                shape = (2,)
            assert trace['goals'].shape == (10,) + shape

    def test_multivariate(self):
        with pm.Model():
            m = pm.Multinomial('m', n=5, p=np.array([0.25, 0.25, 0.25, 0.25]), shape=4)
            trace = pm.sample_prior_predictive(10)

        assert m.random(size=10).shape == (10, 4)
        assert trace['m'].shape == (10, 4)

    def test_multivariate2(self):
        # Added test for issue #3271
        mn_data = np.random.multinomial(n=100, pvals=[1/6.]*6, size=10)
        with pm.Model() as dm_model:
            probs = pm.Dirichlet('probs', a=np.ones(6), shape=6)
            obs = pm.Multinomial('obs', n=100, p=probs, observed=mn_data)
            burned_trace = pm.sample(20, tune=10, cores=1)
        sim_priors = pm.sample_prior_predictive(samples=20,
                                                model=dm_model)
        sim_ppc = pm.sample_posterior_predictive(burned_trace,
                                                 samples=20,
                                                 model=dm_model)
        assert sim_priors['probs'].shape == (20, 6)
        assert sim_priors['obs'].shape == (20, 6)
        assert sim_ppc['obs'].shape == (20,) + obs.distribution.shape

    def test_layers(self):
        with pm.Model() as model:
            a = pm.Uniform('a', lower=0, upper=1, shape=10)
            b = pm.Binomial('b', n=1, p=a, shape=10)

        avg = b.random(size=10000).mean(axis=0)
        npt.assert_array_almost_equal(avg, 0.5 * np.ones_like(b), decimal=2)

    def test_transformed(self):
        n = 18
        at_bats = 45 * np.ones(n, dtype=int)
        hits = np.random.randint(1, 40, size=n, dtype=int)
        draws = 50

        with pm.Model() as model:
            phi = pm.Beta('phi', alpha=1., beta=1.)

            kappa_log = pm.Exponential('logkappa', lam=5.)
            kappa = pm.Deterministic('kappa', tt.exp(kappa_log))

            thetas = pm.Beta('thetas', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=n)

            y = pm.Binomial('y', n=at_bats, p=thetas, observed=hits)
            gen = pm.sample_prior_predictive(draws)

        assert gen['phi'].shape == (draws,)
        assert gen['y'].shape == (draws, n)
        assert 'thetas_logodds__' in gen

    def test_shared(self):
        n1 = 10
        obs = shared(np.random.rand(n1) < .5)
        draws = 50

        with pm.Model() as m:
            p = pm.Beta('p', 1., 1.)
            y = pm.Bernoulli('y', p, observed=obs)
            gen1 = pm.sample_prior_predictive(draws)

        assert gen1['y'].shape == (draws, n1)

        n2 = 20
        obs.set_value(np.random.rand(n2) < .5)
        with m:
            gen2 = pm.sample_prior_predictive(draws)

        assert gen2['y'].shape == (draws, n2)

    def test_density_dist(self):

        obs = np.random.normal(-1, 0.1, size=10)
        with pm.Model():
            mu = pm.Normal('mu', 0, 1)
            sd = pm.Gamma('sd', 1, 2)
            a = pm.DensityDist('a', pm.Normal.dist(mu, sd).logp, random=pm.Normal.dist(mu, sd).random, observed=obs)
            prior = pm.sample_prior_predictive()

        npt.assert_almost_equal(prior['a'].mean(), 0, decimal=1)

    def test_shape_edgecase(self):
        with pm.Model():
            mu = pm.Normal('mu', shape=5)
            sd = pm.Uniform('sd', lower=2, upper=3)
            x = pm.Normal('x', mu=mu, sigma=sd, shape=5)
            prior = pm.sample_prior_predictive(10)
        assert prior['mu'].shape == (10, 5)

    def test_zeroinflatedpoisson(self):
        with pm.Model():
            theta = pm.Beta('theta', alpha=1, beta=1)
            psi = pm.HalfNormal('psi', sd=1)
            pm.ZeroInflatedPoisson('suppliers', psi=psi, theta=theta, shape=20)
            gen_data = pm.sample_prior_predictive(samples=5000)
            assert gen_data['theta'].shape == (5000,)
            assert gen_data['psi'].shape == (5000,)
            assert gen_data['suppliers'].shape == (5000, 20)
