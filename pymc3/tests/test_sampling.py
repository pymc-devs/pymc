from itertools import combinations
import numpy as np

try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock

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
        super(TestSample, self).setup_method()
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
            with pytest.raises(TypeError) as excinfo:
                pm.sample(50, tune=0, init=None, step_kwargs={'nuts': {'foo': 1}})
            assert "'foo'" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                pm.sample(50, tune=0, init=None, step_kwargs={'foo': {}})
            assert 'foo' in str(excinfo.value)

            pm.sample(10, tune=0, init=None, nuts_kwargs={'target_accept': 0.9})

            with pytest.raises(ValueError) as excinfo:
                pm.sample(5, tune=0, init=None, step_kwargs={}, nuts_kwargs={})
            assert 'Specify only one' in str(excinfo.value)

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


class TestChooseBackend(object):
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
            a = pm.Normal('a', mu=0, sd=1)
            trace = pm.sample()

        with model:
            # test list input
            ppc0 = pm.sample_ppc([model.test_point], samples=10)
            ppc = pm.sample_ppc(trace, samples=1000, vars=[])
            assert len(ppc) == 0
            ppc = pm.sample_ppc(trace, samples=1000, vars=[a])
            assert 'a' in ppc
            assert ppc['a'].shape == (1000,)
        _, pval = stats.kstest(ppc['a'], stats.norm().cdf)
        assert pval > 0.001

        with model:
            ppc = pm.sample_ppc(trace, samples=10, size=5, vars=[a])
            assert ppc['a'].shape == (10, 5)

    def test_normal_vector(self):
        with pm.Model() as model:
            a = pm.Normal('a', mu=0, sd=1, shape=2)
            trace = pm.sample()

        with model:
	    # test list input
            ppc0 = pm.sample_ppc([model.test_point], samples=10)
            ppc = pm.sample_ppc(trace, samples=10, vars=[])
            assert len(ppc) == 0
            ppc = pm.sample_ppc(trace, samples=10, vars=[a])
            assert 'a' in ppc
            assert ppc['a'].shape == (10, 2)

            ppc = pm.sample_ppc(trace, samples=10, vars=[a], size=4)
            assert 'a' in ppc
            assert ppc['a'].shape == (10, 4, 2)

    def test_sum_normal(self):
        with pm.Model() as model:
            a = pm.Normal('a', sd=0.2)
            b = pm.Normal('b', mu=a)
            trace = pm.sample()

        with model:
	    # test list input
            ppc0 = pm.sample_ppc([model.test_point], samples=10)
            ppc = pm.sample_ppc(trace, samples=1000, vars=[b])
            assert len(ppc) == 1
            assert ppc['b'].shape == (1000,)
            scale = np.sqrt(1 + 0.2 ** 2)
            _, pval = stats.kstest(ppc['b'], stats.norm(scale=scale).cdf)
            assert pval > 0.001

class TestSamplePPCW(SeededTest):
    def test_sample_ppc_w(self):
        data0 = np.random.normal(0, 1, size=500)

        with pm.Model() as model_0:
            mu = pm.Normal('mu', mu=0, sd=1)
            y = pm.Normal('y', mu=mu, sd=1, observed=data0, shape=500)
            trace_0 = pm.sample()

        with pm.Model() as model_1:
            mu = pm.Normal('mu', mu=0, sd=1, shape=len(data0))
            y = pm.Normal('y', mu=mu, sd=1, observed=data0, shape=500)
            trace_1 = pm.sample()

        traces = [trace_0, trace_0]
        models = [model_0, model_0]
        ppc = pm.sample_ppc_w(traces, 100, models)
        assert ppc['y'].shape == (100, 500)

        traces = [trace_0, trace_1]
        models = [model_0, model_1]
        ppc = pm.sample_ppc_w(traces, 100, models)
        assert ppc['y'].shape == (100, 500)


@pytest.mark.parametrize('method', [
    'jitter+adapt_diag', 'adapt_diag', 'advi', 'ADVI+adapt_diag',
    'advi+adapt_diag_grad', 'map', 'advi_map', 'nuts'
])
def test_exec_nuts_init(method):
    with pm.Model() as model:
        pm.Normal('a', mu=0, sd=1, shape=2)
        pm.HalfNormal('b', sd=1)
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
