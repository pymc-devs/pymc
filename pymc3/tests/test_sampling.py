import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest

import pymc3
from pymc3 import sampling
from pymc3.sampling import sample
from .models import simple_init

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = False
except:
    test_parallel = False

RSEED = 20090425


def test_sample():

    model, start, step, _ = simple_init()

    test_njobs = [1]

    if test_parallel:
        test_samplers.append(psample)

    with model:
        for njobs in test_njobs:
            for n in [1, 10, 300]:
                yield sample, n, step, {}, None, njobs


def test_iter_sample():
    model, start, step, _ = simple_init()
    samps = sampling.iter_sample(5, step, start, model=model)
    for i, trace in enumerate(samps):
        assert i == len(trace) - 1, "Trace does not have correct length."


def test_parallel_start():
    model, _, _, _ = simple_init()
    with model:
        tr = sample(5, njobs=2, start=[{'x': [10, 10]}, {
                    'x': [-10, -10]}], random_seed=RSEED)
    assert tr.get_values('x', chains=0)[0][0] > 0
    assert tr.get_values('x', chains=1)[0][0] < 0


def test_soft_update_all_present():
    start = {'a': 1, 'b': 2}
    test_point = {'a': 3, 'b': 4}
    sampling._soft_update(start, test_point)
    assert start == {'a': 1, 'b': 2}


def test_soft_update_one_missing():
    start = {'a': 1, }
    test_point = {'a': 3, 'b': 4}
    sampling._soft_update(start, test_point)
    assert start == {'a': 1, 'b': 4}


def test_soft_update_empty():
    start = {}
    test_point = {'a': 3, 'b': 4}
    sampling._soft_update(start, test_point)
    assert start == test_point


class TestNamedSampling(unittest.TestCase):

    # TODO: Set a seed for these!

    def test_shared_named(self):
        from theano import shared
        import theano.tensor as tt

        G_var = shared(value=np.atleast_2d(1.), broadcastable=(True, False),
                       name="G")

        with pymc3.Model():
            theta0 = pymc3.Normal('theta0', mu=np.atleast_2d(0),
                                  tau=np.atleast_2d(1e20), shape=(1, 1),
                                  testval=np.atleast_2d(0))
            theta = pymc3.Normal('theta', mu=tt.dot(G_var, theta0),
                                 tau=np.atleast_2d(1e20), shape=(1, 1))

            res = theta.random()
            assert np.isclose(res, 0.)

    def test_shared_unnamed(self):
        from theano import shared
        import theano.tensor as tt
        G_var = shared(value=np.atleast_2d(1.), broadcastable=(True, False))
        with pymc3.Model():
            theta0 = pymc3.Normal('theta0', mu=np.atleast_2d(0),
                                  tau=np.atleast_2d(1e20), shape=(1, 1),
                                  testval=np.atleast_2d(0))
            theta = pymc3.Normal('theta', mu=tt.dot(G_var, theta0),
                                 tau=np.atleast_2d(1e20), shape=(1, 1))

            res = theta.random()
            assert np.isclose(res, 0.)

    def test_constant_named(self):
        import theano.tensor as tt

        G_var = tt.constant(np.atleast_2d(1.), name="G")
        with pymc3.Model():
            theta0 = pymc3.Normal('theta0', mu=np.atleast_2d(0),
                                  tau=np.atleast_2d(1e20), shape=(1, 1),
                                  testval=np.atleast_2d(0))
            theta = pymc3.Normal('theta', mu=tt.dot(G_var, theta0),
                                 tau=np.atleast_2d(1e20), shape=(1, 1))

            res = theta.random()
            assert np.isclose(res, 0.)


class TestChooseBackend(unittest.TestCase):

    def test_choose_backend_none(self):
        with mock.patch('pymc3.sampling.NDArray') as nd:
            sampling._choose_backend(None, 'chain')
        self.assertTrue(nd.called)

    def test_choose_backend_list_of_variables(self):
        with mock.patch('pymc3.sampling.NDArray') as nd:
            sampling._choose_backend(['var1', 'var2'], 'chain')
        nd.assert_called_with(vars=['var1', 'var2'])

    def test_choose_backend_invalid(self):
        self.assertRaises(ValueError,
                          sampling._choose_backend,
                          'invalid', 'chain')

    def test_choose_backend_shortcut(self):
        backend = mock.Mock()
        shortcuts = {'test_backend': {'backend': backend,
                                      'name': None}}
        sampling._choose_backend('test_backend', 'chain', shortcuts=shortcuts)
        self.assertTrue(backend.called)
