import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest

import pymc
from pymc import sampling
from pymc.sampling import sample
from .models import simple_init

## Set to False to keep effect of cea5659. Should this be set to True?
TEST_PARALLEL = False


@mock.patch('pymc.sampling._sample')
def test_sample_check_full_signature_single_process(sample_func):
    sample('draws', 'step', start='start', trace='trace', njobs=1, chain=1,
           tune='tune', progressbar='progressbar', model='model',
           random_seed='random_seed')
    sample_func.assert_called_with('draws', 'step', 'start', 'trace', 1,
                                   'tune', 'progressbar', 'model',
                                   'random_seed')


@mock.patch('pymc.sampling._mp_sample')
def test_sample_check_full_signature_mp(sample_func):
    sample('draws', 'step', start='start', trace='trace', njobs=2, chain=1,
           tune='tune', progressbar='progressbar', model='model',
           random_seed=0)

    args = sample_func.call_args_list[0][0]
    assert args[0] == 2

    expected_argset = [('draws', 'step', 'start', 'trace', 1, 'tune',
                        'progressbar', 'model', 0),
                       ('draws', 'step', 'start', 'trace', 2, 'tune',
                        False, 'model', 0)]
    argset = list(args[1])
    assert argset == expected_argset


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


def test_sample():

    model, start, step, _ = simple_init()

    test_njobs = [1]

    if TEST_PARALLEL:
        test_njobs.append(2)

    with model:
        for njobs in test_njobs:
            for n in [1, 10, 300]:
                yield sample, n, step, {}, None, njobs


def test_iter_sample():
    model, start, step, _ = simple_init()
    samps = sampling.iter_sample(5, step, start, model=model)
    for i, trace in enumerate(samps):
        assert i == len(trace) - 1, "Trace does not have correct length."


class TestChooseBackend(unittest.TestCase):

    def test_choose_backend_none(self):
        with mock.patch('pymc.sampling.NDArray') as nd:
            sampling._choose_backend(None, 'chain')
        self.assertTrue(nd.called)

    def test_choose_backend_list_of_variables(self):
        with mock.patch('pymc.sampling.NDArray') as nd:
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
