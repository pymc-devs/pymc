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

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = False
except:
    test_parallel = False


@mock.patch('pymc.sampling._sample')
def test_sample_check_full_signature_single_thread(sample_func):
    sample('draws', 'step', start='start', db='db', threads=1, chain=1,
           tune='tune', progressbar='progressbar', model='model',
           variables='variables', random_seed='random_seed')
    sample_func.assert_called_with('draws', 'step', 'start', 'db', 1,
                                   'tune', 'progressbar', 'model', 'variables',
                                   'random_seed')


@mock.patch('pymc.sampling._thread_sample')
def test_sample_check_ful_signature_multithreads(sample_func):
    sample('draws', 'step', start='start', db='db', threads=2, chain=1,
           tune='tune', progressbar='progressbar', model='model',
           variables='variables', random_seed=0)

    args = sample_func.call_args_list[0][0]
    assert args[0] == 2

    expected_argset = [('draws', 'step', 'start', 'db', 1, 'tune',
                        False, 'model', 'variables', 0),
                       ('draws', 'step', 'start', 'db', 2, 'tune',
                        False, 'model', 'variables', 0)]
    argset = list(args[1])
    print(argset)
    print(expected_argset)
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

    test_threads = [1]

    if test_parallel:
        test_threads.append(2)

    with model:
        for threads in test_threads:
            for n in [1, 10, 300]:
                yield sample, n, step, {}, None, threads


def test_iter_sample():
    model, start, step, _ = simple_init()
    samps = sampling.iter_sample(5, step, start, model=model)
    for i, trace in enumerate(samps):
        assert i == len(trace) - 1, "Trace does not have correct length."
