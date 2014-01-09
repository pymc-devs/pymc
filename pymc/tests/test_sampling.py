import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest

from pymc import sample, iter_sample
from .models import simple_init

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = False
except:
    test_parallel = False


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
    samps = iter_sample(5, step, start, model=model)
    for i, trace in enumerate(samps):
        assert i == len(trace) - 1, "Trace does not have correct length."
