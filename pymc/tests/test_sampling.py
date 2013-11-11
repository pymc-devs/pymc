import pymc
from pymc import sample, psample
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

    test_samplers = [sample]

    tr = sample(5, step, start, model=model)
    test_traces = [None, tr]

    if test_parallel:
        test_samplers.append(psample)

    with model:
        for trace in test_traces:
            for samplr in test_samplers:
                for n in [0, 1, 10, 300]:

                    yield samplr, n, step, {}
                    yield samplr, n, step, {}, trace
                    yield samplr, n, step, start
