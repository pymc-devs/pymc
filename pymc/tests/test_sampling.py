from pymc import sample, psample
from models import simple_init

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = True
except:
    test_parallel = False

def test_sample():
    model, start, step,_ = simple_init()

    test_samplers = [sample]
    if test_parallel:
        test_samplers.append(psample)

    with model:
        for     samplr  in test_samplers: 
            for n       in [0, 10, 1000]:
                yield samplr, n, step, start 



