from checks import *
from models import simple_init

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = True
except:
    test_parallel = False

def check_trace(trace, n, step, start):

    #try using a trace object a few times
    for i in range(2):
        trace, _, _ = sample(n, step, start, trace)

        for (var, val) in start.iteritems():

            assert np.shape(trace[var]) == (n*(i+1),) + np.shape(val)


def test_trace():
    start, step,_  = simple_init()

    for     h in [pm.NpTrace]:
        for n in [20, 1000]:
            trace = h()

            yield check_trace, trace, n, step, start

def test_multitrace():
    if not test_parallel:
        return
    start, step,_  = simple_init()
    trace = None
    for n in [20, 1000]:

        yield check_multi_trace, trace, n, step, start



def check_multi_trace(trace, n, step, start):

    #try using a trace object a few times
    for i in range(2):
        trace, _, _ = psample(n, step, start, trace)

        for (var, val) in start.iteritems():
            for t in trace[var]:
                assert np.shape(t) == (n*(i+1),) + np.shape(val)

        ctrace = trace.combined()
        for (var, val) in start.iteritems():

            assert np.shape(ctrace[var]) == (len(trace.traces)*n*(i+1),) + np.shape(val)


def test_get_point():
    from pymc import *
    x = NpTrace(10)
    p = {'a' : np.ones(5), 'm' : np.zeros((2,2))}
    x += p
    x += p
    x.point(0)



