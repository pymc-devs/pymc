from checks import *
from models import *
import pymc as pm

# Test if multiprocessing is available
import multiprocessing
try:
    multiprocessing.Pool(2)
    test_parallel = True
except:
    test_parallel = False

def check_trace(model, trace, n, step, start):
    #try using a trace object a few times
    for i in range(2):
        trace = sample(n, step, start, trace, track_progress=False, model = model)

        for (var, val) in start.iteritems():

            assert np.shape(trace[var]) == (n*(i+1),) + np.shape(val)


def test_trace():
    model, start, step,_  = simple_init()

    for h in [pm.NpTrace]:
        for n in [20, 1000]:
            for vars in [model.vars, model.vars + [model.vars[0]**2]]:
                trace = h(vars)


                yield check_trace, model, trace, n, step, start

def test_multitrace():
    if not test_parallel:
        return
    model, start, step,_  = simple_init()
    trace = None
    for n in [20, 1000]:

        yield check_multi_trace, model, trace, n, step, start



def check_multi_trace(model, trace, n, step, start):

    for i in range(2):
        trace = psample(n, step, start, trace, track_progress=False, model = model)


        for (var, val) in start.iteritems():
            print [len(tr.samples[var].vals) for tr in trace.traces]
            for t in trace[var]:
                assert np.shape(t) == (n*(i+1),) + np.shape(val)

        ctrace = trace.combined()
        for (var, val) in start.iteritems():

            assert np.shape(ctrace[var]) == (len(trace.traces)*n*(i+1),) + np.shape(val)


def test_get_point():

    p, model = simple_2model()
    p2 = p.copy()
    p2['x'] *= 2.

    x = pm.NpTrace(model.vars)
    x.record(p)
    x.record(p2)
    assert x.point(1) == x[1]



