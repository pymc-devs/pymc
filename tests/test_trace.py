from checks import *
from models import simple_init

def check_trace(h, n, step, start):

    #try using a history object a few times
    for _ in range(2):
        h, _, _ = psample(n, step, start, h)

        for (var, val) in start.iteritems(): 

            assert np.shape(h[v]) == (n*2,) + np.shape(val)


def test_traces():
    step, start = simple_init()

    for     h in [pm.NpHistory()]:
        for n in [0, 20, 1000]: 

            yield check_trace(h, n, step, start)


