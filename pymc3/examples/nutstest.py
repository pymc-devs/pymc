# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from matplotlib.pylab import *
from pymc3 import *
from numpy.random import normal


xtrue = normal(scale=2., size=1)

with Model() as model:
    x = Normal('x', mu=0., tau=1)

    step = NUTS()


def run(n=5000):
    if n == "short":
        n = 50
    with model:
        trace = sample(n, step)

    # <markdowncell>

    # To use more than one sampler, use a CompoundStep which takes a list of step methods.
    #
    # The trace object can be indexed by the variables returning an array with the first index being the sample index
    # and the other indexes the shape of the parameter. Thus the shape of trace[x].shape == (ndraw, 2,1).
    #
    # Pymc3 provides `traceplot` a simple plot for a trace.

    # <codecell>

    plot(trace[x])

    # <codecell>

    trace[x]

    # <codecell>

    traceplot(trace)

if __name__ == '__main__':
    run()
