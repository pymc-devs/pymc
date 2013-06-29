# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Sometimes, it is very useful to update a set of parameters together. For example, variables that are highly correlated are often good to update together. In PyMC 3 block updating is simple, as example will demonstrate.
#
# Here we have a LASSO regression model where the two coefficients are strongly correlated. Normally, we would define the coefficient parameters as a single random variable, but here we define them separately to show how to do block updates.
#
# First we generate some fake data.

# <codecell>
from matplotlib.pylab import *
from pymc import *
import numpy as np

d = np.random.normal(size=(3, 30))
d1 = d[0] + 4
d2 = d[1] + 4
yd = .2 * d1 + .3 * d2 + d[2]

# <markdowncell>

# Then define the random variables.

# <codecell>

with Model() as model:
    s = Exponential('s', 1)
    m1 = Laplace('m1', 0, 100)
    m2 = Laplace('m2', 0, 100)

    p = d1 * m1 + d2 * m2

    y = Normal('y', p, s ** -2, observed=yd)

# <markdowncell>

# For most samplers, including Metropolis and HamiltonianMC, simply pass a
# list of variables to sample as a block. This works with both scalar and
# array parameters.

# <codecell>

with model:
    start = find_MAP()

    step1 = Metropolis([m1, m2])

    step2 = Metropolis([s], proposal_dist=LaplaceProposal)

    trace = sample(5000, [step1, step2], start)
    
from pymc.model import cont_inputs
import theano.tensor as t


def gradient1(f, v):
    """flat gradient of f wrt v"""
    return t.flatten(t.grad(f, v, disconnected_inputs='warn'))


def hessian_diag1(f, v):

    g = gradient1(f, v)
    idx = t.arange(g.shape[0])

    def hess_ii(i):
        return gradient1(g[i], v)[i]

    return theano.map(hess_ii, idx)[0]


def hessian_diag(f, vars=None):

    if not vars:
        vars = cont_inputs(f)

    return t.concatenate([hessian_diag1(f, v) for v in vars], axis=0)


dh = compilef(hessian_diag(model.logp))

# <codecell>

traceplot(trace)

# <codecell>

hexbin(trace[m1], trace[m2], gridsize=50)

# <codecell>
