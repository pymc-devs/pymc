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
from pymc3 import *
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
    step1 = Metropolis([m1, m2], blocked=True)

    step2 = Metropolis([s], proposal_dist=LaplaceProposal)


def run(n=5000):
    if n == "short":
        n = 300
    with model:
        start = find_MAP()
        trace = sample(n, [step1, step2], start)

        dh = fn(hessian_diag(model.logpt))

    # <codecell>

    traceplot(trace)

    # <codecell>

    hexbin(trace[m1], trace[m2], gridsize=50)

# <codecell>
if __name__ == '__main__':
    run()
