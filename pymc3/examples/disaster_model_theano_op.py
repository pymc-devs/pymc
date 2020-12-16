"""
Similar to disaster_model.py, but for arbitrary
deterministics which are not not working with Theano.
Note that gradient based samplers will not work.
"""


import theano.tensor as tt

from numpy import arange, array, empty
from theano.compile.ops import as_op

import pymc3 as pm

__all__ = ["disasters_data", "switchpoint", "early_mean", "late_mean", "rate", "disasters"]

# Time series of recorded coal mining disasters in the UK from 1851 to 1962
disasters_data = array(
    [
        4,
        5,
        4,
        0,
        1,
        4,
        3,
        4,
        0,
        6,
        3,
        3,
        4,
        0,
        2,
        6,
        3,
        3,
        5,
        4,
        5,
        3,
        1,
        4,
        4,
        1,
        5,
        5,
        3,
        4,
        2,
        5,
        2,
        2,
        3,
        4,
        2,
        1,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        3,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        3,
        1,
        0,
        3,
        2,
        2,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        2,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        2,
        3,
        3,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        2,
        4,
        2,
        0,
        0,
        1,
        4,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
    ]
)
years = len(disasters_data)


@as_op(itypes=[tt.lscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector])
def rate_(switchpoint, early_mean, late_mean):
    out = empty(years)
    out[:switchpoint] = early_mean
    out[switchpoint:] = late_mean
    return out


with pm.Model() as model:

    # Prior for distribution of switchpoint location
    switchpoint = pm.DiscreteUniform("switchpoint", lower=0, upper=years)
    # Priors for pre- and post-switch mean number of disasters
    early_mean = pm.Exponential("early_mean", lam=1.0)
    late_mean = pm.Exponential("late_mean", lam=1.0)

    # Allocate appropriate Poisson rates to years before and after current
    # switchpoint location
    idx = arange(years)
    rate = rate_(switchpoint, early_mean, late_mean)

    # Data likelihood
    disasters = pm.Poisson("disasters", rate, observed=disasters_data)

    # Use slice sampler for means
    step1 = pm.Slice([early_mean, late_mean])
    # Use Metropolis for switchpoint, since it accomodates discrete variables
    step2 = pm.Metropolis([switchpoint])

    # Initial values for stochastic nodes
    start = {"early_mean": 2.0, "late_mean": 3.0}

    tr = pm.sample(1000, tune=500, start=start, step=[step1, step2], cores=2)
    pm.traceplot(tr)
