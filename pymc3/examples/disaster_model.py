"""
A model for the disasters data with a changepoint

changepoint ~ U(1851, 1962)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Poi(early_mean if t <= switchpoint, late_mean otherwise)

"""


import theano.tensor as tt

from numpy import arange, array

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
year = arange(1851, 1962)

with pm.Model() as model:

    switchpoint = pm.DiscreteUniform("switchpoint", lower=year.min(), upper=year.max())
    early_mean = pm.Exponential("early_mean", lam=1.0)
    late_mean = pm.Exponential("late_mean", lam=1.0)

    # Allocate appropriate Poisson rates to years before and after current
    # switchpoint location
    rate = tt.switch(switchpoint >= year, early_mean, late_mean)

    disasters = pm.Poisson("disasters", rate, observed=disasters_data)

    # Initial values for stochastic nodes
    start = {"early_mean": 2.0, "late_mean": 3.0}

    tr = pm.sample(1000, tune=500, start=start)
    pm.traceplot(tr)
