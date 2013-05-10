"""
A model for the disasters data with a changepoint

changepoint ~ U(0, 110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""

from pymc import *

import theano.tensor as t
from numpy import arange, array, ones, concatenate
from numpy.random import randint

__all__ = ['disasters_data', 'switchpoint', 'early_mean', 'late_mean', 'rate', 'disasters']

disasters_data =   array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

with Model() as model:

    # Define data and stochastics
    years = 111
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=years)
    early_mean = Exponential('early_mean', lam=1.)
    late_mean = Exponential('late_mean', lam=1.)

    idx = arange(years)
    rate = switch(switchpoint >= idx, early_mean, late_mean)

    disasters = Poisson('disasters', rate, observed=disasters_data)

    start = {'early_mean':2., 'late_mean':3., 'switchpoint':50}

    step1 = Slice([early_mean, late_mean])
    step2 = Metropolis([switchpoint])

    trace = sample(10000, [step1,step2], start)

    traceplot(trace)
