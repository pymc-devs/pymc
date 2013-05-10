"""
A model for the disasters data with a changepoint

changepoint ~ U(0, 110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""

from pymc import *

import theano.tensor as t
from numpy import array, ones, concatenate
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
    years = 110
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=years)
    early_mean = Exponential('early_mean', lam=1.)
    late_mean = Exponential('late_mean', lam=1.)

    @tensordist(discrete)
    def PoissonMixture(early, late, s):
        pois_early = Poisson(early)
        pois_late = Poisson(late)

        def logp(value):
            return switch(value <= s,
                          pois_early.logp(value),
                          pois_late.logp(value))

        mode = floor((pois_early.mode*s + pois_late.mode*(years-s))/years)

        return locals()

    disasters = PoissonMixture('disasters', early_mean, late_mean, switchpoint, observed=disasters_data)

    start = {'early_mean':2., 'late_mean':3., 'switchpoint':50}

    step1 = Metropolis([switchpoint, early_mean, late_mean])

    trace = sample(3000, step1, start)

    print trace['early_mean'].mean()