"""
A model for the disasters data with no changepoint:

global_rate ~ Exp(3.)
disasters[t] ~ Po(global_rate)
"""

from pymc import *
from numpy import array

__all__ = ['global_rate', 'disasters', 'disasters_array']
disasters_array = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                         2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                         3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# Define the data and stochastics
global_rate = Exponential('global_rate', beta=1. / 3)


@stochastic(observed=True, dtype=int)
def disasters(value=disasters_array, rate=global_rate):
    """Annual occurences of coal mining disasters."""
    return poisson_like(value, rate)
