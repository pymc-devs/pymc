"""
A model for the disasters data with a changepoint, with missing data

changepoint ~ U(0,110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""
__all__ = ['switch', 'early_mean', 'late_mean', 'disasters']

from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform, Lambda, MCMC, observed, poisson_like
from pymc.distributions import Impute
from numpy.ma import masked_values
import numpy as np

# Missing values indicated by None placeholder values
disasters_array = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, None, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, None, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])


# Switchpoint
switch = DiscreteUniform('switch', lower=0, upper=110)
# Early mean
early_mean = Exponential('early_mean', beta=1)
# Late mean
late_mean = Exponential('late_mean', beta=1)


@deterministic(plot=False)
def rate(s=switch, e=early_mean, l=late_mean):
    """Allocate appropriate mean to time series"""
    out = np.empty(len(disasters_array))
    # Early mean prior to switchpoint
    out[:s] = e
    # Late mean following switchpoint
    out[s:] = l
    return out


# The inefficient way, using the Impute function:
# D = Impute('D', Poisson, disasters_array, mu=r)
#
# The efficient way, using masked arrays:
# Generate masked array. Where the mask is true,
# the value is taken as missing.
masked_values = masked_values(disasters_array, value=None)

# Pass masked array to data stochastic, and it does the right thing
disasters = Poisson('disasters', mu=rate, value=masked_values, observed=True)
