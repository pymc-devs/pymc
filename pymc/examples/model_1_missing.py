"""
A model for the disasters data with a changepoint, with missing data

changepoint ~ U(0,110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""
__all__ = ['swichpoint','early_mean','late_mean','disasters']

from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform, Lambda, MCMC, observed, poisson_like
from pymc.distributions import Impute
import numpy as np

# Missing values indicated by -999 placeholders
disasters_array =   np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# Mask for missing values
disasters_mask = disasters_array == -999

# Switchpoint
s = DiscreteUniform('s', lower=0, upper=110)
# Early mean
e = Exponential('e', beta=1)
# Late mean
l = Exponential('l', beta=1)

@deterministic(plot=False)
def r(s=s, e=e, l=l):
    """Allocate appropriate mean to time series"""
    out = np.empty(len(disasters_array))
    # Early mean prior to switchpoint
    out[:s] = e
    # Late mean following switchpoint
    out[s:] = l
    return out


# Where the mask is true, the value is taken as missing.
masked_data = np.ma.masked_array(disasters_array, disasters_mask)
D = ImputeMissing('D', Poisson, masked_data, mu=r)
