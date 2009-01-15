"""
A model for the disasters data with a changepoint

changepoint ~ U(0,110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""

from pymc import *
from numpy import array, ones, append
from numpy.random import randint

__all__ = ['disasters_array', 'switchpoint', 'early_mean', 'late_mean', 'disasters']

disasters_array =   array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
n = len(disasters_array)

# Define data and stochastics

switchpoint = DiscreteUniform('switchpoint',lower=0,upper=110)
means = Exponential('means',beta=ones(2))
    
@stochastic(observed=True, dtype=int)
def disasters(  value = disasters_array, 
                means = means, 
                switchpoint = switchpoint):
    """Annual occurences of coal mining disasters."""
    
    def logp(value, means, switchpoint): 
        return poisson_like(value[:switchpoint], means[0]) + poisson_like(value[switchpoint:], means[1])
        
    def random(means, switchpoint):
        return append(rpoisson(means[0], size=switchpoint), rpoisson(means[1], size=n-switchpoint))