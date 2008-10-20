"""
A model for the disasters data with a changepoint, with missing data

changepoint ~ U(0,110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""

from pymc import *
from numpy import *
from numpy.random import randint

__all__ = ['disasters_data', 'switchpoint', 'early_mean', 'late_mean', 'disasters']

disasters_data =   MissingData('disasters_data', [ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, None, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, None, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])


# Define data and stochastics

switchpoint = DiscreteUniform('switchpoint',lower=0,upper=110)
early_mean = Exponential('early_mean',beta=1.)
late_mean = Exponential('late_mean',beta=1.)
    
@data
@stochastic(dtype=int)
def disasters(  value = disasters_data, 
                early_mean = early_mean, 
                late_mean = late_mean, 
                switchpoint = switchpoint):
    """Annual occurences of coal mining disasters."""
    
    def logp(value, early_mean, late_mean, switchpoint):
        return poisson_like(value[:switchpoint],early_mean) + poisson_like(value[switchpoint:],late_mean)

    def random(early_mean, late_mean, switchpoint):
        return concatenate((rpoisson(early_mean, switchpoint), rpoisson(late_mean, len(value)-switchpoint)))
