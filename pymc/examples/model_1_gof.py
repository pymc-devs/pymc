"""
A model for the disasters data with a changepoint, with GOF diagnostics added

changepoint ~ U(0,110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""

from pymc import *
from numpy import *
from numpy.random import randint

__all__ = ['disasters_array', 'switchpoint', 'early_mean', 'late_mean', 'disasters']

disasters_array =   array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# Define data and stochastics

switchpoint = DiscreteUniform('switchpoint',lower=0,upper=110)
early_mean = Exponential('early_mean',beta=1.)
late_mean = Exponential('late_mean',beta=1.)
    
@data
@stochastic(dtype=int)
def disasters(  value = disasters_array, 
                early_mean = early_mean, 
                late_mean = late_mean, 
                switchpoint = switchpoint):
    """Annual occurences of coal mining disasters."""
    return poisson_like(value[:switchpoint],early_mean) + poisson_like(value[switchpoint:],late_mean)

@deterministic
def discrepancy(  early_mean = early_mean, 
                late_mean = late_mean, 
                switchpoint = switchpoint):
    """Discrepancy measure for GOF using the Freeman-Tukey statistic"""
    
    # Sample size
    n = len(disasters_array)
    # Simulated data
    sim = concatenate((rpoisson(early_mean, size=switchpoint), rpoisson(late_mean, size=n-switchpoint)))
    # Expected values
    expected = concatenate((ones(switchpoint)*early_mean, ones(n-switchpoint)*late_mean))
    # Return discrepancy measures for simulated and observed data
    return sum([(sqrt(s)-sqrt(e))**2 for s,e in zip(sim, expected)]), sum([(sqrt(x)-sqrt(e))**2 for x,e in zip(disasters_array, expected)])