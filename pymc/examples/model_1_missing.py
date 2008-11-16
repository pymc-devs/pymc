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

__all__ = ['switchpoint', 'early_mean', 'late_mean', 'disasters']

# Missing values indicated by None
disasters_data =  (4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, None, 3, 4, 2, 5,
    2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, 1, 2, 1, None, 1, 1, 2, 4, 2, 0, 0, 1, 4,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1)


# Define data and stochastics
switchpoint = DiscreteUniform('switchpoint',lower=0,upper=110)
early_mean = Exponential('early_mean',beta=1.)
late_mean = Exponential('late_mean',beta=1.)

# Initialise array of stochastics
disasters = []

for i,d in enumerate(disasters_data):
    
    # Observed when d is not None
    observed = d is not None
    # If not observed, sample randomly for initial value
    value = (randint(0, 6), d)[observed]
    
    # d is observed if its value is not None
    @stochastic(observed=observed, dtype=int)
    def disaster(  value = value, 
            early_mean = early_mean, 
            late_mean = late_mean, 
            switchpoint = switchpoint):
        """Annual occurences of coal mining disasters."""

        # Pick mean value according to switchpoint
        meanval = (early_mean, late_mean)[i>switchpoint]
        
        return poisson_like(value, meanval)
        
    disasters.append(disaster)

M = MCMC([disasters, switchpoint, early_mean, late_mean])