"""
A model for the disasters data with a changepoint, with GOF diagnostics added

changepoint ~ U(0,110)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)

"""

import pymc as pm
from numpy import array, concatenate, ones
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

switchpoint = pm.DiscreteUniform('switchpoint',lower=0,upper=110)
early_mean = pm.Exponential('early_mean',beta=1.)
late_mean = pm.Exponential('late_mean',beta=1.)

@pm.stochastic(observed=True, dtype=int)
def disasters(  value = disasters_array,
                early_mean = early_mean,
                late_mean = late_mean,
                switchpoint = switchpoint):
    """Annual occurences of coal mining disasters."""
    return pm.poisson_like(value[:switchpoint],early_mean) + pm.poisson_like(value[switchpoint:],late_mean)

@pm.deterministic
def disasters_sim(early_mean = early_mean,
                late_mean = late_mean,
                switchpoint = switchpoint):
    """Coal mining disasters sampled from the posterior predictive distribution"""
    return concatenate( (pm.rpoisson(early_mean, size=switchpoint), pm.rpoisson(late_mean, size=n-switchpoint)))

@pm.deterministic
def expected_values(early_mean = early_mean,
                late_mean = late_mean,
                switchpoint = switchpoint):
    """Discrepancy measure for GOF using the Freeman-Tukey statistic"""

    # Sample size
    n = len(disasters_array)
    # Expected values
    return concatenate((ones(switchpoint)*early_mean, ones(n-switchpoint)*late_mean))
    

if __name__ == '__main__':
    vars = [switchpoint, early_mean, late_mean, disasters, disasters_sim, expected_values]
    # Instiatiate model
    M = pm.MCMC(vars)
    # Sample
    M.sample(10000, burn=5000, verbose=2)
    # Calculate discrepancy function
    D = pm.diagnostics.discrepancy(disasters_array, disasters_sim, expected_values)
    # Plot GOF graphics
    pm.Matplot.discrepancy_plot(D, 'D')
    pm.Matplot.gof_plot(disasters_sim, disasters_array, 'disasters')
