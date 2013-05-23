from pymc import *
from numpy import ones, array

# Samples for each dose level
n = 5 * ones(4, dtype=int)
# Log-dose
dose = array([-.86, -.3, -.05, .73])

# Logit-linear model parameters
alpha = Normal('alpha', 0, 0.01)
beta = Normal('beta', 0, 0.01)

# Calculate probabilities of death
theta = Lambda('theta', lambda a=alpha, b=beta, d=dose: invlogit(a + b * d))

# Data likelihood
deaths = Binomial(
    'deaths',
    n=n,
    p=theta,
    value=array([0,
                 1,
                 3,
                 5],
                dtype=float),
    observed=True)

# Calculate LD50
LD50 = Lambda('LD50', lambda a=alpha, b=beta: -a / b)
