from pymc3 import *
from numpy import ones, array, random

# Samples for each dose level
n = 5 * ones(4, dtype=int)
# Log-dose
dose = array([-.86, -.3, -.05, .73])

with Model() as model:

    # Logit-linear model parameters
    alpha = Normal('alpha', 0, tau=0.01)
    beta = Normal('beta', 0, tau=0.01)

    # Calculate probabilities of death
    theta = Deterministic('theta', invlogit(alpha + beta * dose))

    # Data likelihood
    deaths = Binomial('deaths', n=n, p=theta, observed=[0, 1, 3, 5])

def run(n=1000):
    if n == "short":
        n = 50
    with model:
        random.seed(42)
        trace = sample(n, init='random')
        summary(trace, varnames=['alpha', 'beta'])

if __name__ == '__main__':
    run()
