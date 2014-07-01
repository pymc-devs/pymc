#!/usr/bin/env python
"""
zip.py

Zero-inflated Poisson example using simulated data.
"""
import numpy as np
from pymc import Uniform, Beta, observed, rpoisson, poisson_like

# True parameter values
mu_true = 5
psi_true = 0.75
n = 100

# Simulate some data
data = np.array([rpoisson(mu_true) * (np.random.random() < psi_true)
                 for i in range(n)])

# Uniorm prior on Poisson mean
mu = Uniform('mu', 0, 20)
# Beta prior on psi
psi = Beta('psi', alpha=1, beta=1)


@observed(dtype=int, plot=False)
def zip(value=data, mu=mu, psi=psi):
    """ Zero-inflated Poisson likelihood """

    # Initialize likeihood
    like = 0.0

    # Loop over data
    for x in value:

        if not x:
            # Zero values
            like += np.log((1. - psi) + psi * np.exp(-mu))

        else:
            # Non-zero values
            like += np.log(psi) + poisson_like(x, mu)

    return like

if __name__ == "__main__":

    from pymc import MCMC, Matplot

    # Run model and plot posteriors
    M = MCMC(locals())
    M.sample(100000, 50000)
    Matplot.plot(M)
