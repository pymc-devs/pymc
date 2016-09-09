"""
From the PyMC example list
latent_occupancy.py

Simple model demonstrating the estimation of occupancy, using latent variables. Suppose
a population of n sites, with some proportion pi being occupied. Each site is surveyed,
yielding an array of counts, y:

y = [3, 0, 0, 2, 1, 0, 1, 0, ..., ]

This is a classic zero-inflated count problem, where more zeros appear in the data than would
be predicted by a simple Poisson model. We have, in fact, a mixture of models; one, conditional
on occupancy, with a poisson mean of theta, and another, conditional on absence, with mean zero.
One way to tackle the problem is to model the latent state of 'occupancy' as a Bernoulli variable
at each site, with some unknown probability:

z_i ~ Bern(pi)

These latent variables can then be used to generate an array of Poisson parameters:

t_i = theta (if z_i=1) or 0 (if z_i=0)

Hence, the likelihood is just:

y_i = Poisson(t_i)

(Note in this elementary model, we are ignoring the issue of imperfect detection.)

Created by Chris Fonnesbeck on 2008-07-28.
Copyright (c) 2008 University of Otago. All rights reserved.
"""

# Import statements
from pymc3 import *
from numpy import random, array, arange, ones
# Sample size
n = 100
# True mean count, given occupancy
theta = 2.1
# True occupancy
pi = 0.4

# Simulate some data data
y = array([(random.random() < pi) * random.poisson(theta) for i in range(n)])

model = Model()
with model:
    # Estimated occupancy

    psi = Beta('psi', 1, 1)

    # Estimated mean count
    theta = Uniform('theta', 0, 100)

    # Poisson likelihood
    yd = ZeroInflatedPoisson('y', theta, psi, observed=y)


point = model.test_point

_pymc33_logp = model.logp


def pymc33_logp():
    _pymc33_logp(point)

_pymc33_dlogp = model.dlogp()


def pymc33_dlogp():
    _pymc33_dlogp(point)


def run(n=5000):
    if n == "short":
        n = 50
    with model:
        start = {'psi': 0.5, 'z': (y > 0).astype(int), 'theta': 5}

        step1 = Metropolis([theta, psi])

        trace = sample(n, [step1, step2], start)

if __name__ == '__main__':
    run()
