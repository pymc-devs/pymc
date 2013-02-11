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
from pymc import *
from numpy import random, array, arange, ones 
import theano.tensor as t
# Sample size
n = 100000
# True mean count, given occupancy
theta = 2.1
# True occupancy
pi = 0.4

# Simulate some data data
y = array([(random.random()<pi) * random.poisson(theta) for i in range(n)])

point = {'p' : np.array([.5]),
         'z' : ((y > 0)*1).astype('int8'),
         'theta' : np.array([10.])}
model = Model(test_point = point)
Var = model.Var
Data = model.Data 

# Estimated occupancy

p = Var('p', Beta(1,1))

# Latent variable for occupancy
z = Var('z', Bernoulli(p) , y.shape, dtype = 'int8')

# Estimated mean count
theta = Var('theta', Uniform(0, 100))

# Poisson likelihood



Data(y, ZeroInflatedPoisson(theta, z))



_pymc3_logp = model.logp()
def pymc3_logp():
    _pymc3_logp(point) 

_pymc3_dlogp = model.dlogp()
def pymc3_dlogp():
    _pymc3_dlogp(point) 
