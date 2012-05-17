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
from numpy import random, array, arange, ones 
from mcex import *
import theano.tensor as t
# Sample size
n = 100
# True mean count, given occupancy
theta = 2.1
# True occupancy
pi = 0.4

# Simulate some data data
y = array([(random.random()<pi) * random.poisson(theta) for i in range(n)])


model = Model()
# Estimated occupancy

p = AddVar(model, 'p', Beta(1,1))

# Latent variable for occupancy
z = AddVar(model, 'z', Bernoulli(p) , y.shape, dtype = 'int8')

# Estimated mean count
theta = AddVar( model, 'theta', Uniform(0, 100))

# Poisson likelihood



AddData(model, y, ZeroInflatedPoisson(theta, z))

chain = {'p' : .5,
         'z' : (y > 0)*1,
         'theta' : 10.}

chain, r = find_MAP(model, chain, vars = [p, theta], retall = True)
C = approx_cov(model, chain, [p, theta])

zs = t.constant(arange(0,1+1)[ :, None] * ones(y.shape)[None, :], 'zs', dtype = 'int8')

"""
gibbs sampling for discrete variables is currently done with categorical_gibbs. 
This won't work for discrete but unbounded variables, like poisson.
 it would be really excellent to encapsulate this behavior somehow
 one idea is to have a separate class like "Model" (plus functions) which stores
 these exhaustive conditional probability calculations.
 
 #this will only work for elemwise distributions
def AddDiscrete(model, name, distribution, shape, values):
    var = FreeVar(name)
    
    model.priors[name] = distribution()

#this doesn't work 
    

"""

step_method = compound_step([hmc_step(model, [p, theta], C, step_size_scaling = .25, trajectory_length = 2),
                             elemwise_cat_gibbs_step(model, z,  [0,1])])

ndraw = 3e3

history = NpHistory(model.vars, ndraw) # an object that keeps track
state, t = sample(ndraw, step_method, chain, history)

print "took :", t  