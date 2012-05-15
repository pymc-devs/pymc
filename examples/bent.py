'''
Created on May 10, 2012

@author: jsalvatier
'''
from mcex import * 
from numpy.random import normal
import numpy as np 
import pylab as pl 
from itertools import product

xa = 0.5
ya = -1.1
n = 200
data = normal( size = n) + xa + ya

model = Model()
"""
This model is U shaped because of the non identifiability. 
As n increases, the walls become steeper but the distribution does not shrink towards the mode. 
As n increases this distribution gets harder and harder for HMC to sample.

This example comes from 
Discussion of “Riemann manifold Langevin and
Hamiltonian Monte Carlo methods” by M.
Girolami and B. Calderhead

http://arxiv.org/abs/1011.0057
"""

x = AddVar(model, 'x', Normal(0, 1))
y = AddVar(model, 'y', Normal(0, 1))

AddData(model, data, Normal(x + y**2, 1.) )


chain = {'x' : 0,
         'y' : 0}


chain = find_MAP(model, chain)
hmc_cov = approx_cov(model, chain) 

step_method = hmc_step(model, model.vars, hmc_cov)


ndraw = 3e3
history = NpHistory(model.vars, ndraw)
state, t = sample(ndraw, step_method, chain, history)

print "took :", t
pl.figure()
pl.hexbin(history['x'], history['y'])


# lets plot the samples vs. the actual distribution
logp = model_logp(model)

pts = list(product(np.linspace(-2, 0, 1000), np.linspace(-1,1, 1000)))

values = np.array([logp({'x' : np.array([vx]), 'y' : np.array([vy])}) for vx,vy in pts])
pl.figure()

p = np.array(pts)
xs, ys = p[:,0], p[:,1]
pl.hexbin(xs, ys, exp(values))
pl.show()
