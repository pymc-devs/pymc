'''
Created on May 10, 2012

@author: jsalvatier
'''
from pymc import * 
from numpy.random import normal
import numpy as np 
import pylab as pl 
from itertools import product

xa = 0
ya = 0
n = 200
data = normal( size = n) + xa + ya

start = {'x' : 0,
         'y' : 0}
model = Model(test_point = start)
Data = model.Data 
Var = model.Var
"""
This model is U shaped because of the non identifiability. 
As n increases, the walls become steeper but the distribution does not shrink towards the mode. 
As n increases this distribution gets harder and harder for HMC to sample.

Low Flip HMC seems to do a bit better.

This example comes from 
Discussion of Riemann manifold Langevin and
Hamiltonian Monte Carlo methods by M.
Girolami and B. Calderhead

http://arxiv.org/abs/1011.0057
"""

x = Var('x', Normal(0, 1))
y = Var('y', Normal(0, 1))

Data(data, Normal(x + y**2, 1.) )




hess = np.ones(2)*np.diag(approx_hess(model, start))[0]


#step_method = hmc_lowflip_step(model, model.vars, hess,is_cov = False, step_size = .25, a = .9)
step_method = hmc_step(model, model.vars, hess,is_cov = False)

history, state, t = sample(3e3, step_method, start)

print "took :", t
pl.figure()
pl.hexbin(history['x'], history['y'])


# lets plot the samples vs. the actual distribution
logp = model.logp()

pts = list(product(np.linspace(-2, 2, 1000), np.linspace(-1,1, 1000)))

values = np.array([logp({'x' : np.array([vx]), 'y' : np.array([vy])}) for vx,vy in pts])
pl.figure()

p = np.array(pts)
xs, ys = p[:,0], p[:,1]
pl.hexbin(xs, ys, exp(values))
pl.show()
