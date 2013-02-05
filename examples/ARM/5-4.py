'''
Created on May 18, 2012

@author: jsalvatier
'''
import numpy as np 
from pymc import * 
import theano.tensor as t 
from  utils import * 

data  = readtable('wells.dat', delimiter = ' ', quotechar='"')

id, switched = data[:2]
arsenic, dist, assoc, educ = map(demean, data[2:])

switched = switched.astype(bool)

dist100 = dist/100.
educ4 = educ/4.

predictors = array([np.ones_like(id), dist, arsenic, dist * arsenic, assoc, educ]).T
npr = predictors.shape[1]

model = Model()

effects = AddVar(model, 'effects', Normal(mu = 0, tau = 10.**-2), (1,  npr))
p = invlogit(sum(effects * predictors, 1))

AddData(model, switched, Bernoulli(p))


#make a chain with some starting point 
chain = {'effects' : np.zeros((1,npr))}

#move the chain to the MAP which should be a good starting point
map = find_MAP(model, chain)
hmc_hess = approx_hess(model, chain) #find a good orientation using the hessian at the MAP

step_method = hmc_step(model, model.vars, hmc_hess, is_cov = False)

history, state, t = sample(3e2, step_method, map)
print "took :", t