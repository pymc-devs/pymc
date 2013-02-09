from pymc import *

import theano.tensor as t
import numpy as np 

def invlogit(x):
    import numpy as np
    return np.exp(x)/(1 + np.exp(x)) 

npred = 4
n = 4000

effects_a = np.random.normal(size = npred)
predictors = np.random.normal( size = (n, npred))


outcomes = np.random.binomial(1, invlogit(np.sum(effects_a[None,:] * predictors, 1)))

#make a chain with some starting point 
start = {'effects' : np.zeros((1,npred))}

model = Model(test = start)
Var = model.Var
Data = model.Data 

def tinvlogit(x):
    import theano.tensor as t
    return t.exp(x)/(1 + t.exp(x)) 

effects = Var('effects', Normal(mu = 0, tau = 2.**-2), (1, npred))
p = tinvlogit(sum(effects * predictors, 1))

Data(outcomes, Bernoulli(p))





#move the chain to the MAP which should be a good starting point
start = find_MAP(model, start)
hess = diag(approx_hess(model, start)) #find a good orientation using the hessian at the MAP

step_method = hmc_step(model, model.vars, hess, is_cov = False) 
#step_method = split_hmc_step(model, model.vars, hess,  start, hess, is_cov = False) 

history, state, t = psample(3e2, step_method, start, threads = 2 )
print "took :", t
