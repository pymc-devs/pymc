from pymc import *
import numpy as np 
import theano.tensor as t

#import pydevd 
#pydevd.set_pm_excepthook()

def invlogit(x):
    return np.exp(x)/(1 + np.exp(x)) 

npred = 4
n = 4000

effects_a = np.random.normal(size = npred)
predictors = np.random.normal( size = (n, npred))


outcomes = np.random.binomial(1, invlogit(np.sum(effects_a[None,:] * predictors, 1)))


model = Model()


def tinvlogit(x):
    return t.exp(x)/(1 + t.exp(x)) 

effects = AddVar(model, 'effects', Normal(mu = 0, tau = 2.**-2), (1, npred))
p = tinvlogit(sum(effects * predictors, 1))

AddData(model, outcomes, Bernoulli(p))


#make a chain with some starting point 
start = {'effects' : np.zeros((1,npred))}



#move the chain to the MAP which should be a good starting point
start = find_MAP(model, start)
hess = diag(approx_hess(model, start)) #find a good orientation using the hessian at the MAP

step_method = hmc_step(model, model.vars, hess, is_cov = False) 
#step_method = split_hmc_step(model, model.vars, hess,  start, hess, is_cov = False) 

history, state, t = sample(3e3, step_method, start)
print "took :", t