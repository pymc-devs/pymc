from mcex import *
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
chain = {'effects' : np.zeros((1,npred))}



#move the chain to the MAP which should be a good starting point
chain = find_MAP(model, chain)
hmc_cov = approx_cov(model, chain) #find a good orientation using the hessian at the MAP

#step_method = hmc_step(model, model.vars, hmc_cov) 
step_method = split_hmc_step(model, model.vars, hmc_cov, chain, hmc_cov) 

ndraw = 3e3

history = NpHistory(model.vars, ndraw)
state, t = sample(ndraw, step_method, chain, history)
print "took :", t