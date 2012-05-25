'''
Created on May 18, 2012

@author: jsalvatier
'''
import numpy as np 
from mcex import * 
import theano.tensor as t 
import utils 

data= utils.readcsv('wells.dat', delimiter = ' ', quotechar='"')

switched = data['switch'].astype(bool)

def demean(x, axis = 0):
    return x - np.mean(x, axis)
data = dict((n, demean(v)[:, None]) for n, v in data.iteritems())

data['dist'] = data['dist']/100.
data['educ'] = data['educ']/4.

n = data['switch'].shape[0]

predictors = concatenate((ones((n,1)), data['dist'], data['arsenic'], data['dist']*data['arsenic'], data['assoc'], data['educ']), 1)
npr = predictors.shape[1]

model = Model()

def tinvlogit(x):
    return t.exp(x)/(1 + t.exp(x)) 

effects = AddVar(model, 'effects', Normal(mu = 0, tau = 10.**-2), (1,  npr))
p = tinvlogit(sum(effects * predictors, 1))

AddData(model, switched, Bernoulli(p))


#make a chain with some starting point 
chain = {'effects' : np.zeros((1,npr))}

#move the chain to the MAP which should be a good starting point
map = find_MAP(model, chain)
hmc_cov = approx_cov(model, chain) #find a good orientation using the hessian at the MAP

#step_method = hmc_step(model, model.vars, hmc_cov) 
step_method = hmc_step(model, model.vars, hmc_cov)#,step_size_scaling = .1, trajectory_length =1. ) 

ndraw = 3e3

history = NpHistory(model.vars, ndraw)
state, t = sample(ndraw, step_method, map, history)
print "took :", t


def tocorr(c):
    w = np.diag(1/np.diagonal(c)**.5)
    return w.dot(c).dot(w)