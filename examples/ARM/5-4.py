'''
Created on May 18, 2012

@author: jsalvatier
'''
import numpy as np 
from mcex import * 
import theano.tensor as t 

data = np.loadtxt('wells.dat',  usecols = range(1,6), skiprows = 1)

def demean(x, axis = 0):
    return x - np.mean(x, axis)
cdist100 = demean(data[:,2, None]/100.)
carsenic = demean(data[:,1, None])

assoc = demean(data[:,3, None])
educ4 = demean(data[:,4, None])/4.



data[:,2] /= 100. 

n = data.shape[0]
switch = data[:,0]
data = data - np.mean(data, 0)

predictors = concatenate((ones((n,1)), cdist100, carsenic, cdist100*carsenic, assoc, educ4), 1)
npr = predictors.shape[1]

model = Model()


def tinvlogit(x):
    return t.exp(x)/(1 + t.exp(x)) 

effects = AddVar(model, 'effects', Normal(mu = 0, tau = 10.**-2), (1,  npr))
p = tinvlogit(sum(effects * predictors, 1))

AddData(model, switch, Bernoulli(p))


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