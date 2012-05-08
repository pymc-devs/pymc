from mcex import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))


model = Model()

n = 1
x = AddVar(model, 'x', Normal(mu = 0, tau = 1.), (n))

#make a chain with some starting point 
chain = {'x' : np.zeros(n)}



#move the chain to the MAP which should be a good starting point
chain = find_MAP(model, chain)
hmc_cov = approx_cov(model, chain) #find a good orientation using the hessian at the MAP

step_method = split_hmc_step(model, model.vars, hmc_cov, chain, hmc_cov)
#step_method = hmc_step(model, model.vars, hmc_cov)

ndraw = 3e3
history = NpHistory(model.vars, ndraw) # an object that keeps track
state, t = sample(ndraw, step_method, chain, history)

print "took :", t