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
start = {'x' : np.zeros(n)}



#move the chain to the MAP which should be a good starting point
start = find_MAP(model, start)
hess = approx_hess(model, start) #find a good orientation using the hessian at the MAP

#step_method = split_hmc_step(model, model.vars, hess, chain, hmc_cov)
step_method = hmc_step(model, model.vars, hess, is_cov = False)

ndraw = 3e3
history, state, t = sample(ndraw, step_method, start)

print "took :", t