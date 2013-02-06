from pymc import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))
n = 1
start = {'x' : np.zeros(n)}

model = Model(test = start)
Var = model.Var 
Data = model.Data 


x = Var('x', Normal(mu = 0, tau = 1.), (n))

#make a chain with some starting point 




#move the chain to the MAP which should be a good starting point
start = find_MAP(model, start)
hess = approx_hess(model, start) #find a good orientation using the hessian at the MAP

#step_method = split_hmc_step(model, model.vars, hess, chain, hmc_cov)
step_method = hmc_step(model, model.vars, hess, is_cov = False)

ndraw = 3e3
history, state, t = sample(ndraw, step_method, start)

print "took :", t
