from pymc import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))
n = 1

model = Model()
Var = model.Var 
Data = model.Data 


x = Var('x', Normal(0, tau = 1.), n)

#start sampling at the MAP
start = find_MAP(model)
h = approx_hess(model, start) #find a good orientation using the hessian at the MAP

#step_method = split_hmc_step(model, model.vars, hess, chain, hmc_cov)
step = HamiltonianMC(model, model.vars, h, is_cov = False)

ndraw = 3e3
history, state, t = sample(ndraw, step, start)

print "took :", t
