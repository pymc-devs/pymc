from pymc import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))
n = 1

model = Model()
with model: 
    x = Normal('x', 0, 1., n)

#start sampling at the MAP
start = find_MAP(model)
h = approx_hess(model, start) #find a good orientation using the hessian at the MAP

#step_method = split_hmc_step(model, model.vars, hess, chain, hmc_cov)
step = HamiltonianMC(model, model.vars, h, is_cov = False)

ndraw = 3e3
history, state, t = sample(ndraw, step, start)

print "took :", t
