from mcex import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()


data = np.random.normal(size = (3, 20))

x = FreeVariable('x', (3,1), 'float64')
x_prior = Normal(value = x, mu = .5, tau = 2.**-2) #attach a prior to variable x
ydata = Normal(value = data, mu = x, tau = .75**-2)

#make a chain with some starting point 
chain = ChainState({'x' : [0.2,.3,.1]})


hmc_model = Model(free_vars = [x], 
                  logps = [x_prior, ydata ],
                  derivative_vars = [x])
mapping = VariableMapping([x])

#move the chain to the MAP which should be a good starting point
find_MAP(mapping, hmc_model, chain)
hmc_cov = approx_hessian(mapping, hmc_model, chain) #find a good orientation using the hessian at the MAP

step_method = CompoundStep([hmc.HMCStep(hmc_model,mapping, hmc_cov)])


ndraw = 1e3

history = NpHistory(hmc_model, ndraw) # an object that keeps track
print "took :", sample(ndraw, step_method, chain, history)