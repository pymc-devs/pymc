from mcex import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))

x = FreeVariable('x', (3,1), 'float64')
z = FreeVariable('z', (1,), 'float64')

x_prior = Normal(value = x, mu = .5, tau = 2.**-2) #attach a prior to variable x
ydata = Normal(value = data, mu = x, tau = .75**-2)

z_prior = Beta(value = z, alpha = 10, beta =5.5)

#make a chain with some starting point 
chain = {'x' : np.array([[0.2],[.3],[.1]]),
         'z' : np.array([.5])}


hmc_model = ModelView(free_vars = [x,z], 
                  logps = [x_prior, ydata , z_prior],
                  derivative_vars = [x,z])

#move the chain to the MAP which should be a good starting point


find_MAP(hmc_model, chain)
hmc_cov = approx_cov( hmc_model, chain) #find a good orientation using the hessian at the MAP

step_method = CompoundStep([hmc.HMCStep(hmc_model, hmc_cov)])


ndraw = 3e3
"""
print chain.values
x = np.linspace(.01, .99, 50)
y = np.array([ hmc_model.function(x = np.array([[0],[0],[0]]), z = np.array([xi]) )[0][()] for xi in x])

import pylab 
pylab.figure()
y2 = np.exp(y)/np.sum(np.exp(y))
pylab.plot(x,y2)
pylab.show()
print hmc_cov"""
history = NpHistory(hmc_model, ndraw) # an object that keeps track
print "took :", sample(ndraw, step_method, chain, history)