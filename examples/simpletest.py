from mcex import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (3, 20))


model = Model()


x = AddVar(model, 'x', Normal(mu = .5, tau = 2.**-2), (3,1))


z = AddVar(model, 'z', Beta(alpha = 10, beta =5.5))

AddData(model, data, Normal(mu = x, tau = .75**-2))


#make a chain with some starting point 
chain = {'x' : np.array([[0.2],[.3],[.1]]),
         'z' : np.array([.5])}



#move the chain to the MAP which should be a good starting point
chain = find_MAP(model, chain)
hmc_cov = approx_cov(model, model.vars, chain) #find a good orientation using the hessian at the MAP

step_method = HMCStep(model, model.vars, hmc_cov)


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
history = NpHistory(model.vars, ndraw) # an object that keeps track
print "took :", sample(ndraw, step_method, chain, history)