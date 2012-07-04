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
hmc_cov = approx_cov(model, chain) #find a good orientation using the hessian at the MAP

#step_method = split_hmc_step(model, model.vars, hmc_cov, chain, hmc_cov)
step_method = hmc_step(model, model.vars, hmc_cov)

ndraw = 3e3
history = NpHistory(model.vars, ndraw) # an object that keeps track
state, t = sample(ndraw, step_method, chain, history)

print "took :", t

from pylab import * 
subplot(2,2,1)
plot(history['x'][:,0,0])
subplot(2,2,2)
hist(history['x'][:,0,0])

subplot(2,2,3)
plot(history['z'])
subplot(2,2,4)
hist(history['x'])
show()