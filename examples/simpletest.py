from pymc import *
import numpy as np 
import theano 

#import pydevd 
#pydevd.set_pm_excepthook()
np.seterr(invalid = 'raise')

data = np.random.normal(size = (2, 20))


model = Model()


x = AddVar(model, 'x', Normal(mu = .5, tau = 2.**-2), (2,1))


z = AddVar(model, 'z', Beta(alpha = 10, beta =5.5))

AddData(model, data, Normal(mu = x, tau = .75**-2))


#make a chain with some starting point 
chain = {'x' : np.array([[0.2],[.3]]),
         'z' : np.array([.5])}


step_method = hmc_step(model, model.vars)

history,state, t = sample(1e3, step_method, chain)

print "took :", t

from pylab import * 
subplot(2,2,1)
plot(history['x'][:,0,0])
subplot(2,2,2)
hist(history['x'][:,0,0])

subplot(2,2,3)
plot(history['z'])
subplot(2,2,4)
hist(history['x'][:,0,0])
show()