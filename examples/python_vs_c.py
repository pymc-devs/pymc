from pymc import *
import numpy as np 
import theano.tensor as t

#import pydevd 
#pydevd.set_pm_excepthook()

def invlogit(x):
    return np.exp(x)/(1 + np.exp(x)) 

npred = 4
n = 4000

effects_a = np.random.normal(size = npred)
predictors = np.random.normal( size = (n, npred))


outcomes = np.random.binomial(1, invlogit(np.sum(effects_a[None,:] * predictors, 1)))


model = Model()


def tinvlogit(x):
    return t.exp(x)/(1 + t.exp(x)) 

effects = AddVar(model, 'effects', Normal(mu = 0, tau = 2.**-2), (1, npred))
p = tinvlogit(sum(effects * predictors, 1))

AddData(model, outcomes, Bernoulli(p))


#make a chain with some starting point 
chain = {'effects' : np.zeros((1,npred))}


from theano import ProfileMode
print "python"
mode = theano.ProfileMode(optimizer='fast_run', linker = 'py')
print model_logp_dlogp(model, mode = mode)(chain)
mode.print_summary()

print "C"
mode = theano.ProfileMode(optimizer='fast_run', linker = 'c|py')
print model_logp_dlogp(model, mode = mode)(chain)
mode.print_summary()

