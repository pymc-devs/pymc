from pymc import *
import numpy as np 
import theano.tensor as t

#import pydevd 
#pydevd.set_pm_excepthook()

def invlogit(x):
    import numpy as np
    return np.exp(x)/(1 + np.exp(x)) 

npred = 4
n = 4000

effects_a = np.random.normal(size = npred)
predictors = np.random.normal( size = (n, npred))


outcomes = np.random.binomial(1, invlogit(np.sum(effects_a[None,:] * predictors, 1)))

#make a chain with some starting point 
start = {'effects' : np.zeros((1,npred))}

model = Model(test_point = start)
Var = model.Var
Data = model.Data 


def tinvlogit(x):
    import theano.tensor as t
    return t.exp(x)/(1 + t.exp(x)) 

effects = Var('effects', Normal(mu = 0, tau = 2.**-2), (1, npred))
p = tinvlogit(sum(effects * predictors, 1))

Data(outcomes, Bernoulli(p))



calc = [logp, dlogp()]

from theano import ProfileMode
print "python"
mode = theano.ProfileMode(optimizer='fast_run', linker = 'py')
print model.fn(calc, mode = mode)(start)
mode.print_summary()

print "C"
mode = theano.ProfileMode(optimizer='fast_run', linker = 'c|py')
print model.fn(calc, mode = mode)(start)
mode.print_summary()

