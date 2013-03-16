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

model = Model()
Var = model.Var
Data = model.Data 


def tinvlogit(x):
    import theano.tensor as t
    return t.exp(x)/(1 + t.exp(x)) 

effects = Var('effects', Normal(mu = 0, tau = 2.**-2), (1, npred))
p = tinvlogit(sum(effects * predictors, 1))

Data(outcomes, Bernoulli(p))




start = model.test_point

from theano import ProfileMode

for mode in [ProfileMode(linker = 'py'),
             ProfileMode(linker = 'c|py')]:

    print mode
    logp = model.logp
    f = compilef([logp, gradient(logp)], mode)
    print f(start)
    mode.print_summary()

