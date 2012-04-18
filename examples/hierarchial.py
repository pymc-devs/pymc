from mcex import *
import theano.tensor as T 
from numpy import random, sum as nsum, ones, concatenate, newaxis, dot, arange
from __builtin__ import map
import numpy as np 
#import pydevd 
#pydevd.set_pm_excepthook()


random.seed(1)

n_groups = 29
no_pergroup = 30 
n_observed = no_pergroup * n_groups
n_group_predictors = 2
n_predictors = 3

group = concatenate([ [i]*no_pergroup for i in range(n_groups)])
group_predictors = random.normal(size = (n_groups, n_group_predictors)) #random.normal(size = (n_groups, n_group_predictors))
predictors       = random.normal(size = (n_observed, n_predictors))

group_effects_a = random.normal( size = (n_group_predictors, n_predictors))
effects_a = random.normal(size = (n_groups, n_predictors)) + dot(group_predictors, group_effects_a)

y = nsum(effects_a[group, :] * predictors, 1) + random.normal(size = (n_observed))




model = Model()

#m_g ~ N(0, .1)
group_effects = FreeVariable("group_effects", (1, n_group_predictors, n_predictors), 'float64')
AddVar(model, group_effects, Normal(0, .1))


# sg ~ 
sg = FreeVariable("sg", 1, 'float64')
AddVar(model, sg, Uniform(.05, 10))
#sg = 1

#m ~ N(mg * pg, sg)
effects = FreeVariable("effects", (n_groups, n_predictors), 'float64')
AddVar(model, effects, Normal( sum(group_predictors[:, :, newaxis] * group_effects ,1)  ,sg))

#s ~ 
s = FreeVariable("s", n_groups, 'float64')
AddVar(model, s, Uniform(.01, 10))

g = T.constant(group)

#y ~ Normal(m[g] * p, s)
AddVar(model, T.constant(y), Normal( sum(effects[g] * predictors, 1),s[g]))



view = ModelView(model, 'all')
                 
chain = {'sg' : np.array([2.]), 
         's'  : np.ones(n_groups) * 2.,
         'group_effects' : np.zeros((1,) + group_effects_a.shape),
         'effects' : np.zeros(effects_a.shape ) }

chain2 = find_MAP(view, chain)

hmc_cov = approx_cov( view, chain2) #find a good orientation using the hessian at the MAP

step_method = CompoundStep([hmc.HMCStep(view, hmc_cov, step_size_scaling = .05)])


ndraw = 3e3

history = NpHistory(view, ndraw) # an object that keeps track
print "took :", sample(ndraw, step_method, chain2, history)

def project(chain, name, slc, value):
    c = chain.copy() 
    c[name] = c[name].copy()
    c[name][slc] = value
    return c

def get_like(chain, name, slc):
    def like(x):
        return view.evaluate(project(chain2, name, slc, x))[0]
    return like
    
#x = arange(.001, 5, .001)
#y = np.array(map(get_like(chain2, 's', 0), x))