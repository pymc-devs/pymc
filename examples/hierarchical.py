from pymc import *
import theano.tensor as T 
from numpy import random, sum as nsum, ones, concatenate, newaxis, dot, arange
from __builtin__ import map
import numpy as np 



random.seed(1)

n_groups = 10
no_pergroup = 30 
n_observed = no_pergroup * n_groups
n_group_predictors = 1
n_predictors = 3

group = concatenate([ [i]*no_pergroup for i in range(n_groups)])
group_predictors = random.normal(size = (n_groups, n_group_predictors)) #random.normal(size = (n_groups, n_group_predictors))
predictors       = random.normal(size = (n_observed, n_predictors))

group_effects_a = random.normal( size = (n_group_predictors, n_predictors))
effects_a = random.normal(size = (n_groups, n_predictors)) + dot(group_predictors, group_effects_a)

y = nsum(effects_a[group, :] * predictors, 1) + random.normal(size = (n_observed))



model = Model()
Var = model.Var
Data = model.Data 

#m_g ~ N(0, .1)
group_effects = Var("group_effects", Normal(0, .1), (1, n_group_predictors, n_predictors))


# sg ~ Uniform(.05, 10)
sg = Var("sg", Uniform(.05, 10), testval = 2.)

#m ~ N(mg * pg, sg)
effects = Var("effects", 
                 Normal( sum(group_predictors[:, :, newaxis] * group_effects ,1)  ,sg**-2),
                 (n_groups, n_predictors))

#s ~ 
s = Var("s", Uniform(.01, 10), n_groups)

g = T.constant(group)

#y ~ Normal(m[g] * p, s)
Data(y, Normal( sum(effects[g] * predictors, 1),s[g]**-2))

                 

start = find_MAP(model)
h = approx_hess(model, start) #find a good orientation using the hessian at the MAP

step = HamiltonianMC(model, model.vars, h, is_cov = False)


print "took :", sample(3e3, step, start)
