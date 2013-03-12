'''
Created on May 18, 2012

@author: jsalvatier
'''
import numpy as np 
from pymc import * 
import theano.tensor as t 
import pandas as pd

data  = pd.read_csv('wells.dat', skip_header = True, delimiter = ' ', index_col = 'id', dtype = {'switch': np.int8})

col = data.columns

credit = data[col[-1]]
P =data[col[:-1]]

P = P - P.mean()
P['1'] = 1

Pa = np.array(P)

model = Model()

effects = model.Var('effects', Normal(mu = 0, tau = 100.**-2), len(P.columns))
p = sigmoid(dot(Pa, effects))

model.Data(np.array(credit), Bernoulli(p))


#move the chain to the MAP which should be a good starting point
start = find_MAP(model)
H = model.fn(dlogp(n = 2)) #find a good orientation using the hessian at the MAP
hess = H(start)

step_method = hmc_step(model, model.vars, hess, is_cov = False)

trace, state, t = sample(3e2, step_method, start)
print "took :", t
