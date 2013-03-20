'''
Created on May 18, 2012

@author: jsalvatier
'''
import numpy as np 
from pymc import * 
import theano.tensor as t 
import pandas as pd

data  = pd.read_csv('data/wells.dat', delimiter = ' ', index_col = 'id', dtype = {'switch': np.int8})

col = data.columns

P =data[col[1:]]
P.dist /= 100
P.educ /= 4

P = P - P.mean()
P['1'] = 1

Pa = np.array(P)

model = Model()

effects = model.Var('effects', Normal(mu = 0, tau = 100.**-2), len(P.columns))
p = sigmoid(dot(Pa, effects))

model.Data(np.array(data.switch), Bernoulli(p))


#move the chain to the MAP which should be a good starting point
start = find_MAP(model)
H = model.d2logpc() #find a good orientation using the hessian at the MAP
h = H(start)

step_method = HamiltonianMC(model, model.vars, h)

trace, state, t = sample(3e3, step_method, start)
print "took :", t
