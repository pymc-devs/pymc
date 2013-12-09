'''
Created on May 18, 2012

@author: jsalvatier
'''
import numpy as np
from pymc import *
import theano.tensor as t
import pandas as pd

data = pd.read_csv('data/wells.dat', delimiter=' ', index_col='id',
                   dtype={'switch': np.int8})

col = data.columns

P = data[col[1:]]
P.dist /= 100
P.educ /= 4

P = P - P.mean()
P['1'] = 1

Pa = np.array(P)

with Model() as model:
    effects = Normal(
        'effects', mu=0, tau=100. ** -2, shape=len(P.columns))
    p = sigmoid(dot(Pa, effects))

    s = Bernoulli('s', p, observed=np.array(data.switch))

if __name__ == '__main__':

    with model:
        # move the chain to the MAP which should be a good starting point
        start = find_MAP()
        H = model.fastd2logp()  # find a good orientation using the hessian at the MAP
        h = H(start)

        step = HamiltonianMC(model.vars, h)

        trace = sample(3e3, step, start)
