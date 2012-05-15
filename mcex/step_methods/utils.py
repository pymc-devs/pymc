from ..core import *
import numpy as np 

def array_step(astep, f, vars):
    mapping = DASpaceMap(vars)
    
    def step(state, chain):
        def fn( a):
            return f(mapping.rproject(a, chain))
        
        state, achain = astep(fn, state, mapping.project(chain))
        return state, mapping.rproject(achain, chain)
    return step


from numpy.random import uniform, normal
from numpy import dot, log , isfinite

def metrop_select(mr, q, q0):
    if isfinite(mr) and log(uniform()) < mr:
        return q
    else: 
        return q0
    
def cholesky_normal(size , cholC):
    return dot(normal(size = size), cholC)




class SamplerHist(object):
    def __init__(self):
        self.metrops =[]
    def acceptr(self):
        return np.minimum(np.exp(self.metrops), 1)