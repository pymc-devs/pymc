from ..core import *
import numpy as np 

class array_step(object):
    def __init__(self, vars, fs, provide_full = False):
        self.mapping = DASpaceMap(vars)
        self.fs = fs
        self.provide_full = provide_full
        
    def step(self, state, chain):
        
        fns = [ArrWrap(f, self.mapping, chain) for f in self.fs]
        if self.provide_full:
            fns += [chain]	
        
        state, achain = self.astep(state, self.mapping.project(chain), *fns)
        return state, self.mapping.rproject(achain, chain)

class ArrWrap(object):
    def __init__(self, f, mapping, chain):
        self.f = f
        self.mapping = mapping
        self.chain = chain

    def __call__(self, a):
        return self.f(self.mapping.rproject(a, self.chain))

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
