from ..core import *
import numpy as np 

class array_step(object):
    def __init__(self, vars, fs, provide_full = False):
        self.idxmap = IdxMap(vars)
        self.fs = fs
        self.provide_full = provide_full
        
    def step(self, state, point):
        bij = DictArrBij(self.idxmap, point)
        
        fns = map(bij.mapf, self.fs)

        if self.provide_full:
            fns += [point]	
        
        state, apoint = self.astep(state, bij.map(point), *fns)
        return state, bij.rmap(apoint)


from numpy.random import uniform
from numpy import dot, log , isfinite

def metrop_select(mr, q, q0):
    if isfinite(mr) and log(uniform()) < mr:
        return q
    else: 
        return q0
    

class SamplerHist(object):
    def __init__(self):
        self.metrops =[]
    def acceptr(self):
        return np.minimum(np.exp(self.metrops), 1)
