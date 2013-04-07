from ..core import * 
import numpy as np 
from numpy.random import uniform
from numpy import log , isfinite

__all__ = ['ArrayStep', 'metrop_select', 'SamplerHist']

class ArrayStep(object):
    def __init__(self, vars, fs, allvars = False):
        self.ordering = ArrayOrdering(vars)
        self.fs = fs
        self.allvars = allvars
        
    def step(self, point):
        bij = DictToArrayBijection(self.ordering, point)
        
        inputs = map(bij.mapf, self.fs) 
        if self.allvars:
            inputs += [point]

        apoint = self.astep(bij.map(point), *inputs)
        return bij.rmap(apoint)

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
