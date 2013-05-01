from ..core import *
import numpy as np
from numpy.random import uniform
from numpy import log , isfinite

__all__ = ['ArrayStep', 'metrop_select', 'SamplerHist']

# TODO Add docstrings to ArrayStep
class ArrayStep(object):
    def __init__(self, vars, fs, allvars = False, tune=False):
        self.ordering = ArrayOrdering(vars)
        self.fs = fs
        self.allvars = allvars
        self.tune = tune

    def step(self, point):
        bij = DictToArrayBijection(self.ordering, point)

        inputs = map(bij.mapf, self.fs)
        if self.allvars:
            inputs += [point]

        apoint = self.astep(bij.map(point), *inputs)
        return bij.rmap(apoint)

def metrop_select(mr, q, q0):
    # Perform rejection/acceptance step for Metropolis class samplers

    # Compare acceptance ratio to uniform random number
    if isfinite(mr) and log(uniform()) < mr:
        # Accept proposed value
        return q, True
    else:
        # Reject proposed value
        return q0, False


class SamplerHist(object):
    def __init__(self):
        self.metrops =[]
    def acceptr(self):
        return np.minimum(np.exp(self.metrops), 1)
