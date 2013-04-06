'''
Created on Mar 15, 2011

@author: jsalvatier
'''
import numpy as np
from core import *

__all__ = ['NpTrace', 'MultiTrace']

class NpTrace(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, vars):
        self.f = compilef(vars)
        self.vars = map(str,vars)
        self.samples = dict((v, ListArray()) for v in self.vars)
    
    def __add__(self, point):
        """
        records the position of a chain at a certain point in time
        """
        for var, value in zip(self.vars, self.f(point)):
            self.samples[var].append(value)
        return self
        
    def __getitem__(self, key): 
        return self.samples[str(key)].value

    def point(self, index):
        return Point((k, v.value[index]) for (k,v) in self.samples.iteritems())

class ListArray(object):
    def __init__(self):
        self.vals = []

    @property
    def value(self):
        if len(self.vals) > 1:
            self.vals = [np.concatenate(self.vals, axis =0)]
        return self.vals[0]

    def append(self, v):
        self.vals.append(v[np.newaxis])
        

class MultiTrace(object): 
    def __init__(self, traces): 
        self.traces = traces 

    def __getitem__(self, key): 
        return [h[key] for h in self.traces]
    def point(self, index): 
        return [h.point(index) for h in self.traces]

    def combined(self):
        h = NpTrace()
        for k in self.traces[0].samples: 
            h.samples[k] = np.concatenate([s[k] for s in self.traces])
            h.n = h.samples[k].shape[0]
        return h
