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
        self.vars = vars
        self.varnames = map(str, vars)
        self.samples = dict((v, ListArray()) for v in self.varnames)

    def record(self, point):
        """
        records the position of a chain at a certain point in time
        """
        for var, value in zip(self.varnames, self.f(point)):
            self.samples[var].append(value)
        return self

    def __getitem__(self, key):
        try:
            return self.point(key)
        except ValueError:
            pass
        except TypeError:
            pass
        return self.samples[str(key)].value

    def point(self, index):
        return dict((k, v.value[index]) for (k, v) in self.samples.iteritems())


class ListArray(object):
    def __init__(self):
        self.vals = []

    @property
    def value(self):
        if len(self.vals) > 1:
            self.vals = [np.concatenate(self.vals, axis=0)]
        return self.vals[0]

    def append(self, v):
        self.vals.append(v[np.newaxis])


class MultiTrace(object):
    def __init__(self, traces, vars=None):
        try:
            self.traces = list(traces)
        except TypeError:
            if vars is None:
                raise ValueError("vars can't be None if trace count specified")
            self.traces = [NpTrace(vars) for _ in xrange(traces)]

    def __getitem__(self, key):
        return [h[key] for h in self.traces]

    def point(self, index):
        return [h.point(index) for h in self.traces]

    def combined(self):
        h = NpTrace(self.traces[0].vars)
        for k in self.traces[0].samples:
            h.samples[k].vals = [s[k] for s in self.traces]
        return h
