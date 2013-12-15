import numpy as np
from .core import *
import copy
import types

__all__ = ['NpTrace', 'MultiTrace']

class NpTrace(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, vars):
        vars = list(vars)
        model = vars[0].model
        self.f = model.fastfn(vars)
        self.vars = vars
        self.varnames = list(map(str, vars))
        self.samples = dict((v, ListArray()) for v in self.varnames)

    def record(self, point):
        """
        Records the position of a chain at a certain point in time.
        """
        for var, value in zip(self.varnames, self.f(point)):
            self.samples[var].append(value)
        return self

    def __getitem__(self, index_value):
        """
        Return copy NpTrace with sliced sample values if a slice is passed,
        or the array of samples if a varname is passed.
        """

        if isinstance(index_value, slice):

            sliced_trace = NpTrace(self.vars)
            sliced_trace.samples = dict((name, vals[index_value]) for (name, vals) in self.samples.items())

            return sliced_trace

        else:
            try:
                return self.point(index_value)
            except ValueError:
                pass
            except TypeError:
                pass

            return self.samples[str(index_value)].value

    def __len__(self):
        return len(self.samples[self.varnames[0]])

    def point(self, index):
        return dict((k, v.value[index]) for (k, v) in self.samples.items())


class ListArray(object):
    def __init__(self, *args):
        self.vals = list(args)

    @property
    def value(self):
        if len(self.vals) > 1:
            self.vals = [np.concatenate(self.vals, axis=0)]

        return self.vals[0]

    def __getitem__(self, idx): 
        return ListArray(self.value[idx])


    def append(self, v):
        self.vals.append(v[np.newaxis])

    def __len__(self):
        if self.vals:
            return self.value.shape[0]
        else:
            return 0


class MultiTrace(object):
    def __init__(self, traces, vars=None):
        try:
            self.traces = list(traces)
        except TypeError:
            if vars is None:
                raise ValueError("vars can't be None if trace count specified")
            self.traces = [NpTrace(vars) for _ in range(traces)]

    def __getitem__(self, index_value):

        item_list = [h[index_value] for h in self.traces]

        if isinstance(index_value, slice):
            return MultiTrace(item_list)
        return item_list

    @property
    def varnames(self):
        return self.traces[0].varnames

    def point(self, index):
        return [h.point(index) for h in self.traces]

    def combined(self):
        # Returns a trace consisting of concatenated MultiTrace elements
        h = NpTrace(self.traces[0].vars)
        for k in self.traces[0].samples:
            h.samples[k].vals = [s[k] for s in self.traces]
        return h
