import theano.tensor as t
import numpy as np
from ..model import Model

__all__ = ['DensityDist', 'Distribution', 'Continuous', 'Discrete']


class Distribution(object):
    """Statistical distribution"""
    def __new__(cls, name, *args, **kwargs):
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to use the Normal('x', 0,1) syntax. Add a 'with model:' block")

        if isinstance(name, str):  
            data = kwargs.pop('observed', None)
            dist = cls.dist(*args, **kwargs)
            return model.Var(name, dist, data)
        elif name is None:
            return object.__new__(cls) #for pickle
        else: 
            raise TypeError("needed name or None but got: " + name)

    def __getnewargs__(self):
        return None, 

    @classmethod
    def dist(cls, *args, **kwargs):
        dist = object.__new__(cls)
        dist.__init__(*args, **kwargs)
        return dist

    def __init__(self, shape, dtype, testval=None):
        self.shape = np.atleast_1d(shape)
        self.dtype = dtype
        self.type = TensorType(dtype, shape)
        self.testval = testval

    def default(self, testval):
        if testval is None:
            for val in self.default_testvals:
                if hasattr(self, val):
                    return getattr(self, val)
            raise AttributeError(str(self) + " has no default value to use, checked for: " +
                                 str(self.default_testvals) + " pass testval argument or provide one of these.")
        return testval

def TensorType(dtype, shape):
    return t.TensorType(str(dtype), np.atleast_1d(shape) == 1)

class Discrete(Distribution): 
    """Base class for discrete distributions"""
    def __init__(self, shape=(), dtype='int64', *args, **kwargs):
        Distribution.__init__(self, shape, dtype, *args, **kwargs)
        self.default_testvals = ['mode']

class Continuous(Distribution): 
    """Base class for continuous distributions"""
    def __init__(self, shape=(), dtype='float64', *args, **kwargs):
        Distribution.__init__(self, shape, dtype, *args, **kwargs)
        self.default_testvals = ['median', 'mean', 'mode']

class DensityDist(Distribution):
    """Distribution based on a given log density function."""
    def __init__(self, logp, shape=(), dtype='float64', *args, **kwargs):
        Distribution.__init__(self, shape, dtype, *args, **kwargs)
        self.default_testvals = [0]

        self.logpf = logp

    def logp(self, value): 
        return self.logpf(value)
