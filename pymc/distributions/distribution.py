import theano.tensor as t
import numpy as np
from ..model import *
import warnings

__all__ = ['DensityDist', 'TensorDist', 'evaluate', 'Arbitrary', 'Continuous', 'Discrete']


class Distribution(object):
    def __new__(cls, name, *args, **kwargs):
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to use the Normal('x', 0,1) syntax. Add a 'with model:' block")

        if 'observed' in kwargs:
            data = kwargs.pop('observed')
            dist = cls.dist(*args, **kwargs)
            return model.Data(name, dist, data)
        elif isinstance(name, str):  
            dist = cls.dist(*args, **kwargs)
            return model.Var(name, dist)
        elif name is None:
            return object.__new__(cls)
        else: 
            raise TypeError("needed name or None but got: " + name)

    def __getnewargs__(self):
        return None, 

    @classmethod
    def dist(cls, *args, **kwargs):
        dist = object.__new__(cls)
        dist.__init__(*args, **kwargs)
        return dist



def get_test_val(dist, val):
    try:
        val = getattr(dist, val)
    except TypeError:
        pass

    if hasattr(val, '__call__'):
        val = val(dist)

    if isinstance(val, t.TensorVariable):
        return val.tag.test_value
    else:
        return val


# Convenience function for evaluating distributions at test point
evaluate = lambda dist: dist.tag.test_value


class TensorDist(Distribution):
    def default(self, testval):
        if testval is None:
            for val in self.default_testvals:
                if hasattr(self, val):
                    return getattr(self, val)
            raise AttributeError(str(self) + " has no default value to use, checked for: " +
                                 str(self.default_testvals) + " pass testval argument or provide one of these.")
        return testval

    def makevar(self, name):
        var = self.type(name)
        var.dshape = tuple(self.shape)
        var.dsize = int(np.prod(self.shape))
        var.distribution = self

        testval = self.default(self.testval)
        var.tag.test_value = np.ones(
            self.shape, self.dtype) * get_test_val(self, testval)

        var.logp = self.logp(var)
        return var

def TensorType(dtype, shape):
    return t.TensorType(str(dtype), np.atleast_1d(shape) == 1)



class Arbitrary(TensorDist): 
    def __init__(self, shape=(), dtype='float64', *args, **kwargs):
        TensorDist.__init__(self, *args, **kwargs)

        self.shape = np.atleast_1d(shape)
        self.type = TensorType(dtype, shape)
        self.__dict__.update(locals())

class Discrete(TensorDist): 
    def __init__(self, shape=(), dtype='int64', testval=None, *args, **kwargs):
        TensorDist.__init__(self, *args, **kwargs)
        self.shape = np.atleast_1d(shape)
        self.type = TensorType(dtype, shape)
        self.default_testvals = ['mode']
        self.__dict__.update(locals())

class Continuous(TensorDist): 
    def __init__(self, shape=(), dtype='float64', testval=None, *args, **kwargs):
        TensorDist.__init__(self, *args, **kwargs)

        shape = np.atleast_1d(shape)
        type = TensorType(dtype, shape)
        default_testvals = ['median', 'mean', 'mode']

        self.__dict__.update(locals())


class DensityDist(Arbitrary):
    def __init__(self, logp, *args, **kwargs):
        Arbitrary.__init__(self, *args, **kwargs)

        logpf = logp
        testval = 0
        
        self.__dict__.update(locals())

    def logp(self, value): 
        return self.logpf(value)

