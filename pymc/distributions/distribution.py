import theano.tensor as t
from theano.tensor.var import TensorVariable
import theano
import numpy as np
from ..model import *
import warnings
from inspect import getargspec

__all__ = ['DensityDist', 'TensorDist', 'evaluate', 'Arbitrary', 'Continuous', 'Discrete']


class Distribution(object):
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



def get_test_val(dist, val):
    try:
        val = getattr(dist, val)
    except TypeError:
        pass

    if hasattr(val, '__call__'):
        val = val(dist)

    if isinstance(val, TensorVariable):
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

    def makeFreeRV(self, name):
        return TheanoFreeRV(name=name, distribution=self)

    def makeObservedRV(self, name, data):
        return TheanoObservedRV(name=name, data=data, distribution=self)


class TheanoFreeRV(Factor, TensorVariable):
    def __init__(self, type=None, owner=None, index=None, name=None, distribution=None):
        if type is None:
            type = distribution.type
        TensorVariable.__init__(self, type, owner, index, name)

        if distribution is not None:
            self.dshape = tuple(distribution.shape)
            self.dsize = int(np.prod(distribution.shape))
            self.distribution = distribution
            testval = distribution.default(distribution.testval)
            self.tag.test_value = np.ones(
                distribution.shape, distribution.dtype) * get_test_val(distribution, testval)
            self.logpt = distribution.logp(self)

class TheanoObservedRV(Factor):
    def __init__(self, name, data, distribution):
        self.name = name
        data = getattr(data, 'values', data) #handle pandas
        args = as_iterargs(data)

        if len(args) > 1:
            params = getargspec(distribution.logp).args
            args = [t.constant(d, name=name + "_" + param) 
                    for d,param in zip(args,params) ]
        else: 
            args = [t.constant(args[0], name=name)]
            
        self.logpt = distribution.logp(*args)


def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    if hasattr(data, 'columns'):  # data frames
        return [np.asarray(data[c]) for c in data.columns]
    else:
        return [data]


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

