import theano.tensor as t
import numpy as np
from ..quickclass import *
from ..model import *
import warnings

__all__ = ['DensityDist', 'TensorDist', 'tensordist', 'continuous',
           'discrete', 'arbitrary']


class Distribution(object):
    def __new__(cls,name, *args, **kwargs):
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to use the Normal('x', 0,1) syntax. Add a 'with model:' block")

        if 'observed' in kwargs:
            obs = kwargs.pop('observed')
            dist = cls.dist(*args, **kwargs)
            return model.Data(obs, dist)
        else:
            dist = cls.dist(*args, **kwargs)
            return model.Var(name, dist)

    @classmethod
    def dist(cls, *args,**kwargs):
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

        testval = self.default(self.testval)
        var.tag.test_value = np.ones(
            self.shape, self.dtype) * get_test_val(self, testval)
        return var


def tensordist(defaults):
    def decorator(fn):
        fn = withdefaults(defaults)(fn)
        return quickclass(TensorDist)(fn)
    return decorator


def TensorType(dtype, shape):
    return t.TensorType(str(dtype), np.atleast_1d(shape) == 1)


def continuous(shape=(), dtype='float64', testval=None):
    shape = np.atleast_1d(shape)
    type = TensorType(dtype, shape)
    default_testvals = ['median', 'mean', 'mode']
    return locals()


def discrete(shape=(), dtype='int64', testval=None):
    shape = np.atleast_1d(shape)
    type = TensorType(dtype, shape)
    default_testvals = ['mode']
    return locals()


def arbitrary(shape=(), dtype='float64', testval=0):
    shape = np.atleast_1d(shape)
    type = TensorType(dtype, shape)
    return locals()


@tensordist(arbitrary)
def DensityDist(logp):
    return locals()
