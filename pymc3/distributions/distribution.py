import theano.tensor as t
import numpy as np
from ..model import Model

from theano import function
from ..model import get_named_nodes

__all__ = ['DensityDist', 'Distribution', 'Continuous', 'Discrete', 'NoDistribution', 'TensorType']


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

    def __init__(self, shape, dtype, testval=None, defaults=[], transform=None):
        self.shape = np.atleast_1d(shape)
        self.dtype = dtype
        self.type = TensorType(self.dtype, self.shape)
        self.testval = testval
        self.defaults = defaults
        self.transform = transform

    def default(self):
        return self.get_test_val(self.testval, self.defaults)


    def get_test_val(self, val, defaults):
        if val is None:
            for v in defaults:
                if hasattr(self, v) and np.all(np.isfinite(self.getattr_value(v))):
                    return self.getattr_value(v)
        else:
            return self.getattr_value(val)

        if val is None:
            raise AttributeError(str(self) + " has no finite default value to use, checked: " +
                         str(defaults) + " pass testval argument or adjust so value is finite.")


    def getattr_value(self, val):
        if isinstance(val, str):
            val = getattr(self, val)

        if isinstance(val, t.TensorVariable):
            return val.tag.test_value

        return val


def TensorType(dtype, shape):
    return t.TensorType(str(dtype), np.atleast_1d(shape) == 1)

class NoDistribution(Distribution):
    def logp(self, x):
        return 0

class Discrete(Distribution):
    """Base class for discrete distributions"""
    def __init__(self, shape=(), dtype='int64', defaults=['mode'], *args, **kwargs):
        super(Discrete, self).__init__(shape, dtype, defaults=defaults, *args, **kwargs)

class Continuous(Distribution):
    """Base class for continuous distributions"""
    def __init__(self, shape=(), dtype='float64', defaults=['median', 'mean', 'mode'], *args, **kwargs):
        super(Continuous, self).__init__(shape, dtype, defaults=defaults, *args, **kwargs)

class DensityDist(Distribution):
    """Distribution based on a given log density function."""
    def __init__(self, logp, shape=(), dtype='float64',testval=0, *args, **kwargs):
        super(DensityDist, self).__init__(shape, dtype, testval, *args, **kwargs)
        self.logp = logp

class MultivariateContinuous(Continuous):

    pass

class MultivariateDiscrete(Discrete):

    pass

def draw_values(params, point=None):
    """
    Draw (fix) parameter values. Handles a number of cases:

        1) The parameter is a scalar
        2) The parameter is an *RV

            a) parameter can be fixed to the value in the point
            b) parameter can be fixed by sampling from the *RV
            c) parameter can be fixed using tag.test_value (last resort)

        3) The parameter is a tensor variable/constant. Can be evaluated using
        theano.function, but a variable may contain nodes which

            a) are named parameters in the point
            b) are *RVs with a random method

    """
    # Distribution parameters may be nodes which have named node-inputs
    # specified in the point. Need to find the node-inputs to replace them.
    givens = {}
    for param in params:
        if hasattr(param, 'name'):
            named_nodes = get_named_nodes(param)
            if param.name in named_nodes:
                named_nodes.pop(param.name)
            for name, node in named_nodes.items():
                givens[name] = (node, draw_value(node, point=point))
    values = [None for _ in params]
    for i, param in enumerate(params):
        # "Homogonise" output
        values[i] = np.atleast_1d(draw_value(param, point=point, givens=givens.values()))
    if len(values) == 1:
        return values[0]
    else:
        return values

def draw_value(param, point=None, givens={}):
    if hasattr(param, 'name'):
        if hasattr(param, 'model'):
            if point is not None and param.name in point:
                return point[param.name]
            elif hasattr(param, 'random') and param.random is not None:
                return param.random(point=point, size=None)
            else:
                return param.tag.test_value
        else:
            return function([], param,
                            givens=givens,
                            rebuild_strict=False,
                            on_unused_input='ignore',
                            allow_input_downcast=True)()
    else:
        return param

def get_sample_shape(shape, *args, **kwargs):
    """Calculate the shape of random samples from a random variable.

    Parameters
    ----------
    shape : int or tuple of int or numpy.ndarray of int
        The dimensions of the random variable.
    size : int or tuple of int or numpy.ndarray of int or None
        The number of samples to draw
    repeat : int or tuple of int or numpy.ndarray of int or None

    Returns
    -------
    An array describing the shape of the random samples.

    - If :size: is not None, then then `size` is returned.
    - If `repeat` is not None, an array representing the dimensions of 
     `repeat` *followed* by `shape` is returned. So if `repeat` is (1, 2) 
      and `shape` is (3,4) then the numpy array [1, 2, 3, 4] is returned.
    - If `size` and `repeat` are None, then an `shape` is returned.
    """
    try:
        size = args[0]
    except IndexError:
        size = kwargs.pop('size', None)
    repeat = kwargs.pop('repeat', None)
    if size is not None:
        return np.atleast_1d(size)
    elif repeat is not None:
        return np.append(np.atleast_1d(repeat), np.atleast_1d(shape))
    else:
        z = max(np.atleast_1d(shape).shape) == 0 
        return np.atleast_1d(1 if z else shape)
