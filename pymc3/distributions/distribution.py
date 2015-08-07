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
                value = point[param.name]
            elif hasattr(param, 'random') and param.random is not None:
                value = param.random(point=point, size=None)
            else:
                value = param.tag.test_value
        else:
            value = function([], param,
                             givens=givens,
                             rebuild_strict=True,
                             on_unused_input='ignore',
                             allow_input_downcast=True)()
    else:
        value = param
    # Sanitising values may be necessary.
    if hasattr(param, 'dtype'):
        value = np.atleast_1d(value).astype(param.dtype)
    if hasattr(param, 'shape'):
        try:
            shape = param.shape.tag.test_value
        except:
           shape = param.shape
        if len(shape) == 0 and len(value) == 1:
            value = value[0]
    return value


def generate_samples(generator, *args, **kwargs):
    """Generate samples from the distribution of a random variable.

    Parameters
    ----------
    generator : function
        Function to generate the random samples. The function is
        expected to follow the same behaviour as the scipy.stats
        rvs methods and the numpy.random functions. The *args
        and **kwargs (stripped of the keywords below.
    
    keyword aguments
    ~~~~~~~~~~~~~~~~

    dist_shape : int or tuple of int
        The shape of the random variable (i.e., the distribution.shape
        attribute)
    size : int or tuple of int
        The required shape of the samples. This key will be ignored
        if the parameters of the random variable are not *all* of
        shape 1.
    repeat : int or tuple of int
        While the size argument can return an arbitrary number of samples,
        this argument returns samples whose shape is multiples of the distribution
        shape, namely `np.append(repeat, dist_shape)`.
    shape_only : boolean
        Return only the shape of the samples. For debugging only.

    Any remaining *args and **kwargs are passed on to the generator function.
    """
    dist_shape = kwargs.pop('dist_shape', ())
    size = kwargs.pop('size', None)
    repeat = kwargs.pop('repeat', None)
    shape_only = kwargs.pop('shape_only', False)
    if len(dist_shape) == 0:
        dist_shape = 1
    dist_shape = np.atleast_1d(dist_shape)
    params = args + tuple(value for value in kwargs.values())
    param_lens = np.atleast_1d([len(np.atleast_1d(param)) for param in params])
    if len(params) == 0 or np.all(param_lens == 1):
        if size is not None:
            sample_shape = size
        elif repeat is not None:
            sample_shape = np.append(np.atleast_1d(repeat), dist_shape)
        else:
            sample_shape = dist_shape
        samples = generator(size=sample_shape, *args, **kwargs)
    else:
        # Have to do this due to differences between scipy
        # rvs methods with or without loc and scale parameters,
        # particularly when parameters are not all of shape (1,).
        try:
            samples = generator(size=dist_shape, *args, **kwargs)
            generator_size = dist_shape
        
        except ValueError:
            samples = generator(size=None, *args, **kwargs)
            generator_size = None
        generator_shape = samples.shape
        if repeat is not None:
            samples = np.reshape(np.array([generator(size=generator_size, *args, **kwargs) \
                                           for _ in range(int(np.prod(repeat)))]),
                          np.append(np.atleast_1d(repeat), generator_shape))
        elif generator_size is None:
            samples = generator(size=generator_size, *args, **kwargs)

    if shape_only:
        return samples.shape
    else:
        return samples
