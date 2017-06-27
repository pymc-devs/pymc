import numbers
import numpy as np
import theano.tensor as tt
from theano import function
import theano
from ..memoize import memoize
from ..model import Model, get_named_nodes, FreeRV, ObservedRV
from ..vartypes import string_types
from .dist_math import bound
# To avoid circular import for transform below
import pymc3 as pm

__all__ = ['DensityDist', 'Distribution', 'Continuous', 'Bound',
           'Discrete', 'NoDistribution', 'TensorType', 'draw_values']


class _Unpickling(object):
    pass


class Distribution(object):
    """Statistical distribution"""
    def __new__(cls, name, *args, **kwargs):
        if name is _Unpickling:
            return object.__new__(cls)  # for pickle
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to "
                            "use the Normal('x', 0,1) syntax. "
                            "Add a 'with model:' block")

        if isinstance(name, string_types):
            data = kwargs.pop('observed', None)
            if isinstance(data, ObservedRV) or isinstance(data, FreeRV):
                raise TypeError("observed needs to be data but got: {}".format(type(data)))
            total_size = kwargs.pop('total_size', None)
            dist = cls.dist(*args, **kwargs)
            return model.Var(name, dist, data, total_size)
        else:
            raise TypeError("Name needs to be a string but got: {}".format(name))

    def __getnewargs__(self):
        return _Unpickling,

    @classmethod
    def dist(cls, *args, **kwargs):
        dist = object.__new__(cls)
        dist.__init__(*args, **kwargs)
        return dist

    def __init__(self, shape, dtype, testval=None, defaults=(),
                 transform=None, broadcastable=None):
        self.shape = np.atleast_1d(shape)
        if False in (np.floor(self.shape) == self.shape):
            raise TypeError("Expected int elements in shape")
        self.dtype = dtype
        self.type = TensorType(self.dtype, self.shape, broadcastable)
        self.testval = testval
        self.defaults = defaults
        self.transform = transform

    def default(self):
        return np.asarray(self.get_test_val(self.testval, self.defaults), self.dtype)

    def get_test_val(self, val, defaults):
        if val is None:
            for v in defaults:
                if hasattr(self, v) and np.all(np.isfinite(self.getattr_value(v))):
                    return self.getattr_value(v)
        else:
            return self.getattr_value(val)

        if val is None:
            raise AttributeError("%s has no finite default value to use, "
                                 "checked: %s. Pass testval argument or "
                                 "adjust so value is finite."
                                 % (self, str(defaults)))

    def getattr_value(self, val):
        if isinstance(val, string_types):
            val = getattr(self, val)

        if isinstance(val, tt.TensorVariable):
            return val.tag.test_value

        if isinstance(val, tt.TensorConstant):
            return val.value

        return val

    def _repr_latex_(self, name=None, dist=None):
        return None


def TensorType(dtype, shape, broadcastable=None):
    if broadcastable is None:
        broadcastable = np.atleast_1d(shape) == 1
    return tt.TensorType(str(dtype), broadcastable)


class NoDistribution(Distribution):

    def __init__(self, shape, dtype, testval=None, defaults=(),
                 transform=None, parent_dist=None, *args, **kwargs):
        super(NoDistribution, self).__init__(shape=shape, dtype=dtype,
                                             testval=testval, defaults=defaults,
                                             *args, **kwargs)
        self.parent_dist = parent_dist

    def __getattr__(self, name):
        try:
            self.__dict__[name]
        except KeyError:
            return getattr(self.parent_dist, name)

    def logp(self, x):
        return 0


class Discrete(Distribution):
    """Base class for discrete distributions"""

    def __init__(self, shape=(), dtype=None, defaults=('mode',),
                 *args, **kwargs):
        if dtype is None:
            if theano.config.floatX == 'float32':
                dtype = 'int16'
            else:
                dtype = 'int64'
        if dtype != 'int16' and dtype != 'int64':
            raise TypeError('Discrete classes expect dtype to be int16 or int64.')

        if kwargs.get('transform', None) is not None:
            raise ValueError("Transformations for discrete distributions "
                             "are not allowed.")

        super(Discrete, self).__init__(
            shape, dtype, defaults=defaults, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""

    def __init__(self, shape=(), dtype=None, defaults=('median', 'mean', 'mode'),
                 *args, **kwargs):
        if dtype is None:
            dtype = theano.config.floatX
        super(Continuous, self).__init__(
            shape, dtype, defaults=defaults, *args, **kwargs)


class DensityDist(Distribution):
    """Distribution based on a given log density function."""

    def __init__(self, logp, shape=(), dtype=None, testval=0, *args, **kwargs):
        if dtype is None:
            dtype = theano.config.floatX
        super(DensityDist, self).__init__(
            shape, dtype, testval, *args, **kwargs)
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
                if not isinstance(node, (tt.sharedvar.SharedVariable,
                                         tt.TensorConstant)):
                    givens[name] = (node, _draw_value(node, point=point))
    values = []
    for param in params:
        values.append(_draw_value(param, point=point, givens=givens.values()))
    return values


@memoize
def _compile_theano_function(param, vars, givens=None):
    """Compile theano function for a given parameter and input variables.

    This function is memoized to avoid repeating costly theano compilations
    when repeatedly drawing values, which is done when generating posterior
    predictive samples.

    Parameters
    ----------
    param : Model variable from which to draw value
    vars : Children variables of `param`
    givens : Variables to be replaced in the Theano graph

    Returns
    -------
    A compiled theano function that takes the values of `vars` as input
        positional args
    """
    return function(vars, param, givens=givens,
                    rebuild_strict=True,
                    on_unused_input='ignore',
                    allow_input_downcast=True)


def _draw_value(param, point=None, givens=None):
    """Draw a random value from a distribution or return a constant.

    Parameters
    ----------
    param : number, array like, theano variable or pymc3 random variable
        The value or distribution. Constants or shared variables
        will be converted to an array and returned. Theano variables
        are evaluated. If `param` is a pymc3 random variables, draw
        a new value from it and return that, unless a value is specified
        in `point`.
    point : dict, optional
        A dictionary from pymc3 variable names to their values.
    givens : dict, optional
        A dictionary from theano variables to their values. These values
        are used to evaluate `param` if it is a theano variable.
    """
    if isinstance(param, numbers.Number):
        return param
    elif isinstance(param, np.ndarray):
        return param
    elif isinstance(param, tt.TensorConstant):
        return param.value
    elif isinstance(param, tt.sharedvar.SharedVariable):
        return param.get_value()
    elif isinstance(param, tt.TensorVariable):
        if point and hasattr(param, 'model') and param.name in point:
            return point[param.name]
        elif hasattr(param, 'random') and param.random is not None:
            return param.random(point=point, size=None)
        else:
            if givens:
                variables, values = list(zip(*givens))
            else:
                variables = values = []
            func = _compile_theano_function(param, variables)
            return func(*values)
    else:
        raise ValueError('Unexpected type in draw_value: %s' % type(param))


def broadcast_shapes(*args):
    """Return the shape resulting from broadcasting multiple shapes.
    Represents numpy's broadcasting rules.

    Parameters
    ----------
    *args : array-like of int
        Tuples or arrays or lists representing the shapes of arrays to be broadcast.

    Returns
    -------
    Resulting shape or None if broadcasting is not possible.
    """
    x = list(np.atleast_1d(args[0])) if args else ()
    for arg in args[1:]:
        y = list(np.atleast_1d(arg))
        if len(x) < len(y):
            x, y = y, x
        x[-len(y):] = [j if i == 1 else i if j == 1 else i if i == j else 0
                       for i, j in zip(x[-len(y):], y)]
        if not all(x):
            return None
    return tuple(x)


def infer_shape(shape):
    try:
        shape = tuple(shape or ())
    except TypeError:  # If size is an int
        shape = tuple((shape,))
    except ValueError:  # If size is np.array
        shape = tuple(shape)
    return shape


def reshape_sampled(sampled, size, dist_shape):
    dist_shape = infer_shape(dist_shape)
    repeat_shape = infer_shape(size)

    if np.size(sampled) == 1 or repeat_shape or dist_shape:
        return np.reshape(sampled, repeat_shape + dist_shape)
    else:
        return sampled


def replicate_samples(generator, size, repeats, *args, **kwargs):
    n = int(np.prod(repeats))
    if n == 1:
        samples = generator(size=size, *args, **kwargs)
    else:
        samples = np.array([generator(size=size, *args, **kwargs)
                            for _ in range(n)])
        samples = np.reshape(samples, tuple(repeats) + tuple(size))
    return samples


def generate_samples(generator, *args, **kwargs):
    """Generate samples from the distribution of a random variable.

    Parameters
    ----------
    generator : function
        Function to generate the random samples. The function is
        expected take parameters for generating samples and
        a keyword argument `size` which determines the shape
        of the samples.
        The *args and **kwargs (stripped of the keywords below) will be
        passed to the generator function.

    keyword arguments
    ~~~~~~~~~~~~~~~~

    dist_shape : int or tuple of int
        The shape of the random variable (i.e., the shape attribute).
    size : int or tuple of int
        The required shape of the samples.
    broadcast_shape: tuple of int or None
        The shape resulting from the broadcasting of the parameters.
        If not specified it will be inferred from the shape of the
        parameters. This may be required when the parameter shape
        does not determine the shape of a single sample, for example,
        the shape of the probabilities in the Categorical distribution.

    Any remaining *args and **kwargs are passed on to the generator function.
    """
    dist_shape = kwargs.pop('dist_shape', ())
    size = kwargs.pop('size', None)
    broadcast_shape = kwargs.pop('broadcast_shape', None)
    params = args + tuple(kwargs.values())

    if broadcast_shape is None:
        broadcast_shape = broadcast_shapes(*[np.atleast_1d(p).shape for p in params
                                             if not isinstance(p, tuple)])
    if broadcast_shape == ():
        broadcast_shape = (1,)

    args = tuple(p[0] if isinstance(p, tuple) else p for p in args)
    for key in kwargs:
        p = kwargs[key]
        kwargs[key] = p[0] if isinstance(p, tuple) else p

    if np.all(dist_shape[-len(broadcast_shape):] == broadcast_shape):
        prefix_shape = tuple(dist_shape[:-len(broadcast_shape)])
    else:
        prefix_shape = tuple(dist_shape)

    repeat_shape = infer_shape(size)

    if broadcast_shape == (1,) and prefix_shape == ():
        if size is not None:
            samples = generator(size=size, *args, **kwargs)
        else:
            samples = generator(size=1, *args, **kwargs)
    else:
        if size is not None:
            samples = replicate_samples(generator,
                                        broadcast_shape,
                                        repeat_shape + prefix_shape,
                                        *args, **kwargs)
        else:
            samples = replicate_samples(generator,
                                        broadcast_shape,
                                        prefix_shape,
                                        *args, **kwargs)
    return reshape_sampled(samples, size, dist_shape)


class Bounded(Distribution):
    R"""
    An upper, lower or upper+lower bounded distribution

    Parameters
    ----------
    distribution : pymc3 distribution
        Distribution to be transformed into a bounded distribution
    lower : float (optional)
        Lower bound of the distribution, set to -inf to disable.
    upper : float (optional)
        Upper bound of the distribibution, set to inf to disable.
    tranform : 'infer' or object
        If 'infer', infers the right transform to apply from the supplied bounds.
        If transform object, has to supply .forward() and .backward() methods.
        See pymc3.distributions.transforms for more information.
    """

    def __init__(self, distribution, lower, upper, transform='infer', *args, **kwargs):
        self.dist = distribution.dist(*args, **kwargs)

        self.__dict__.update(self.dist.__dict__)
        self.__dict__.update(locals())

        if hasattr(self.dist, 'mode'):
            self.mode = self.dist.mode

        if transform == 'infer':

            default = self.dist.default()

            if not np.isinf(lower) and not np.isinf(upper):
                self.transform = pm.distributions.transforms.interval(lower, upper)
                if default <= lower or default >= upper:
                    self.testval = 0.5 * (upper + lower)

            if not np.isinf(lower) and np.isinf(upper):
                self.transform = pm.distributions.transforms.lowerbound(lower)
                if default <= lower:
                    self.testval = lower + 1

            if np.isinf(lower) and not np.isinf(upper):
                self.transform = pm.distributions.transforms.upperbound(upper)
                if default >= upper:
                    self.testval = upper - 1

        if issubclass(distribution, Discrete):
            self.transform = None

    def _random(self, lower, upper, point=None, size=None):
        samples = np.zeros(size).flatten()
        i, n = 0, len(samples)
        while i < len(samples):
            sample = self.dist.random(point=point, size=n)
            select = sample[np.logical_and(sample > lower, sample <= upper)]
            samples[i:(i + len(select))] = select[:]
            i += len(select)
            n -= len(select)
        if size is not None:
            return np.reshape(samples, size)
        else:
            return samples

    def random(self, point=None, size=None, repeat=None):
        lower, upper = draw_values([self.lower, self.upper], point=point)
        return generate_samples(self._random, lower, upper, point,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        return bound(self.dist.logp(value),
                     value >= self.lower, value <= self.upper)


class Bound(object):
    R"""
    Creates a new upper, lower or upper+lower bounded distribution

    Parameters
    ----------
    distribution : pymc3 distribution
        Distribution to be transformed into a bounded distribution
    lower : float (optional)
        Lower bound of the distribution
    upper : float (optional)

    Example
    -------
    # Bounded distribution can be defined before the model context
    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
    with pm.Model():
        par1 = PositiveNormal('par1', mu=0.0, sd=1.0, testval=1.0)
        # or within the model context
        NegativeNormal = pm.Bound(pm.Normal, upper=0.0)
        par2 = NegativeNormal('par2', mu=0.0, sd=1.0, testval=1.0)

        # or you can define it implicitly within the model context
        par3 = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)(
                'par3', mu=0.0, sd=1.0, testval=1.0)
    """

    def __init__(self, distribution, lower=-np.inf, upper=np.inf):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, *args, **kwargs):
        if 'observed' in kwargs:
            raise ValueError('Observed Bound distributions are not allowed. '
                             'If you want to model truncated data '
                             'you can use a pm.Potential in combination '
                             'with the cumulative probability function. See '
                             'pymc3/examples/censored_data.py for an example.')
        first, args = args[0], args[1:]

        return Bounded(first, self.distribution, self.lower, self.upper,
                       *args, **kwargs)

    def dist(self, *args, **kwargs):
        return Bounded.dist(self.distribution, self.lower, self.upper,
                            *args, **kwargs)
