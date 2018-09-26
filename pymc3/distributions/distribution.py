import numbers
import numpy as np
import theano.tensor as tt
from theano import function
import theano
from ..memoize import memoize
from ..model import (
    Model, modelcontext, FreeRV, ObservedRV, MultiObservedRV,
    not_shared_or_constant_variable, DependenceDAG
)
from ..vartypes import string_types
from ..util import WrapAsHashable

__all__ = ['DensityDist', 'Distribution', 'Continuous', 'Discrete',
           'NoDistribution', 'TensorType', 'draw_values', 'generate_samples']


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
                            "instantiate distributions. Add variable inside "
                            "a 'with model:' block, or use the '.dist' syntax "
                            "for a standalone distribution.")

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
        self.conditional_on = None

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
        """Magic method name for IPython to use for LaTeX formatting."""
        return None

    def logp_nojac(self, *args, **kwargs):
        """Return the logp, but do not include a jacobian term for transforms.

        If we use different parametrizations for the same distribution, we
        need to add the determinant of the jacobian of the transformation
        to make sure the densities still describe the same distribution.
        However, MAP estimates are not invariant with respect to the
        parametrization, we need to exclude the jacobian terms in this case.

        This function should be overwritten in base classes for transformed
        distributions.
        """
        return self.logp(*args, **kwargs)

    def logp_sum(self, *args, **kwargs):
        """Return the sum of the logp values for the given observations.

        Subclasses can use this to improve the speed of logp evaluations
        if only the sum of the logp values is needed.
        """
        return tt.sum(self.logp(*args, **kwargs))

    __latex__ = _repr_latex_


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
        # Do not use __getstate__ and __setstate__ from parent_dist
        # to avoid infinite recursion during unpickling
        if name.startswith('__'):
            raise AttributeError(
                "'NoDistribution' has no attribute '%s'" % name)
        if name == 'conditional_on':
            return None
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
    """Distribution based on a given log density function.

        A distribution with the passed log density function is created.
        Requires a custom random function passed as kwarg `random` to
        enable sampling.

        Example:
        --------
        .. code-block:: python
            with pm.Model():
                mu = pm.Normal('mu',0,1)
                normal_dist = pm.Normal.dist(mu, 1)
                pm.DensityDist('density_dist', normal_dist.logp, observed=np.random.randn(100), random=normal_dist.random)
                trace = pm.sample(100)

    """

    def __init__(self, logp, shape=(), dtype=None, testval=0, random=None, *args, **kwargs):
        if dtype is None:
            dtype = theano.config.floatX
        super(DensityDist, self).__init__(
            shape, dtype, testval, *args, **kwargs)
        self.logp = logp
        self.rand = random

    def random(self, *args, **kwargs):
        if self.rand is not None:
            return self.rand(*args, **kwargs)
        else:
            raise ValueError("Distribution was not passed any random method "
                            "Define a custom random method and pass it as kwarg random")


def draw_values(params, point=None, size=None, model=None):
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
    # Get the nodes that we must draw from as list of lists
    try:
        model = modelcontext(model)
        dependence_dag, index = (
                model.variable_dependence_dag.get_sub_dag(params,
                                                          return_index=True))
    except Exception:
        dependence_dag = DependenceDAG()
        dependence_dag, index = dependence_dag.get_sub_dag(params,
                                                           return_index=True)
    layers = dependence_dag.get_nodes_in_depth_layers()

    # Init drawn values and updatable point and givens
    drawn = {n: None for n in dependence_dag}
    givens = []
    if point is None:
        point = {}
    else:
        point = point.copy()
    nodes_missing_inputs = {}

    for depth, layer in enumerate(layers):
        while True:
            try:
                # Pop nodes from layer because we may use stack new nodes
                # onto the layer list in the middle of the loop
                node = layer.pop(0)
            except Exception:
                break

            # We may have flaged this node to be computed by compiling the
            # theano function, because deterministic relations take precedence
            # over conditional relations
            if isinstance(node, tuple):
                node, is_determined = node
            else:
                is_determined = False

            # Node's value had already been determined so we jump onto the next
            if drawn[node] is not None:
                continue
            try:
                if is_determined:
                    node_value = _compute_value(node,
                                                givens=givens,
                                                size=size)
                else:
                    node_value = _draw_value(node,
                                             point=point,
                                             givens=givens,
                                             size=size)
                drawn[node] = node_value
            except theano.gof.fg.MissingInputError as e:
                # Expected to fail for auto-transformed RVs whos values were
                # not provided in point
                nodes_missing_inputs[node] = e
                continue

            # If the node is a theano Variable, which is not TensorConstant,
            # nor SharedVariable, we must add its value to point (only if it
            # has conditional children) and givens (only if it has
            # deterministic children)
            if not_shared_or_constant_variable(node):
                if dependence_dag.conditional_children[node]:
                    point[node.name] = node_value
                if dependence_dag.deterministic_children:
                    givens.append((node, node_value))

            # If the node has deterministic children, check if the children's
            # values can be computed. This must be done because deterministic
            # relations must take precedence over conditional relations
            # amongst variables.
            for child in dependence_dag.deterministic_children[node]:
                # Check if all of the child's deterministic parents have their
                # values set, allowing us to compute the child's value.
                if not any([drawn[p] is None for p in
                            dependence_dag.deterministic_parents[child]]):
                    # Append child to the current layer's node stack, to
                    # compute its value ahead of its scheduled depth
                    layer.append((child, True))

    # Now that the DAG has been transversed, drawing values, we can place them
    # in a list following the indexing given by the input list `params`
    output = []
    for ind in range(len(params)):
        node = index[ind]
        value = drawn[node]
        if value is None:
            # We failed to draw the params[i] value. This could be due to an
            # ignored MissingInputError, in which case we reraise it, or it
            # could be some other form of unexpected RuntimeError.
            if node in nodes_missing_inputs:
                raise nodes_missing_inputs[node]
            else:
                raise RuntimeError('Failed to draw value for parameter {}'.
                                   format(params[ind]))
        output.append(value)
    return output


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


def _draw_value(param, point=None, givens=None, size=None):
    """Draw a random value from a distribution or return a constant.

    Parameters
    ----------
    param : WrapAsHashable, theano variable or pymc3 random variable
        The value or distribution. Constants or shared variables
        will be converted to an array and returned. Theano variables
        are evaluated. If `param` is a pymc3 random variables, draw
        a new value from it and return that, unless a value is specified
        in `point`.
    point : dict, optional
        A dictionary from pymc3 variable names to their values.
    givens : list, optional
        A list of tuples of theano variables and their values. These values
        are used to evaluate `param` if it is a theano variable.
    size : int, optional
        Number of samples
    """
    if isinstance(param, numbers.Number):
        output = param
    elif isinstance(param, np.ndarray):
        output = param
    elif isinstance(param, theano.tensor.TensorConstant):
        output = param.value
    elif isinstance(param, theano.tensor.sharedvar.SharedVariable):
        output = param.get_value()
    elif isinstance(param, WrapAsHashable):
        output = param.get_value()
    elif isinstance(param, tt.TensorVariable):
        if point and hasattr(param, 'model') and param.name in point:
            output = point[param.name]
        elif hasattr(param, 'random') and param.random is not None:
            output = param.random(point=point, size=size)
        elif (hasattr(param, 'distribution') and
                hasattr(param.distribution, 'random') and
                param.distribution.random is not None):
            if hasattr(param, 'observations'):
                # shape inspection for ObservedRV
                dist_tmp = param.distribution
                try:
                    distshape = param.observations.shape.eval()
                except AttributeError:
                    distshape = param.observations.shape

                dist_tmp.shape = distshape
                try:
                    dist_tmp.random(point=point, size=size)
                except (ValueError, TypeError):
                    # reset shape to account for shape changes
                    # with theano.shared inputs
                    dist_tmp.shape = np.array([])
                    val = np.atleast_1d(dist_tmp.random(point=point,
                                                        size=None))
                    # Sometimes point may change the size of val but not the
                    # distribution's shape
                    if point and size is not None:
                        temp_size = np.atleast_1d(size)
                        if all(val.shape[:len(temp_size)] == temp_size):
                            dist_tmp.shape = val.shape[len(temp_size):]
                        else:
                            dist_tmp.shape = val.shape
                output = dist_tmp.random(point=point, size=size)
            else:
                output = param.distribution.random(point=point, size=size)
        else:
            output = _compute_value(param, givens=givens, size=size)
    else:
        raise ValueError('Unexpected type in draw_value: %s' % type(param))
    # Maybe at this point we should control the output shape depending on size?
    return output


def _compute_value(param, givens=None, size=None):
    if givens:
        variables, values = list(zip(*givens))
    else:
        variables = values = []
    func = _compile_theano_function(param, variables)
    if size is not None:
        size = np.atleast_1d(size)
    if (values and all((hasattr(var, 'dshape') for var in variables)) and
        not all(var.dshape == getattr(val, 'shape', tuple())
                for var, val in zip(variables, values))):
        output = np.array([func(*v) for v in zip(*values)])
    elif (size is not None and any((val.ndim > var.ndim)
          for var, val in zip(variables, values))):
        output = np.array([func(*v) for v in zip(*values)])
    else:
        output = func(*values)
    return output


def to_tuple(shape):
    """Convert ints, arrays, and Nones to tuples"""
    try:
        shape = tuple(shape or ())
    except TypeError:  # If size is an int
        shape = tuple((shape,))
    except ValueError:  # If size is np.array
        shape = tuple(shape)
    return shape

def _is_one_d(dist_shape):
    if hasattr(dist_shape, 'dshape') and dist_shape.dshape in ((), (0,), (1,)):
        return True
    elif hasattr(dist_shape, 'shape') and dist_shape.shape in ((), (0,), (1,)):
        return True
    elif dist_shape == ():
        return True
    return False

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
    one_d = _is_one_d(dist_shape)
    size = kwargs.pop('size', None)
    broadcast_shape = kwargs.pop('broadcast_shape', None)
    if size is None:
        size = 1

    args = tuple(p[0] if isinstance(p, tuple) else p for p in args)

    for key in kwargs:
        p = kwargs[key]
        kwargs[key] = p[0] if isinstance(p, tuple) else p

    if broadcast_shape is None:
        inputs = args + tuple(kwargs.values())
        try:
            broadcast_shape = np.broadcast(*inputs).shape  # size of generator(size=1)
        except ValueError:  # sometimes happens if args have shape (500,) and (500, 4)
            max_dims = max(j.ndim for j in args + tuple(kwargs.values()))
            args = tuple([j.reshape(j.shape + (1,) * (max_dims - j.ndim)) for j in args])
            kwargs = {k: v.reshape(v.shape + (1,) * (max_dims - v.ndim)) for k, v in kwargs.items()}
            inputs = args + tuple(kwargs.values())
            broadcast_shape = np.broadcast(*inputs).shape  # size of generator(size=1)

    dist_shape = to_tuple(dist_shape)
    broadcast_shape = to_tuple(broadcast_shape)
    size_tup = to_tuple(size)

    # All inputs are scalars, end up size (size_tup, dist_shape)
    if broadcast_shape in {(), (0,), (1,)}:
        samples = generator(size=size_tup + dist_shape, *args, **kwargs)
    # Inputs already have the right shape. Just get the right size.
    elif broadcast_shape[-len(dist_shape):] == dist_shape or len(dist_shape) == 0:
        if size == 1 or (broadcast_shape == size_tup + dist_shape):
            samples = generator(size=broadcast_shape, *args, **kwargs)
        elif dist_shape == broadcast_shape:
            samples = generator(size=size_tup + dist_shape, *args, **kwargs)
        else:
            samples = None
    # Args have been broadcast correctly, can just ask for the right shape out
    elif dist_shape[-len(broadcast_shape):] == broadcast_shape:
        samples = generator(size=size_tup + dist_shape, *args, **kwargs)
    # Inputs have the right size, have to manually broadcast to the right dist_shape
    elif broadcast_shape[:len(size_tup)] == size_tup:
        suffix = broadcast_shape[len(size_tup):] + dist_shape
        samples = [generator(*args, **kwargs).reshape(size_tup + (1,)) for _ in range(np.prod(suffix, dtype=int))]
        samples = np.hstack(samples).reshape(size_tup + suffix)
    else:
        samples = None

    if samples is None:
        raise TypeError('''Attempted to generate values with incompatible shapes:
            size: {size}
            dist_shape: {dist_shape}
            broadcast_shape: {broadcast_shape}
        '''.format(size=size, dist_shape=dist_shape, broadcast_shape=broadcast_shape))

    # reshape samples here
    if samples.shape[0] == 1 and size == 1:
        if len(samples.shape) > len(dist_shape) and samples.shape[-len(dist_shape):] == dist_shape:
            samples = samples.reshape(samples.shape[1:])

    if one_d and samples.shape[-1] == 1:
        samples = samples.reshape(samples.shape[:-1])
    return np.asarray(samples)
