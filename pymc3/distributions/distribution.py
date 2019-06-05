import numbers

import numpy as np
import theano.tensor as tt
from theano import function
import theano
from ..memoize import memoize
from ..model import (
    Model, get_named_nodes_and_relations, FreeRV,
    ObservedRV, MultiObservedRV, Context, InitContextMeta
)
from ..vartypes import string_types
from .shape_utils import (
    to_tuple,
    get_broadcastable_dist_samples,
    broadcast_dist_samples_shape,
)

__all__ = ['DensityDist', 'Distribution', 'Continuous', 'Discrete',
           'NoDistribution', 'TensorType', 'draw_values', 'generate_samples']


class _Unpickling:
    pass


class Distribution:
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
        super().__init__(shape=shape, dtype=dtype,
                         testval=testval, defaults=defaults,
                         *args, **kwargs)
        self.parent_dist = parent_dist

    def __getattr__(self, name):
        # Do not use __getstate__ and __setstate__ from parent_dist
        # to avoid infinite recursion during unpickling
        if name.startswith('__'):
            raise AttributeError(
                "'NoDistribution' has no attribute '%s'" % name)
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

        super().__init__(shape, dtype, defaults=defaults, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""

    def __init__(self, shape=(), dtype=None, defaults=('median', 'mean', 'mode'),
                 *args, **kwargs):
        if dtype is None:
            dtype = theano.config.floatX
        super().__init__(shape, dtype, defaults=defaults, *args, **kwargs)


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
        super().__init__(shape, dtype, testval, *args, **kwargs)
        self.logp = logp
        self.rand = random

    def random(self, *args, **kwargs):
        if self.rand is not None:
            return self.rand(*args, **kwargs)
        else:
            raise ValueError("Distribution was not passed any random method "
                            "Define a custom random method and pass it as kwarg random")


class _DrawValuesContext(Context, metaclass=InitContextMeta):
    """ A context manager class used while drawing values with draw_values
    """

    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if cls.get_contexts():
            potential_parent = cls.get_contexts()[-1]
            # We have to make sure that the context is a _DrawValuesContext
            # and not a Model
            if isinstance(potential_parent, _DrawValuesContext):
                instance._parent = potential_parent
            else:
                instance._parent = None
        else:
            instance._parent = None
        return instance

    def __init__(self):
        if self.parent is not None:
            # All _DrawValuesContext instances that are in the context of
            # another _DrawValuesContext will share the reference to the
            # drawn_vars dictionary. This means that separate branches
            # in the nested _DrawValuesContext context tree will see the
            # same drawn values.
            # The drawn_vars keys shall be (RV, size) tuples
            self.drawn_vars = self.parent.drawn_vars
        else:
            self.drawn_vars = dict()

    @property
    def parent(self):
        return self._parent


class _DrawValuesContextBlocker(_DrawValuesContext, metaclass=InitContextMeta):
    """
    Context manager that starts a new drawn variables context disregarding all
    parent contexts. This can be used inside a random method to ensure that
    the drawn values wont be the ones cached by previous calls
    """
    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        instance._parent = None
        return instance

    def __init__(self):
        self.drawn_vars = dict()


def is_fast_drawable(var):
    return isinstance(var, (numbers.Number,
                            np.ndarray,
                            tt.TensorConstant,
                            tt.sharedvar.SharedVariable))


def draw_values(params, point=None, size=None):
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
    # Get fast drawable values (i.e. things in point or numbers, arrays,
    # constants or shares, or things that were already drawn in related
    # contexts)
    if point is None:
        point = {}
    with _DrawValuesContext() as context:
        params = dict(enumerate(params))
        drawn = context.drawn_vars
        evaluated = {}
        symbolic_params = []
        for i, p in params.items():
            # If the param is fast drawable, then draw the value immediately
            if is_fast_drawable(p):
                v = _draw_value(p, point=point, size=size)
                evaluated[i] = v
                continue

            name = getattr(p, 'name', None)
            if (p, size) in drawn:
                # param was drawn in related contexts
                v = drawn[(p, size)]
                evaluated[i] = v
            # We filter out Deterministics by checking for `model` attribute
            elif name is not None and hasattr(p, 'model') and name in point:
                # param.name is in point
                v = point[name]
                evaluated[i] = drawn[(p, size)] = v
            else:
                # param still needs to be drawn
                symbolic_params.append((i, p))

        if not symbolic_params:
            # We only need to enforce the correct order if there are symbolic
            # params that could be drawn in variable order
            return [evaluated[i] for i in params]

        # Distribution parameters may be nodes which have named node-inputs
        # specified in the point. Need to find the node-inputs, their
        # parents and children to replace them.
        leaf_nodes = {}
        named_nodes_parents = {}
        named_nodes_children = {}
        for _, param in symbolic_params:
            if hasattr(param, 'name'):
                # Get the named nodes under the `param` node
                nn, nnp, nnc = get_named_nodes_and_relations(param)
                leaf_nodes.update(nn)
                # Update the discovered parental relationships
                for k in nnp.keys():
                    if k not in named_nodes_parents.keys():
                        named_nodes_parents[k] = nnp[k]
                    else:
                        named_nodes_parents[k].update(nnp[k])
                # Update the discovered child relationships
                for k in nnc.keys():
                    if k not in named_nodes_children.keys():
                        named_nodes_children[k] = nnc[k]
                    else:
                        named_nodes_children[k].update(nnc[k])

        # Init givens and the stack of nodes to try to `_draw_value` from
        givens = {p.name: (p, v) for (p, size), v in drawn.items()
                  if getattr(p, 'name', None) is not None}
        stack = list(leaf_nodes.values())  # A queue would be more appropriate
        while stack:
            next_ = stack.pop(0)
            if (next_, size) in drawn:
                # If the node already has a givens value, skip it
                continue
            elif isinstance(next_, (tt.TensorConstant,
                                    tt.sharedvar.SharedVariable)):
                # If the node is a theano.tensor.TensorConstant or a
                # theano.tensor.sharedvar.SharedVariable, its value will be
                # available automatically in _compile_theano_function so
                # we can skip it. Furthermore, if this node was treated as a
                # TensorVariable that should be compiled by theano in
                # _compile_theano_function, it would raise a `TypeError:
                # ('Constants not allowed in param list', ...)` for
                # TensorConstant, and a `TypeError: Cannot use a shared
                # variable (...) as explicit input` for SharedVariable.
                # ObservedRV and MultiObservedRV instances are ViewOPs
                # of TensorConstants or SharedVariables, we must add them
                # to the stack or risk evaluating deterministics with the
                # wrong values (issue #3354)
                stack.extend([node for node in named_nodes_parents[next_]
                              if isinstance(node, (ObservedRV,
                                                   MultiObservedRV))
                              and (node, size) not in drawn])
                continue
            else:
                # If the node does not have a givens value, try to draw it.
                # The named node's children givens values must also be taken
                # into account.
                children = named_nodes_children[next_]
                temp_givens = [givens[k] for k in givens if k in children]
                try:
                    # This may fail for autotransformed RVs, which don't
                    # have the random method
                    value = _draw_value(next_,
                                        point=point,
                                        givens=temp_givens,
                                        size=size)
                    givens[next_.name] = (next_, value)
                    drawn[(next_, size)] = value
                except theano.gof.fg.MissingInputError:
                    # The node failed, so we must add the node's parents to
                    # the stack of nodes to try to draw from. We exclude the
                    # nodes in the `params` list.
                    stack.extend([node for node in named_nodes_parents[next_]
                                  if node is not None and
                                  (node, size) not in drawn])

        # the below makes sure the graph is evaluated in order
        # test_distributions_random::TestDrawValues::test_draw_order fails without it
        # The remaining params that must be drawn are all hashable
        to_eval = set()
        missing_inputs = set([j for j, p in symbolic_params])
        while to_eval or missing_inputs:
            if to_eval == missing_inputs:
                raise ValueError('Cannot resolve inputs for {}'.format([str(params[j]) for j in to_eval]))
            to_eval = set(missing_inputs)
            missing_inputs = set()
            for param_idx in to_eval:
                param = params[param_idx]
                if (param, size) in drawn:
                    evaluated[param_idx] = drawn[(param, size)]
                else:
                    try:  # might evaluate in a bad order,
                        # Sometimes _draw_value recurrently calls draw_values.
                        # This may set values for certain nodes in the drawn
                        # dictionary, but they don't get added to the givens
                        # dictionary. Here, we try to fix that.
                        if param in named_nodes_children:
                            for node in named_nodes_children[param]:
                                if (
                                    node.name not in givens and
                                    (node, size) in drawn
                                ):
                                    givens[node.name] = (
                                        node,
                                        drawn[(node, size)]
                                    )
                        value = _draw_value(param,
                                            point=point,
                                            givens=givens.values(),
                                            size=size)
                        evaluated[param_idx] = drawn[(param, size)] = value
                        givens[param.name] = (param, value)
                    except theano.gof.fg.MissingInputError:
                        missing_inputs.add(param_idx)

    return [evaluated[j] for j in params] # set the order back


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
    f = function(
        vars,
        param,
        givens=givens,
        rebuild_strict=True,
        on_unused_input="ignore",
        allow_input_downcast=True,
    )
    return vectorize_theano_function(f, inputs=vars, output=param)


def vectorize_theano_function(f, inputs, output):
    """Takes a compiled theano function and wraps it with a vectorized version.
    Theano compiled functions expect inputs and outputs of a fixed number of
    dimensions. In our context, these usually come from deterministics which
    are compiled against a given RV, with its core shape. If we draw i.i.d.
    samples from said RV, we would not be able to compute the deterministic
    over the i.i.d sampled dimensions (i.e. those that are not the core
    dimensions of the RV). To deal with this problem, we wrap the theano
    compiled function with numpy.vectorize, providing the correct signature
    for the core dimensions. The extra dimensions, will be interpreted as
    i.i.d. sampled axis and will be broadcast following the usual rules.

    Parameters
    ----------
    f : theano compiled function
    inputs : list of theano variables used as inputs for the function
    givens : theano variable which is the output of the function

    Notes
    -----
    If inputs is an empty list (theano function with no inputs needed), then
    the same `f` is returned.
    Only functions that return a single theano variable's value can be
    vectorized.

    Returns
    -------
    A function which wraps `f` with numpy.vectorize with the apropriate call
    signature.
    """
    inputs_signatures = ",".join(
        [
            get_vectorize_signature(var, var_name="i_{}".format(input_ind))
            for input_ind, var in enumerate(inputs)
        ]
    )
    if len(inputs_signatures) > 0:
        output_signature = get_vectorize_signature(output, var_name="o")
        signature = inputs_signatures + "->" + output_signature

        return np.vectorize(f, signature=signature)
    else:
        return f


def get_vectorize_signature(var, var_name="i"):
    if var.ndim == 0:
        return "()"
    else:
        sig = ",".join(
            ["{}_{}".format(var_name, axis_ind) for axis_ind in range(var.ndim)]
        )
        return "({})".format(sig)


def _draw_value(param, point=None, givens=None, size=None):
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
    size : int, optional
        Number of samples
    """
    if isinstance(param, (numbers.Number, np.ndarray)):
        return param
    elif isinstance(param, tt.TensorConstant):
        return param.value
    elif isinstance(param, tt.sharedvar.SharedVariable):
        return param.get_value()
    elif isinstance(param, (tt.TensorVariable, MultiObservedRV)):
        if point and hasattr(param, 'model') and param.name in point:
            return point[param.name]
        elif hasattr(param, 'random') and param.random is not None:
            return param.random(point=point, size=size)
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
                    return dist_tmp.random(point=point, size=size)
                except (ValueError, TypeError):
                    # reset shape to account for shape changes
                    # with theano.shared inputs
                    dist_tmp.shape = np.array([])
                    # We want to draw values to infer the dist_shape,
                    # we don't want to store these drawn values to the context
                    with _DrawValuesContextBlocker():
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
                return dist_tmp.random(point=point, size=size)
            else:
                return param.distribution.random(point=point, size=size)
        else:
            if givens:
                variables, values = list(zip(*givens))
            else:
                variables = values = []
            # We only truly care if the ancestors of param that were given
            # value have the matching dshape and val.shape
            param_ancestors = \
                set(theano.gof.graph.ancestors([param],
                                               blockers=list(variables))
                    )
            inputs = [(var, val) for var, val in
                      zip(variables, values)
                      if var in param_ancestors]
            if inputs:
                input_vars, input_vals = list(zip(*inputs))
            else:
                input_vars = []
                input_vals = []
            func = _compile_theano_function(param, input_vars)
            output = func(*input_vals)
            return output
    raise ValueError('Unexpected type in draw_value: %s' % type(param))

def _is_one_d(dist_shape):
    if hasattr(dist_shape, 'dshape') and dist_shape.dshape in ((), (0,), (1,)):
        return True
    elif hasattr(dist_shape, 'shape') and dist_shape.shape in ((), (0,), (1,)):
        return True
    elif to_tuple(dist_shape) == ():
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
    not_broadcast_kwargs: dict or None
        Key word argument dictionary to provide to the random generator, which
        must not be broadcasted with the rest of the *args and **kwargs.

    Any remaining *args and **kwargs are passed on to the generator function.
    """
    dist_shape = kwargs.pop('dist_shape', ())
    one_d = _is_one_d(dist_shape)
    size = kwargs.pop('size', None)
    broadcast_shape = kwargs.pop('broadcast_shape', None)
    not_broadcast_kwargs = kwargs.pop('not_broadcast_kwargs', None)
    if not_broadcast_kwargs is None:
        not_broadcast_kwargs = dict()

    # Parse out raw input parameters for the generator
    args = tuple(p[0] if isinstance(p, tuple) else p for p in args)
    for key in kwargs:
        p = kwargs[key]
        kwargs[key] = p[0] if isinstance(p, tuple) else p

    # Convert size and dist_shape to tuples
    size_tup = to_tuple(size)
    dist_shape = to_tuple(dist_shape)
    if dist_shape[:len(size_tup)] == size_tup:
        # dist_shape is prepended with size_tup. This is not a consequence
        # of the parameters being drawn size_tup times! By chance, the
        # distribution's shape has its first elements equal to size_tup.
        # This means that we must prepend the size_tup to dist_shape, and
        # check if that broadcasts well with the parameters
        _dist_shape = size_tup + dist_shape
    else:
        _dist_shape = dist_shape


    if broadcast_shape is None:
        # If broadcast_shape is not explicitly provided, it is inferred as the
        # broadcasted shape of the input parameter and dist_shape, taking into
        # account the potential size prefix
        inputs = args + tuple(kwargs.values())
        broadcast_shape = broadcast_dist_samples_shape(
            [np.asarray(i).shape for i in inputs] + [_dist_shape],
            size=size_tup
        )
        # We do this instead of broadcast_distribution_samples to avoid
        # creating a dummy array with dist_shape in memory
        inputs = get_broadcastable_dist_samples(
            inputs,
            size=size_tup,
            must_bcast_with=broadcast_shape,
        )
        # We modify the arguments with their broadcasted counterparts
        args = tuple(inputs[:len(args)])
        for offset, key in enumerate(kwargs):
            kwargs[key] = inputs[len(args) + offset]
    # Update kwargs with the keyword arguments that were not broadcasted
    kwargs.update(not_broadcast_kwargs)

    # We ensure that broadcast_shape is a tuple
    broadcast_shape = to_tuple(broadcast_shape)

    try:
        dist_bcast_shape = broadcast_dist_samples_shape(
            [_dist_shape, broadcast_shape],
            size=size,
        )
    except (ValueError, TypeError):
        raise TypeError('''Attempted to generate values with incompatible shapes:
            size: {size}
            size_tup: {size_tup}
            broadcast_shape[:len(size_tup)] == size_tup: {size_prepended}
            dist_shape: {dist_shape}
            broadcast_shape: {broadcast_shape}
        '''.format(size=size,
                   size_tup=size_tup,
                   dist_shape=dist_shape,
                   broadcast_shape=broadcast_shape,
                   size_prepended=broadcast_shape[:len(size_tup)] == size_tup)
        )
    if dist_bcast_shape[:len(size_tup)] == size_tup:
        samples = generator(size=dist_bcast_shape, *args, **kwargs)
    else:
        samples = generator(size=size_tup + dist_bcast_shape, *args, **kwargs)
    samples = np.asarray(samples)

    # reshape samples here
    if samples.ndim > 0 and samples.shape[0] == 1 and size_tup == (1,):
        if (len(samples.shape) > len(dist_shape) and
            samples.shape[-len(dist_shape):] == dist_shape[-len(dist_shape):]
        ):
            samples = samples.reshape(samples.shape[1:])

    if (one_d and samples.ndim > 0 and samples.shape[-1] == 1
        and (samples.shape != size_tup or
             size_tup == tuple() or
             size_tup == (1,))
    ):
        samples = samples.reshape(samples.shape[:-1])
    return np.asarray(samples)
