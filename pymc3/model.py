#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import collections
import itertools
import threading
import warnings

from sys import modules
from typing import TYPE_CHECKING, Any, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import scipy.sparse as sps
import theano
import theano.graph.basic
import theano.sparse as sparse
import theano.tensor as tt

from pandas import Series
from theano.compile import SharedVariable
from theano.graph.basic import Apply
from theano.tensor.var import TensorVariable

import pymc3 as pm

from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.exceptions import ImputationWarning
from pymc3.math import flatten_list
from pymc3.memoize import WithMemoization, memoize
from pymc3.theanof import floatX, generator, gradient, hessian, inputvars
from pymc3.util import get_transformed_name, get_var_name
from pymc3.vartypes import continuous_types, discrete_types, isgenerator, typefilter

__all__ = [
    "Model",
    "Factor",
    "compilef",
    "fn",
    "fastfn",
    "modelcontext",
    "Point",
    "Deterministic",
    "Potential",
    "set_data",
]

FlatView = collections.namedtuple("FlatView", "input, replacements, view")


class PyMC3Variable(TensorVariable):
    """Class to wrap Theano TensorVariable for custom behavior."""

    # Implement matrix multiplication infix operator: X @ w
    __matmul__ = tt.dot

    def __rmatmul__(self, other):
        return tt.dot(other, self)

    def _str_repr(self, name=None, dist=None, formatting="plain"):
        if getattr(self, "distribution", None) is None:
            if "latex" in formatting:
                return None
            else:
                return super().__str__()

        if name is None and hasattr(self, "name"):
            name = self.name
        if dist is None and hasattr(self, "distribution"):
            dist = self.distribution
        return self.distribution._str_repr(name=name, dist=dist, formatting=formatting)

    def _repr_latex_(self, *, formatting="latex_with_params", **kwargs):
        return self._str_repr(formatting=formatting, **kwargs)

    def __str__(self, **kwargs):
        try:
            return self._str_repr(formatting="plain", **kwargs)
        except:
            return super().__str__()

    __latex__ = _repr_latex_


class InstanceMethod:
    """Class for hiding references to instance methods so they can be pickled.

    >>> self.method = InstanceMethod(some_object, 'method_name')
    """

    def __init__(self, obj, method_name):
        self.obj = obj
        self.method_name = method_name

    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.method_name)(*args, **kwargs)


def incorporate_methods(source, destination, methods, wrapper=None, override=False):
    """
    Add attributes to a destination object which point to
    methods from from a source object.

    Parameters
    ----------
    source: object
        The source object containing the methods.
    destination: object
        The destination object for the methods.
    methods: list of str
        Names of methods to incorporate.
    wrapper: function
        An optional function to allow the source method to be
        wrapped. Should take the form my_wrapper(source, method_name)
        and return a single value.
    override: bool
        If the destination object already has a method/attribute
        an AttributeError will be raised if override is False (the default).
    """
    for method in methods:
        if hasattr(destination, method) and not override:
            raise AttributeError(
                f"Cannot add method {method!r}" + "to destination object as it already exists. "
                "To prevent this error set 'override=True'."
            )
        if hasattr(source, method):
            if wrapper is None:
                setattr(destination, method, getattr(source, method))
            else:
                setattr(destination, method, wrapper(source, method))
        else:
            setattr(destination, method, None)


def get_named_nodes_and_relations(graph):
    """Get the named nodes in a theano graph (i.e., nodes whose name
    attribute is not None) along with their relationships (i.e., the
    node's named parents, and named children, while skipping unnamed
    intermediate nodes)

    Parameters
    ----------
    graph: a theano node

    Returns:
    --------
    leaf_dict: Dict[str, node]
        A dictionary of name:node pairs, of the named nodes that
        have no named ancestors in the provided theano graph.
    descendents: Dict[node, Set[node]]
        Each key is a theano named node, and the corresponding value
        is the set of theano named nodes that are descendents with no
        intervening named nodes in the supplied ``graph``.
    ancestors: Dict[node, Set[node]]
        A dictionary of node:set([ancestors]) pairs. Each key
        is a theano named node, and the corresponding value is the set
        of theano named nodes that are ancestors with no intervening named
        nodes in the supplied ``graph``.

    """
    # We don't enforce distribution parameters to have a name but we may
    # attempt to get_named_nodes_and_relations from them anyway in
    # distributions.draw_values. This means that must take care only to add
    # graph to the ancestors and descendents dictionaries if it has a name.
    if graph.name is not None:
        ancestors = {graph: set()}
        descendents = {graph: set()}
    else:
        ancestors = {}
        descendents = {}
    descendents, ancestors = _get_named_nodes_and_relations(graph, None, ancestors, descendents)
    leaf_dict = {node.name: node for node, ancestor in ancestors.items() if len(ancestor) == 0}
    return leaf_dict, descendents, ancestors


def _get_named_nodes_and_relations(graph, descendent, descendents, ancestors):
    if getattr(graph, "owner", None) is None:  # Leaf node
        if graph.name is not None:  # Named leaf node
            if descendent is not None:  # Is None for the first node
                try:
                    descendents[graph].add(descendent)
                except KeyError:
                    descendents[graph] = {descendent}
                ancestors[descendent].add(graph)
            else:
                descendents[graph] = set()
            # Flag that the leaf node has no children
            ancestors[graph] = set()
    else:  # Intermediate node
        if graph.name is not None:  # Intermediate named node
            if descendent is not None:  # Is only None for the root node
                try:
                    descendents[graph].add(descendent)
                except KeyError:
                    descendents[graph] = {descendent}
                ancestors[descendent].add(graph)
            else:
                descendents[graph] = set()
            # The current node will be set as the descendent of the next
            # nodes only if it is a named node
            descendent = graph
            # Init the nodes children to an empty set
            ancestors[graph] = set()
        for i in graph.owner.inputs:
            temp_desc, temp_ances = _get_named_nodes_and_relations(
                i, descendent, descendents, ancestors
            )
            descendents.update(temp_desc)
            ancestors.update(temp_ances)
    return descendents, ancestors


def build_named_node_tree(graphs):
    """Build the combined descence/ancestry tree of named nodes (i.e., nodes
    whose name attribute is not None) in a list (or iterable) of theano graphs.
    The relationship tree does not include unnamed intermediate nodes present
    in the supplied graphs.

    Parameters
    ----------
    graphs - iterable of theano graphs

    Returns:
    --------
    leaf_dict: Dict[str, node]
        A dictionary of name:node pairs, of the named nodes that
        have no named ancestors in the provided theano graphs.
    descendents: Dict[node, Set[node]]
        A dictionary of node:set([parents]) pairs. Each key is
        a theano named node, and the corresponding value is the set of
        theano named nodes that are descendents with no intervening named
        nodes in the supplied ``graphs``.
    ancestors: Dict[node, Set[node]]
        A dictionary of node:set([ancestors]) pairs. Each key
        is a theano named node, and the corresponding value is the set
        of theano named nodes that are ancestors with no intervening named
        nodes in the supplied ``graphs``.

    """
    leaf_dict = {}
    named_nodes_descendents = {}
    named_nodes_ancestors = {}
    for graph in graphs:
        # Get the named nodes under the `param` node
        nn, nnd, nna = get_named_nodes_and_relations(graph)
        leaf_dict.update(nn)
        # Update the discovered parental relationships
        for k in nnd.keys():
            if k not in named_nodes_descendents.keys():
                named_nodes_descendents[k] = nnd[k]
            else:
                named_nodes_descendents[k].update(nnd[k])
        # Update the discovered child relationships
        for k in nna.keys():
            if k not in named_nodes_ancestors.keys():
                named_nodes_ancestors[k] = nna[k]
            else:
                named_nodes_ancestors[k].update(nna[k])
    return leaf_dict, named_nodes_descendents, named_nodes_ancestors


T = TypeVar("T", bound="ContextMeta")


class ContextMeta(type):
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """

    def __new__(cls, name, bases, dct, **kargs):  # pylint: disable=unused-argument
        "Add __enter__ and __exit__ methods to the class."

        def __enter__(self):
            self.__class__.context_class.get_contexts().append(self)
            # self._theano_config is set in Model.__new__
            self._config_context = None
            if hasattr(self, "_theano_config"):
                self._config_context = theano.config.change_flags(**self._theano_config)
                self._config_context.__enter__()
            return self

        def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
            self.__class__.context_class.get_contexts().pop()
            # self._theano_config is set in Model.__new__
            if self._config_context:
                self._config_context.__exit__(typ, value, traceback)

        dct[__enter__.__name__] = __enter__
        dct[__exit__.__name__] = __exit__

        # We strip off keyword args, per the warning from
        # StackExchange:
        # DO NOT send "**kargs" to "type.__new__".  It won't catch them and
        # you'll get a "TypeError: type() takes 1 or 3 arguments" exception.
        return super().__new__(cls, name, bases, dct)

    # FIXME: is there a more elegant way to automatically add methods to the class that
    # are instance methods instead of class methods?
    def __init__(
        cls, name, bases, nmspc, context_class: Optional[Type] = None, **kwargs
    ):  # pylint: disable=unused-argument
        """Add ``__enter__`` and ``__exit__`` methods to the new class automatically."""
        if context_class is not None:
            cls._context_class = context_class
        super().__init__(name, bases, nmspc)

    def get_context(cls, error_if_none=True) -> Optional[T]:
        """Return the most recently pushed context object of type ``cls``
        on the stack, or ``None``. If ``error_if_none`` is True (default),
        raise a ``TypeError`` instead of returning ``None``."""
        try:
            candidate = cls.get_contexts()[-1]  # type: Optional[T]
        except IndexError as e:
            # Calling code expects to get a TypeError if the entity
            # is unfound, and there's too much to fix.
            if error_if_none:
                raise TypeError("No %s on context stack" % str(cls))
            return None
        return candidate

    def get_contexts(cls) -> List[T]:
        """Return a stack of context instances for the ``context_class``
        of ``cls``."""
        # This lazily creates the context class's contexts
        # thread-local object, as needed. This seems inelegant to me,
        # but since the context class is not guaranteed to exist when
        # the metaclass is being instantiated, I couldn't figure out a
        # better way. [2019/10/11:rpg]

        # no race-condition here, contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        context_class = cls.context_class
        assert isinstance(context_class, type), (
            "Name of context class, %s was not resolvable to a class" % context_class
        )
        if not hasattr(context_class, "contexts"):
            context_class.contexts = threading.local()

        contexts = context_class.contexts

        if not hasattr(contexts, "stack"):
            contexts.stack = []
        return contexts.stack

    # the following complex property accessor is necessary because the
    # context_class may not have been created at the point it is
    # specified, so the context_class may be a class *name* rather
    # than a class.
    @property
    def context_class(cls) -> Type:
        def resolve_type(c: Union[Type, str]) -> Type:
            if isinstance(c, str):
                c = getattr(modules[cls.__module__], c)
            if isinstance(c, type):
                return c
            raise ValueError("Cannot resolve context class %s" % c)

        assert cls is not None
        if isinstance(cls._context_class, str):
            cls._context_class = resolve_type(cls._context_class)
        if not isinstance(cls._context_class, (str, type)):
            raise ValueError(
                "Context class for %s, %s, is not of the right type"
                % (cls.__name__, cls._context_class)
            )
        return cls._context_class

    # Inherit context class from parent
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.context_class = super().context_class

    # Initialize object in its own context...
    # Merged from InitContextMeta in the original.
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        with instance:  # appends context
            instance.__init__(*args, **kwargs)
        return instance


def modelcontext(model: Optional["Model"]) -> "Model":
    """
    Return the given model or, if none was supplied, try to find one in
    the context stack.
    """
    if model is None:
        model = Model.get_context(error_if_none=False)

        if model is None:
            # TODO: This should be a ValueError, but that breaks
            # ArviZ (and others?), so might need a deprecation.
            raise TypeError("No model on context stack.")
    return model


class Factor:
    """Common functionality for objects with a log probability density
    associated with them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def logp(self):
        """Compiled log probability density function"""
        return self.model.fn(self.logpt)

    @property
    def logp_elemwise(self):
        return self.model.fn(self.logp_elemwiset)

    def dlogp(self, vars=None):
        """Compiled log probability density gradient function"""
        return self.model.fn(gradient(self.logpt, vars))

    def d2logp(self, vars=None):
        """Compiled log probability density hessian function"""
        return self.model.fn(hessian(self.logpt, vars))

    @property
    def logp_nojac(self):
        return self.model.fn(self.logp_nojact)

    def dlogp_nojac(self, vars=None):
        """Compiled log density gradient function, without jacobian terms."""
        return self.model.fn(gradient(self.logp_nojact, vars))

    def d2logp_nojac(self, vars=None):
        """Compiled log density hessian function, without jacobian terms."""
        return self.model.fn(hessian(self.logp_nojact, vars))

    @property
    def fastlogp(self):
        """Compiled log probability density function"""
        return self.model.fastfn(self.logpt)

    def fastdlogp(self, vars=None):
        """Compiled log probability density gradient function"""
        return self.model.fastfn(gradient(self.logpt, vars))

    def fastd2logp(self, vars=None):
        """Compiled log probability density hessian function"""
        return self.model.fastfn(hessian(self.logpt, vars))

    @property
    def fastlogp_nojac(self):
        return self.model.fastfn(self.logp_nojact)

    def fastdlogp_nojac(self, vars=None):
        """Compiled log density gradient function, without jacobian terms."""
        return self.model.fastfn(gradient(self.logp_nojact, vars))

    def fastd2logp_nojac(self, vars=None):
        """Compiled log density hessian function, without jacobian terms."""
        return self.model.fastfn(hessian(self.logp_nojact, vars))

    @property
    def logpt(self):
        """Theano scalar of log-probability of the model"""
        if getattr(self, "total_size", None) is not None:
            logp = self.logp_sum_unscaledt * self.scaling
        else:
            logp = self.logp_sum_unscaledt
        if self.name is not None:
            logp.name = "__logp_%s" % self.name
        return logp

    @property
    def logp_nojact(self):
        """Theano scalar of log-probability, excluding jacobian terms."""
        if getattr(self, "total_size", None) is not None:
            logp = tt.sum(self.logp_nojac_unscaledt) * self.scaling
        else:
            logp = tt.sum(self.logp_nojac_unscaledt)
        if self.name is not None:
            logp.name = "__logp_%s" % self.name
        return logp


def withparent(meth):
    """Helper wrapper that passes calls to parent's instance"""

    def wrapped(self, *args, **kwargs):
        res = meth(self, *args, **kwargs)
        if getattr(self, "parent", None) is not None:
            getattr(self.parent, meth.__name__)(*args, **kwargs)
        return res

    # Unfortunately functools wrapper fails
    # when decorating built-in methods so we
    # need to fix that improper behaviour
    wrapped.__name__ = meth.__name__
    return wrapped


class treelist(list):
    """A list that passes mutable extending operations used in Model
    to parent list instance.
    Extending treelist you will also extend its parent
    """

    def __init__(self, iterable=(), parent=None):
        super().__init__(iterable)
        assert isinstance(parent, list) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.extend(self)

    # typechecking here works bad
    append = withparent(list.append)
    __iadd__ = withparent(list.__iadd__)
    extend = withparent(list.extend)

    def tree_contains(self, item):
        if isinstance(self.parent, treedict):
            return list.__contains__(self, item) or self.parent.tree_contains(item)
        elif isinstance(self.parent, list):
            return list.__contains__(self, item) or self.parent.__contains__(item)
        else:
            return list.__contains__(self, item)

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Method is removed as we are not able to determine appropriate logic for it"
        )

    # Added this because mypy didn't like having __imul__ without __mul__
    # This is my best guess about what this should do.  I might be happier
    # to kill both of these if they are not used.
    def __mul__(self, other) -> "treelist":
        return cast("treelist", list.__mul__(self, other))

    def __imul__(self, other) -> "treelist":
        t0 = len(self)
        list.__imul__(self, other)
        if self.parent is not None:
            self.parent.extend(self[t0:])
        return self  # python spec says should return the result.


class treedict(dict):
    """A dict that passes mutable extending operations used in Model
    to parent dict instance.
    Extending treedict you will also extend its parent
    """

    def __init__(self, iterable=(), parent=None, **kwargs):
        super().__init__(iterable, **kwargs)
        assert isinstance(parent, dict) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.update(self)

    # typechecking here works bad
    __setitem__ = withparent(dict.__setitem__)
    update = withparent(dict.update)

    def tree_contains(self, item):
        # needed for `add_random_variable` method
        if isinstance(self.parent, treedict):
            return dict.__contains__(self, item) or self.parent.tree_contains(item)
        elif isinstance(self.parent, dict):
            return dict.__contains__(self, item) or self.parent.__contains__(item)
        else:
            return dict.__contains__(self, item)


class ValueGradFunction:
    """Create a theano function that computes a value and its gradient.

    Parameters
    ----------
    costs: list of theano variables
        We compute the weighted sum of the specified theano values, and the gradient
        of that sum. The weights can be specified with `ValueGradFunction.set_weights`.
    grad_vars: list of named theano variables or None
        The arguments with respect to which the gradient is computed.
    extra_vars: list of named theano variables or None
        Other arguments of the function that are assumed constant. They
        are stored in shared variables and can be set using
        `set_extra_values`.
    dtype: str, default=theano.config.floatX
        The dtype of the arrays.
    casting: {'no', 'equiv', 'save', 'same_kind', 'unsafe'}, default='no'
        Casting rule for casting `grad_args` to the array dtype.
        See `numpy.can_cast` for a description of the options.
        Keep in mind that we cast the variables to the array *and*
        back from the array dtype to the variable dtype.
    compute_grads: bool, default=True
        If False, return only the logp, not the gradient.
    kwargs
        Extra arguments are passed on to `theano.function`.

    Attributes
    ----------
    size: int
        The number of elements in the parameter array.
    profile: theano profiling object or None
        The profiling object of the theano function that computes value and
        gradient. This is None unless `profile=True` was set in the
        kwargs.
    """

    def __init__(
        self,
        costs,
        grad_vars,
        extra_vars=None,
        *,
        dtype=None,
        casting="no",
        compute_grads=True,
        **kwargs,
    ):
        from pymc3.distributions import TensorType

        if extra_vars is None:
            extra_vars = []

        names = [arg.name for arg in grad_vars + extra_vars]
        if any(name is None for name in names):
            raise ValueError("Arguments must be named.")
        if len(set(names)) != len(names):
            raise ValueError("Names of the arguments are not unique.")

        self._grad_vars = grad_vars
        self._extra_vars = extra_vars
        self._extra_var_names = {var.name for var in extra_vars}

        if dtype is None:
            dtype = theano.config.floatX
        self.dtype = dtype

        self._n_costs = len(costs)
        if self._n_costs == 0:
            raise ValueError("At least one cost is required.")
        weights = np.ones(self._n_costs - 1, dtype=self.dtype)
        self._weights = theano.shared(weights, "__weights")

        cost = costs[0]
        for i, val in enumerate(costs[1:]):
            if cost.ndim > 0 or val.ndim > 0:
                raise ValueError("All costs must be scalar.")
            cost = cost + self._weights[i] * val

        self._cost = cost
        self._ordering = ArrayOrdering(grad_vars)
        self.size = self._ordering.size
        self._extra_are_set = False
        for var in self._grad_vars:
            if not np.can_cast(var.dtype, self.dtype, casting):
                raise TypeError(
                    f"Invalid dtype for variable {var.name}. Can not "
                    f"cast to {self.dtype} with casting rule {casting}."
                )
            if not np.issubdtype(var.dtype, np.floating):
                raise TypeError(
                    f"Invalid dtype for variable {var.name}. Must be "
                    f"floating point but is {var.dtype}."
                )

        givens = []
        self._extra_vars_shared = {}
        for var in extra_vars:
            shared = theano.shared(var.tag.test_value, var.name + "_shared__")
            # test TensorType compatibility
            if hasattr(var.tag.test_value, "shape"):
                testtype = TensorType(var.dtype, var.tag.test_value.shape)

                if testtype != shared.type:
                    shared.type = testtype
            self._extra_vars_shared[var.name] = shared
            givens.append((var, shared))

        self._vars_joined, self._cost_joined = self._build_joined(
            self._cost, grad_vars, self._ordering.vmap
        )

        if compute_grads:
            grad = tt.grad(self._cost_joined, self._vars_joined)
            grad.name = "__grad"
            outputs = [self._cost_joined, grad]
        else:
            outputs = self._cost_joined

        inputs = [self._vars_joined]

        self._theano_function = theano.function(inputs, outputs, givens=givens, **kwargs)

    def set_weights(self, values):
        if values.shape != (self._n_costs - 1,):
            raise ValueError("Invalid shape. Must be (n_costs - 1,).")
        self._weights.set_value(values)

    def set_extra_values(self, extra_vars):
        self._extra_are_set = True
        for var in self._extra_vars:
            self._extra_vars_shared[var.name].set_value(extra_vars[var.name])

    def get_extra_values(self):
        if not self._extra_are_set:
            raise ValueError("Extra values are not set.")

        return {var.name: self._extra_vars_shared[var.name].get_value() for var in self._extra_vars}

    def __call__(self, array, grad_out=None, extra_vars=None):
        if extra_vars is not None:
            self.set_extra_values(extra_vars)

        if not self._extra_are_set:
            raise ValueError("Extra values are not set.")

        if array.shape != (self.size,):
            raise ValueError(
                "Invalid shape for array. Must be {} but is {}.".format((self.size,), array.shape)
            )

        if grad_out is None:
            out = np.empty_like(array)
        else:
            out = grad_out

        output = self._theano_function(array)
        if grad_out is None:
            return output
        else:
            np.copyto(out, output[1])
            return output[0]

    @property
    def profile(self):
        """Profiling information of the underlying theano function."""
        return self._theano_function.profile

    def dict_to_array(self, point):
        """Convert a dictionary with values for grad_vars to an array."""
        array = np.empty(self.size, dtype=self.dtype)
        for varmap in self._ordering.vmap:
            array[varmap.slc] = point[varmap.var].ravel().astype(self.dtype)
        return array

    def array_to_dict(self, array):
        """Convert an array to a dictionary containing the grad_vars."""
        if array.shape != (self.size,):
            raise ValueError(f"Array should have shape ({self.size},) but has {array.shape}")
        if array.dtype != self.dtype:
            raise ValueError(
                f"Array has invalid dtype. Should be {self._dtype} but is {self.dtype}"
            )
        point = {}
        for varmap in self._ordering.vmap:
            data = array[varmap.slc].reshape(varmap.shp)
            point[varmap.var] = data.astype(varmap.dtyp)

        return point

    def array_to_full_dict(self, array):
        """Convert an array to a dictionary with grad_vars and extra_vars."""
        point = self.array_to_dict(array)
        for name, var in self._extra_vars_shared.items():
            point[name] = var.get_value()
        return point

    def _build_joined(self, cost, args, vmap):
        args_joined = tt.vector("__args_joined")
        args_joined.tag.test_value = np.zeros(self.size, dtype=self.dtype)

        joined_slices = {}
        for vmap in vmap:
            sliced = args_joined[vmap.slc].reshape(vmap.shp)
            sliced.name = vmap.var
            joined_slices[vmap.var] = sliced

        replace = {var: joined_slices[var.name] for var in args}
        return args_joined, theano.clone(cost, replace=replace)


class Model(Factor, WithMemoization, metaclass=ContextMeta):
    """Encapsulates the variables and likelihood factors of a model.

    Model class can be used for creating class based models. To create
    a class based model you should inherit from :class:`~.Model` and
    override :meth:`~.__init__` with arbitrary definitions (do not
    forget to call base class :meth:`__init__` first).

    Parameters
    ----------
    name: str
        name that will be used as prefix for names of all random
        variables defined within model
    model: Model
        instance of Model that is supposed to be a parent for the new
        instance. If ``None``, context will be used. All variables
        defined within instance will be passed to the parent instance.
        So that 'nested' model contributes to the variables and
        likelihood factors of parent model.
    theano_config: dict
        A dictionary of theano config values that should be set
        temporarily in the model context. See the documentation
        of theano for a complete list. Set config key
        ``compute_test_value`` to `raise` if it is None.
    check_bounds: bool
        Ensure that input parameters to distributions are in a valid
        range. If your model is built in a way where you know your
        parameters can only take on valid values you can set this to
        False for increased speed.

    Examples
    --------

    How to define a custom model

    .. code-block:: python

        class CustomModel(Model):
            # 1) override init
            def __init__(self, mean=0, sigma=1, name='', model=None):
                # 2) call super's init first, passing model and name
                # to it name will be prefix for all variables here if
                # no name specified for model there will be no prefix
                super().__init__(name, model)
                # now you are in the context of instance,
                # `modelcontext` will return self you can define
                # variables in several ways note, that all variables
                # will get model's name prefix

                # 3) you can create variables with Var method
                self.Var('v1', Normal.dist(mu=mean, sigma=sd))
                # this will create variable named like '{prefix_}v1'
                # and assign attribute 'v1' to instance created
                # variable can be accessed with self.v1 or self['v1']

                # 4) this syntax will also work as we are in the
                # context of instance itself, names are given as usual
                Normal('v2', mu=mean, sigma=sd)

                # something more complex is allowed, too
                half_cauchy = HalfCauchy('sd', beta=10, testval=1.)
                Normal('v3', mu=mean, sigma=half_cauchy)

                # Deterministic variables can be used in usual way
                Deterministic('v3_sq', self.v3 ** 2)

                # Potentials too
                Potential('p1', tt.constant(1))

        # After defining a class CustomModel you can use it in several
        # ways

        # I:
        #   state the model within a context
        with Model() as model:
            CustomModel()
            # arbitrary actions

        # II:
        #   use new class as entering point in context
        with CustomModel() as model:
            Normal('new_normal_var', mu=1, sigma=0)

        # III:
        #   just get model instance with all that was defined in it
        model = CustomModel()

        # IV:
        #   use many custom models within one context
        with Model() as model:
            CustomModel(mean=1, name='first')
            CustomModel(mean=2, name='second')
    """

    if TYPE_CHECKING:

        def __enter__(self: "Model") -> "Model":
            ...

        def __exit__(self: "Model", *exc: Any) -> bool:
            ...

    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get("model") is not None:
            instance._parent = kwargs.get("model")
        else:
            instance._parent = cls.get_context(error_if_none=False)
        theano_config = kwargs.get("theano_config", None)
        if theano_config is None or "compute_test_value" not in theano_config:
            theano_config = {"compute_test_value": "raise"}
        instance._theano_config = theano_config
        return instance

    def __init__(self, name="", model=None, theano_config=None, coords=None, check_bounds=True):
        self.name = name
        self.coords = {}
        self.RV_dims = {}
        self.add_coords(coords)
        self.check_bounds = check_bounds

        if self.parent is not None:
            self.named_vars = treedict(parent=self.parent.named_vars)
            self.free_RVs = treelist(parent=self.parent.free_RVs)
            self.observed_RVs = treelist(parent=self.parent.observed_RVs)
            self.deterministics = treelist(parent=self.parent.deterministics)
            self.potentials = treelist(parent=self.parent.potentials)
            self.missing_values = treelist(parent=self.parent.missing_values)
        else:
            self.named_vars = treedict()
            self.free_RVs = treelist()
            self.observed_RVs = treelist()
            self.deterministics = treelist()
            self.potentials = treelist()
            self.missing_values = treelist()

    @property
    def model(self):
        return self

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        model = self
        while not model.isroot:
            model = model.parent
        return model

    @property
    def isroot(self):
        return self.parent is None

    @property  # type: ignore
    @memoize(bound=True)
    def bijection(self):
        vars = inputvars(self.vars)

        bij = DictToArrayBijection(ArrayOrdering(vars), self.test_point)

        return bij

    @property
    def dict_to_array(self):
        return self.bijection.map

    @property
    def ndim(self):
        return sum(var.dsize for var in self.free_RVs)

    @property
    def logp_array(self):
        return self.bijection.mapf(self.fastlogp)

    @property
    def dlogp_array(self):
        vars = inputvars(self.cont_vars)
        return self.bijection.mapf(self.fastdlogp(vars))

    def logp_dlogp_function(self, grad_vars=None, tempered=False, **kwargs):
        """Compile a theano function that computes logp and gradient.

        Parameters
        ----------
        grad_vars: list of random variables, optional
            Compute the gradient with respect to those variables. If None,
            use all free random variables of this model.
        tempered: bool
            Compute the tempered logp `free_logp + alpha * observed_logp`.
            `alpha` can be changed using `ValueGradFunction.set_weights([alpha])`.
        """
        if grad_vars is None:
            grad_vars = list(typefilter(self.free_RVs, continuous_types))
        else:
            for var in grad_vars:
                if var.dtype not in continuous_types:
                    raise ValueError("Can only compute the gradient of continuous types: %s" % var)

        if tempered:
            with self:
                free_RVs_logp = tt.sum(
                    [tt.sum(var.logpt) for var in self.free_RVs + self.potentials]
                )
                observed_RVs_logp = tt.sum([tt.sum(var.logpt) for var in self.observed_RVs])

            costs = [free_RVs_logp, observed_RVs_logp]
        else:
            costs = [self.logpt]
        varnames = [var.name for var in grad_vars]
        extra_vars = [var for var in self.free_RVs if var.name not in varnames]
        return ValueGradFunction(costs, grad_vars, extra_vars, **kwargs)

    @property
    def logpt(self):
        """Theano scalar of log-probability of the model"""
        with self:
            factors = [var.logpt for var in self.basic_RVs] + self.potentials
            logp = tt.sum([tt.sum(factor) for factor in factors])
            if self.name:
                logp.name = "__logp_%s" % self.name
            else:
                logp.name = "__logp"
            return logp

    @property
    def logp_nojact(self):
        """Theano scalar of log-probability of the model but without the jacobian
        if transformed Random Variable is presented.
        Note that If there is no transformed variable in the model, logp_nojact
        will be the same as logpt as there is no need for Jacobian correction.
        """
        with self:
            factors = [var.logp_nojact for var in self.basic_RVs] + self.potentials
            logp = tt.sum([tt.sum(factor) for factor in factors])
            if self.name:
                logp.name = "__logp_nojac_%s" % self.name
            else:
                logp.name = "__logp_nojac"
            return logp

    @property
    def varlogpt(self):
        """Theano scalar of log-probability of the unobserved random variables
        (excluding deterministic)."""
        with self:
            factors = [var.logpt for var in self.free_RVs]
            return tt.sum(factors)

    @property
    def datalogpt(self):
        with self:
            factors = [var.logpt for var in self.observed_RVs]
            factors += [tt.sum(factor) for factor in self.potentials]
            return tt.sum(factors)

    @property
    def vars(self):
        """List of unobserved random variables used as inputs to the model
        (which excludes deterministics).
        """
        return self.free_RVs

    @property
    def basic_RVs(self):
        """List of random variables the model is defined in terms of
        (which excludes deterministics).
        """
        return self.free_RVs + self.observed_RVs

    @property
    def unobserved_RVs(self):
        """List of all random variable, including deterministic ones."""
        return self.vars + self.deterministics

    @property
    def test_point(self):
        """Test point used to check that the model doesn't generate errors"""
        return Point(((var, var.tag.test_value) for var in self.vars), model=self)

    @property
    def disc_vars(self):
        """All the discrete variables in the model"""
        return list(typefilter(self.vars, discrete_types))

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.vars, continuous_types))

    def shape_from_dims(self, dims):
        shape = []
        if len(set(dims)) != len(dims):
            raise ValueError("Can not contain the same dimension name twice.")
        for dim in dims:
            if dim not in self.coords:
                raise ValueError(
                    "Unknown dimension name '%s'. All dimension "
                    "names must be specified in the `coords` "
                    "argument of the model or through a pm.Data "
                    "variable." % dim
                )
            shape.extend(np.shape(self.coords[dim]))
        return tuple(shape)

    def add_coords(self, coords):
        if coords is None:
            return

        for name in coords:
            if name in {"draw", "chain"}:
                raise ValueError(
                    "Dimensions can not be named `draw` or `chain`, as they are reserved for the sampler's outputs."
                )
            if name in self.coords:
                if not coords[name].equals(self.coords[name]):
                    raise ValueError("Duplicate and incompatiple coordinate: %s." % name)
            else:
                self.coords[name] = coords[name]

    def Var(self, name, dist, data=None, total_size=None, dims=None):
        """Create and add (un)observed random variable to the model with an
        appropriate prior distribution.

        Parameters
        ----------
        name: str
        dist: distribution for the random variable
        data: array_like (optional)
           If data is provided, the variable is observed. If None,
           the variable is unobserved.
        total_size: scalar
            upscales logp of variable with ``coef = total_size/var.shape[0]``
        dims : tuple
            Dimension names for the variable.

        Returns
        -------
        FreeRV or ObservedRV
        """
        name = self.name_for(name)

        if data is None:
            if getattr(dist, "transform", None) is None:
                with self:
                    var = FreeRV(name=name, distribution=dist, total_size=total_size, model=self)
                self.free_RVs.append(var)
            else:
                with self:
                    var = TransformedRV(
                        name=name,
                        distribution=dist,
                        transform=dist.transform,
                        total_size=total_size,
                        model=self,
                    )
                pm._log.debug(
                    "Applied {transform}-transform to {name}"
                    " and added transformed {orig_name} to model.".format(
                        transform=dist.transform.name,
                        name=name,
                        orig_name=get_transformed_name(name, dist.transform),
                    )
                )
                self.deterministics.append(var)
                self.add_random_variable(var, dims)
                return var
        elif isinstance(data, dict):
            with self:
                var = MultiObservedRV(
                    name=name,
                    data=data,
                    distribution=dist,
                    total_size=total_size,
                    model=self,
                )
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs += var.missing_values
                self.missing_values += var.missing_values
                for v in var.missing_values:
                    self.named_vars[v.name] = v
        else:
            with self:
                var = ObservedRV(
                    name=name,
                    data=data,
                    distribution=dist,
                    total_size=total_size,
                    model=self,
                )
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs.append(var.missing_values)
                self.missing_values.append(var.missing_values)
                self.named_vars[var.missing_values.name] = var.missing_values

        self.add_random_variable(var, dims)
        return var

    def add_random_variable(self, var, dims=None):
        """Add a random variable to the named variables of the model."""
        if self.named_vars.tree_contains(var.name):
            raise ValueError(f"Variable name {var.name} already exists.")

        if dims is not None:
            if isinstance(dims, str):
                dims = (dims,)
            assert all(dim in self.coords for dim in dims)
            self.RV_dims[var.name] = dims

        self.named_vars[var.name] = var
        if not hasattr(self, self.name_of(var.name)):
            setattr(self, self.name_of(var.name), var)

    @property
    def prefix(self):
        return "%s_" % self.name if self.name else ""

    def name_for(self, name):
        """Checks if name has prefix and adds if needed"""
        if self.prefix:
            if not name.startswith(self.prefix):
                return f"{self.prefix}{name}"
            else:
                return name
        else:
            return name

    def name_of(self, name):
        """Checks if name has prefix and deletes if needed"""
        if not self.prefix or not name:
            return name
        elif name.startswith(self.prefix):
            return name[len(self.prefix) :]
        else:
            return name

    def __getitem__(self, key):
        try:
            return self.named_vars[key]
        except KeyError as e:
            try:
                return self.named_vars[self.name_for(key)]
            except KeyError:
                raise e

    def makefn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns ``outs`` and takes the variable
        ancestors of ``outs`` as inputs.

        Parameters
        ----------
        outs: Theano variable or iterable of Theano variables
        mode: Theano compilation mode

        Returns
        -------
        Compiled Theano function
        """
        with self:
            return theano.function(
                self.vars,
                outs,
                allow_input_downcast=True,
                on_unused_input="ignore",
                accept_inplace=True,
                mode=mode,
                *args,
                **kwargs,
            )

    def fn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns the values of ``outs``
        and takes values of model vars as arguments.

        Parameters
        ----------
        outs: Theano variable or iterable of Theano variables
        mode: Theano compilation mode

        Returns
        -------
        Compiled Theano function
        """
        return LoosePointFunc(self.makefn(outs, mode, *args, **kwargs), self)

    def fastfn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns ``outs`` and takes values
        of model vars as a dict as an argument.

        Parameters
        ----------
        outs: Theano variable or iterable of Theano variables
        mode: Theano compilation mode

        Returns
        -------
        Compiled Theano function as point function.
        """
        f = self.makefn(outs, mode, *args, **kwargs)
        return FastPointFunc(f)

    def profile(self, outs, n=1000, point=None, profile=True, *args, **kwargs):
        """Compiles and profiles a Theano function which returns ``outs`` and
        takes values of model vars as a dict as an argument.

        Parameters
        ----------
        outs: Theano variable or iterable of Theano variables
        n: int, default 1000
            Number of iterations to run
        point: point
            Point to pass to the function
        profile: True or ProfileStats
        args, kwargs
            Compilation args

        Returns
        -------
        ProfileStats
            Use .summary() to print stats.
        """
        f = self.makefn(outs, profile=profile, *args, **kwargs)
        if point is None:
            point = self.test_point

        for _ in range(n):
            f(**point)

        return f.profile

    def flatten(self, vars=None, order=None, inputvar=None):
        """Flattens model's input and returns:

        FlatView with
            * input vector variable
            * replacements ``input_var -> vars``
            * view `{variable: VarMap}`

        Parameters
        ----------
        vars: list of variables or None
            if None, then all model.free_RVs are used for flattening input
        order: ArrayOrdering
            Optional, use predefined ordering
        inputvar: tt.vector
            Optional, use predefined inputvar

        Returns
        -------
        flat_view
        """
        if vars is None:
            vars = self.free_RVs
        if order is None:
            order = ArrayOrdering(vars)
        if inputvar is None:
            inputvar = tt.vector("flat_view", dtype=theano.config.floatX)
            if theano.config.compute_test_value != "off":
                if vars:
                    inputvar.tag.test_value = flatten_list(vars).tag.test_value
                else:
                    inputvar.tag.test_value = np.asarray([], inputvar.dtype)
        replacements = {
            self.named_vars[name]: inputvar[slc].reshape(shape).astype(dtype)
            for name, slc, shape, dtype in order.vmap
        }
        view = {vm.var: vm for vm in order.vmap}
        flat_view = FlatView(inputvar, replacements, view)
        return flat_view

    def check_test_point(self, test_point=None, round_vals=2):
        """Checks log probability of test_point for all random variables in the model.

        Parameters
        ----------
        test_point: Point
            Point to be evaluated.
            if None, then all model.test_point is used
        round_vals: int
            Number of decimals to round log-probabilities

        Returns
        -------
        Pandas Series
        """
        if test_point is None:
            test_point = self.test_point

        return Series(
            {RV.name: np.round(RV.logp(test_point), round_vals) for RV in self.basic_RVs},
            name="Log-probability of test_point",
        )

    def _str_repr(self, formatting="plain", **kwargs):
        all_rv = itertools.chain(self.unobserved_RVs, self.observed_RVs)

        if "latex" in formatting:
            rv_reprs = [rv.__latex__(formatting=formatting) for rv in all_rv]
            rv_reprs = [
                rv_repr.replace(r"\sim", r"&\sim &").strip("$")
                for rv_repr in rv_reprs
                if rv_repr is not None
            ]
            return r"""$$
                \begin{{array}}{{rcl}}
                {}
                \end{{array}}
                $$""".format(
                "\\\\".join(rv_reprs)
            )
        else:
            rv_reprs = [rv.__str__() for rv in all_rv]
            rv_reprs = [
                rv_repr for rv_repr in rv_reprs if not "TransformedDistribution()" in rv_repr
            ]
            # align vars on their ~
            names = [s[: s.index("~") - 1] for s in rv_reprs]
            distrs = [s[s.index("~") + 2 :] for s in rv_reprs]
            maxlen = str(max(len(x) for x in names))
            rv_reprs = [
                ("{name:>" + maxlen + "} ~ {distr}").format(name=n, distr=d)
                for n, d in zip(names, distrs)
            ]
            return "\n".join(rv_reprs)

    def __str__(self, **kwargs):
        return self._str_repr(formatting="plain", **kwargs)

    def _repr_latex_(self, *, formatting="latex", **kwargs):
        return self._str_repr(formatting=formatting, **kwargs)

    __latex__ = _repr_latex_


# this is really disgusting, but it breaks a self-loop: I can't pass Model
# itself as context class init arg.
Model._context_class = Model


def set_data(new_data, model=None):
    """Sets the value of one or more data container variables.

    Parameters
    ----------
    new_data: dict
        New values for the data containers. The keys of the dictionary are
        the variables' names in the model and the values are the objects
        with which to update.
    model: Model (optional if in `with` context)

    Examples
    --------

    .. code:: ipython

        >>> import pymc3 as pm
        >>> with pm.Model() as model:
        ...     x = pm.Data('x', [1., 2., 3.])
        ...     y = pm.Data('y', [1., 2., 3.])
        ...     beta = pm.Normal('beta', 0, 1)
        ...     obs = pm.Normal('obs', x * beta, 1, observed=y)
        ...     trace = pm.sample(1000, tune=1000)

    Set the value of `x` to predict on new data.

    .. code:: ipython

        >>> with model:
        ...     pm.set_data({'x': [5., 6., 9.]})
        ...     y_test = pm.sample_posterior_predictive(trace)
        >>> y_test['obs'].mean(axis=0)
        array([4.6088569 , 5.54128318, 8.32953844])
    """
    model = modelcontext(model)

    for variable_name, new_value in new_data.items():
        if isinstance(model[variable_name], SharedVariable):
            if isinstance(new_value, list):
                new_value = np.array(new_value)
            model[variable_name].set_value(pandas_to_array(new_value))
        else:
            message = (
                "The variable `{}` must be defined as `pymc3."
                "Data` inside the model to allow updating. The "
                "current type is: "
                "{}.".format(variable_name, type(model[variable_name]))
            )
            raise TypeError(message)


def fn(outs, mode=None, model=None, *args, **kwargs):
    """Compiles a Theano function which returns the values of ``outs`` and
    takes values of model vars as arguments.

    Parameters
    ----------
    outs: Theano variable or iterable of Theano variables
    mode: Theano compilation mode

    Returns
    -------
    Compiled Theano function
    """
    model = modelcontext(model)
    return model.fn(outs, mode, *args, **kwargs)


def fastfn(outs, mode=None, model=None):
    """Compiles a Theano function which returns ``outs`` and takes values of model
    vars as a dict as an argument.

    Parameters
    ----------
    outs: Theano variable or iterable of Theano variables
    mode: Theano compilation mode

    Returns
    -------
    Compiled Theano function as point function.
    """
    model = modelcontext(model)
    return model.fastfn(outs, mode)


def Point(*args, **kwargs):
    """Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
    args, kwargs
        arguments to build a dict
    """
    model = modelcontext(kwargs.pop("model", None))
    args = list(args)
    try:
        d = dict(*args, **kwargs)
    except Exception as e:
        raise TypeError(f"can't turn {args} and {kwargs} into a dict. {e}")
    return {
        get_var_name(k): np.array(v)
        for k, v in d.items()
        if get_var_name(k) in map(get_var_name, model.vars)
    }


class FastPointFunc:
    """Wraps so a function so it takes a dict of arguments instead of arguments."""

    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)


class LoosePointFunc:
    """Wraps so a function so it takes a dict of arguments instead of arguments
    but can still take arguments."""

    def __init__(self, f, model):
        self.f = f
        self.model = model

    def __call__(self, *args, **kwargs):
        point = Point(model=self.model, *args, **kwargs)
        return self.f(**point)


compilef = fastfn


def _get_scaling(total_size, shape, ndim):
    """
    Gets scaling constant for logp

    Parameters
    ----------
    total_size: int or list[int]
    shape: shape
        shape to scale
    ndim: int
        ndim hint

    Returns
    -------
    scalar
    """
    if total_size is None:
        coef = floatX(1)
    elif isinstance(total_size, int):
        if ndim >= 1:
            denom = shape[0]
        else:
            denom = 1
        coef = floatX(total_size) / floatX(denom)
    elif isinstance(total_size, (list, tuple)):
        if not all(isinstance(i, int) for i in total_size if (i is not Ellipsis and i is not None)):
            raise TypeError(
                "Unrecognized `total_size` type, expected "
                "int or list of ints, got %r" % total_size
            )
        if Ellipsis in total_size:
            sep = total_size.index(Ellipsis)
            begin = total_size[:sep]
            end = total_size[sep + 1 :]
            if Ellipsis in end:
                raise ValueError(
                    "Double Ellipsis in `total_size` is restricted, got %r" % total_size
                )
        else:
            begin = total_size
            end = []
        if (len(begin) + len(end)) > ndim:
            raise ValueError(
                "Length of `total_size` is too big, "
                "number of scalings is bigger that ndim, got %r" % total_size
            )
        elif (len(begin) + len(end)) == 0:
            return floatX(1)
        if len(end) > 0:
            shp_end = shape[-len(end) :]
        else:
            shp_end = np.asarray([])
        shp_begin = shape[: len(begin)]
        begin_coef = [floatX(t) / shp_begin[i] for i, t in enumerate(begin) if t is not None]
        end_coef = [floatX(t) / shp_end[i] for i, t in enumerate(end) if t is not None]
        coefs = begin_coef + end_coef
        coef = tt.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return tt.as_tensor(floatX(coef))


class FreeRV(Factor, PyMC3Variable):
    """Unobserved random variable that a model is specified in terms of."""

    dshape = None  # type: Tuple[int, ...]
    size = None  # type: int
    distribution = None  # type: Optional[Distribution]
    model = None  # type: Optional[Model]

    def __init__(
        self,
        type=None,
        owner=None,
        index=None,
        name=None,
        distribution=None,
        total_size=None,
        model=None,
    ):
        """
        Parameters
        ----------
        type: theano type (optional)
        owner: theano owner (optional)
        name: str
        distribution: Distribution
        model: Model
        total_size: scalar Tensor (optional)
            needed for upscaling logp
        """
        if type is None:
            type = distribution.type
        super().__init__(type, owner, index, name)

        if distribution is not None:
            self.dshape = tuple(distribution.shape)
            self.dsize = int(np.prod(distribution.shape))
            self.distribution = distribution
            self.tag.test_value = (
                np.ones(distribution.shape, distribution.dtype) * distribution.default()
            )
            self.logp_elemwiset = distribution.logp(self)
            # The logp might need scaling in minibatches.
            # This is done in `Factor`.
            self.logp_sum_unscaledt = distribution.logp_sum(self)
            self.logp_nojac_unscaledt = distribution.logp_nojac(self)
            self.total_size = total_size
            self.model = model
            self.scaling = _get_scaling(total_size, self.shape, self.ndim)

            incorporate_methods(
                source=distribution,
                destination=self,
                methods=["random"],
                wrapper=InstanceMethod,
            )

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


def pandas_to_array(data):
    """Convert a pandas object to a NumPy array.

    XXX: When `data` is a generator, this will return a Theano tensor!

    """
    if hasattr(data, "values"):  # pandas
        if data.isnull().any().any():  # missing values
            ret = np.ma.MaskedArray(data.values, data.isnull().values)
        else:
            ret = data.values
    elif hasattr(data, "mask"):
        if data.mask.any():
            ret = data
        else:  # empty mask
            ret = data.filled()
    elif isinstance(data, theano.graph.basic.Variable):
        ret = data
    elif sps.issparse(data):
        ret = data
    elif isgenerator(data):
        ret = generator(data)
    else:
        ret = np.asarray(data)

    # type handling to enable index variables when data is int:
    if hasattr(data, "dtype"):
        if "int" in str(data.dtype):
            return pm.intX(ret)
        # otherwise, assume float:
        else:
            return pm.floatX(ret)
    # needed for uses of this function other than with pm.Data:
    else:
        return pm.floatX(ret)


def as_tensor(data, name, model, distribution):
    dtype = distribution.dtype
    data = pandas_to_array(data).astype(dtype)

    if hasattr(data, "mask"):
        impute_message = (
            "Data in {name} contains missing values and"
            " will be automatically imputed from the"
            " sampling distribution.".format(name=name)
        )
        warnings.warn(impute_message, ImputationWarning)
        from pymc3.distributions import NoDistribution

        testval = np.broadcast_to(distribution.default(), data.shape)[data.mask]
        fakedist = NoDistribution.dist(
            shape=data.mask.sum(),
            dtype=dtype,
            testval=testval,
            parent_dist=distribution,
        )
        missing_values = FreeRV(name=name + "_missing", distribution=fakedist, model=model)
        constant = tt.as_tensor_variable(data.filled())

        dataTensor = tt.set_subtensor(constant[data.mask.nonzero()], missing_values)
        dataTensor.missing_values = missing_values
        return dataTensor
    elif sps.issparse(data):
        data = sparse.basic.as_sparse(data, name=name)
        data.missing_values = None
        return data
    else:
        data = tt.as_tensor_variable(data, name=name)
        data.missing_values = None
        return data


class ObservedRV(Factor, PyMC3Variable):
    """Observed random variable that a model is specified in terms of.
    Potentially partially observed.
    """

    def __init__(
        self,
        type=None,
        owner=None,
        index=None,
        name=None,
        data=None,
        distribution=None,
        total_size=None,
        model=None,
    ):
        """
        Parameters
        ----------
        type: theano type (optional)
        owner: theano owner (optional)
        name: str
        distribution: Distribution
        model: Model
        total_size: scalar Tensor (optional)
            needed for upscaling logp
        """
        from pymc3.distributions import TensorType

        if hasattr(data, "type") and isinstance(data.type, tt.TensorType):
            type = data.type

        if type is None:
            data = pandas_to_array(data)
            if isinstance(data, theano.graph.basic.Variable):
                type = data.type
            else:
                type = TensorType(distribution.dtype, data.shape)

        self.observations = data

        super().__init__(type, owner, index, name)

        if distribution is not None:
            data = as_tensor(data, name, model, distribution)

            self.missing_values = data.missing_values
            self.logp_elemwiset = distribution.logp(data)
            # The logp might need scaling in minibatches.
            # This is done in `Factor`.
            self.logp_sum_unscaledt = distribution.logp_sum(data)
            self.logp_nojac_unscaledt = distribution.logp_nojac(data)
            self.total_size = total_size
            self.model = model
            self.distribution = distribution

            # make this RV a view on the combined missing/nonmissing array
            Apply(theano.compile.view_op, inputs=[data], outputs=[self])
            self.tag.test_value = theano.compile.view_op(data).tag.test_value.astype(self.dtype)
            self.scaling = _get_scaling(total_size, data.shape, data.ndim)

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


class MultiObservedRV(Factor):
    """Observed random variable that a model is specified in terms of.
    Potentially partially observed.
    """

    def __init__(self, name, data, distribution, total_size=None, model=None):
        """
        Parameters
        ----------
        type: theano type (optional)
        owner: theano owner (optional)
        name: str
        distribution: Distribution
        model: Model
        total_size: scalar Tensor (optional)
            needed for upscaling logp
        """
        self.name = name
        self.data = {
            name: as_tensor(data, name, model, distribution) for name, data in data.items()
        }

        self.missing_values = [
            datum.missing_values for datum in self.data.values() if datum.missing_values is not None
        ]
        self.logp_elemwiset = distribution.logp(**self.data)
        # The logp might need scaling in minibatches.
        # This is done in `Factor`.
        self.logp_sum_unscaledt = distribution.logp_sum(**self.data)
        self.logp_nojac_unscaledt = distribution.logp_nojac(**self.data)
        self.total_size = total_size
        self.model = model
        self.distribution = distribution
        self.scaling = _get_scaling(total_size, self.logp_elemwiset.shape, self.logp_elemwiset.ndim)

    # Make hashable by id for draw_values
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        "Use object identity for MultiObservedRV equality."
        # This is likely a Bad Thing, but changing it would break a lot of code.
        return self is other

    def __ne__(self, other):
        return not self == other


def _walk_up_rv(rv, formatting="plain"):
    """Walk up theano graph to get inputs for deterministic RV."""
    all_rvs = []
    parents = list(itertools.chain(*[j.inputs for j in rv.get_parents()]))
    if parents:
        for parent in parents:
            all_rvs.extend(_walk_up_rv(parent, formatting=formatting))
    else:
        name = rv.name if rv.name else "Constant"
        fmt = r"\text{{{name}}}" if "latex" in formatting else "{name}"
        all_rvs.append(fmt.format(name=name))
    return all_rvs


class DeterministicWrapper(tt.TensorVariable):
    def _str_repr(self, formatting="plain"):
        if "latex" in formatting:
            if formatting == "latex_with_params":
                return r"$\text{{{name}}} \sim \text{{Deterministic}}({args})$".format(
                    name=self.name, args=r",~".join(_walk_up_rv(self, formatting=formatting))
                )
            return fr"$\text{{{self.name}}} \sim \text{{Deterministic}}$"
        else:
            if formatting == "plain_with_params":
                args = ", ".join(_walk_up_rv(self, formatting=formatting))
                return f"{self.name} ~ Deterministic({args})"
            return f"{self.name} ~ Deterministic"

    def _repr_latex_(self, *, formatting="latex_with_params", **kwargs):
        return self._str_repr(formatting=formatting)

    __latex__ = _repr_latex_

    def __str__(self):
        return self._str_repr(formatting="plain")


def Deterministic(name, var, model=None, dims=None):
    """Create a named deterministic variable

    Parameters
    ----------
    name: str
    var: theano variables

    Returns
    -------
    var: var, with name attribute
    """
    model = modelcontext(model)
    var = var.copy(model.name_for(name))
    model.deterministics.append(var)
    model.add_random_variable(var, dims)
    var.__class__ = DeterministicWrapper  # adds str and latex functionality

    return var


def Potential(name, var, model=None):
    """Add an arbitrary factor potential to the model likelihood

    Parameters
    ----------
    name: str
    var: theano variables

    Returns
    -------
    var: var, with name attribute
    """
    model = modelcontext(model)
    var.name = model.name_for(name)
    model.potentials.append(var)
    model.add_random_variable(var)
    return var


class TransformedRV(PyMC3Variable):
    """
    Parameters
    ----------

    type: theano type (optional)
    owner: theano owner (optional)
    name: str
    distribution: Distribution
    model: Model
    total_size: scalar Tensor (optional)
        needed for upscaling logp
    """

    def __init__(
        self,
        type=None,
        owner=None,
        index=None,
        name=None,
        distribution=None,
        model=None,
        transform=None,
        total_size=None,
    ):
        if type is None:
            type = distribution.type
        super().__init__(type, owner, index, name)

        self.transformation = transform

        if distribution is not None:
            self.model = model
            self.distribution = distribution
            self.dshape = tuple(distribution.shape)
            self.dsize = int(np.prod(distribution.shape))

            transformed_name = get_transformed_name(name, transform)

            self.transformed = model.Var(
                transformed_name, transform.apply(distribution), total_size=total_size
            )

            normalRV = transform.backward(self.transformed)

            Apply(theano.compile.view_op, inputs=[normalRV], outputs=[self])
            self.tag.test_value = normalRV.tag.test_value
            self.scaling = _get_scaling(total_size, self.shape, self.ndim)
            incorporate_methods(
                source=distribution,
                destination=self,
                methods=["random"],
                wrapper=InstanceMethod,
            )

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    else:
        return [data]


def all_continuous(vars):
    """Check that vars not include discrete variables or BART variables, excepting ObservedRVs."""

    vars_ = [var for var in vars if not isinstance(var, pm.model.ObservedRV)]
    if any(
        [(var.dtype in pm.discrete_types or isinstance(var.distribution, pm.BART)) for var in vars_]
    ):
        return False
    else:
        return True
