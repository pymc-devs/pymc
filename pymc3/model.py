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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union

import aesara
import aesara.graph.basic
import aesara.sparse as sparse
import aesara.tensor as at
import numpy as np
import scipy.sparse as sps

from aesara.compile.sharedvalue import SharedVariable
from aesara.gradient import grad
from aesara.graph.basic import Constant, Variable, graph_inputs
from aesara.graph.fg import FunctionGraph, MissingInputError
from aesara.tensor.random.opt import local_subtensor_rv_lift
from aesara.tensor.var import TensorVariable
from pandas import Series

from pymc3.aesaraf import (
    change_rv_size,
    gradient,
    hessian,
    inputvars,
    pandas_to_array,
    rvs_to_value_vars,
)
from pymc3.blocking import DictToArrayBijection, RaveledVars
from pymc3.data import GenTensorVariable, Minibatch
from pymc3.distributions import logp_transform, logpt, logpt_sum
from pymc3.exceptions import ImputationWarning, SamplingError
from pymc3.math import flatten_list
from pymc3.util import UNSET, WithMemoization, get_var_name, treedict, treelist
from pymc3.vartypes import continuous_types, discrete_types, typefilter

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

FlatView = collections.namedtuple("FlatView", "input, replacements")


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


T = TypeVar("T", bound="ContextMeta")


class ContextMeta(type):
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """

    def __new__(cls, name, bases, dct, **kargs):  # pylint: disable=unused-argument
        "Add __enter__ and __exit__ methods to the class."

        def __enter__(self):
            self.__class__.context_class.get_contexts().append(self)
            # self._aesara_config is set in Model.__new__
            self._config_context = None
            if hasattr(self, "_aesara_config"):
                self._config_context = aesara.config.change_flags(**self._aesara_config)
                self._config_context.__enter__()
            return self

        def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
            self.__class__.context_class.get_contexts().pop()
            # self._aesara_config is set in Model.__new__
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
        """Aesara scalar of log-probability of the model"""
        if getattr(self, "total_size", None) is not None:
            logp = self.logp_sum_unscaledt * self.scaling
        else:
            logp = self.logp_sum_unscaledt
        if self.name is not None:
            logp.name = "__logp_%s" % self.name
        return logp

    @property
    def logp_nojact(self):
        """Aesara scalar of log-probability, excluding jacobian terms."""
        if getattr(self, "total_size", None) is not None:
            logp = at.sum(self.logp_nojac_unscaledt) * self.scaling
        else:
            logp = at.sum(self.logp_nojac_unscaledt)
        if self.name is not None:
            logp.name = "__logp_%s" % self.name
        return logp


class ValueGradFunction:
    """Create an Aesara function that computes a value and its gradient.

    Parameters
    ----------
    costs: list of Aesara variables
        We compute the weighted sum of the specified Aesara values, and the gradient
        of that sum. The weights can be specified with `ValueGradFunction.set_weights`.
    grad_vars: list of named Aesara variables or None
        The arguments with respect to which the gradient is computed.
    extra_vars_and_values: dict of Aesara variables and their initial values
        Other arguments of the function that are assumed constant and their
        values. They are stored in shared variables and can be set using
        `set_extra_values`.
    dtype: str, default=aesara.config.floatX
        The dtype of the arrays.
    casting: {'no', 'equiv', 'save', 'same_kind', 'unsafe'}, default='no'
        Casting rule for casting `grad_args` to the array dtype.
        See `numpy.can_cast` for a description of the options.
        Keep in mind that we cast the variables to the array *and*
        back from the array dtype to the variable dtype.
    compute_grads: bool, default=True
        If False, return only the logp, not the gradient.
    kwargs
        Extra arguments are passed on to `aesara.function`.

    Attributes
    ----------
    profile: Aesara profiling object or None
        The profiling object of the Aesara function that computes value and
        gradient. This is None unless `profile=True` was set in the
        kwargs.
    """

    def __init__(
        self,
        costs,
        grad_vars,
        extra_vars_and_values=None,
        *,
        dtype=None,
        casting="no",
        compute_grads=True,
        **kwargs,
    ):
        if extra_vars_and_values is None:
            extra_vars_and_values = {}

        names = [arg.name for arg in grad_vars + list(extra_vars_and_values.keys())]
        if any(name is None for name in names):
            raise ValueError("Arguments must be named.")
        if len(set(names)) != len(names):
            raise ValueError("Names of the arguments are not unique.")

        self._grad_vars = grad_vars
        self._extra_vars = list(extra_vars_and_values.keys())
        self._extra_var_names = {var.name for var in extra_vars_and_values.keys()}

        if dtype is None:
            dtype = aesara.config.floatX
        self.dtype = dtype

        self._n_costs = len(costs)
        if self._n_costs == 0:
            raise ValueError("At least one cost is required.")
        weights = np.ones(self._n_costs - 1, dtype=self.dtype)
        self._weights = aesara.shared(weights, "__weights")

        cost = costs[0]
        for i, val in enumerate(costs[1:]):
            if cost.ndim > 0 or val.ndim > 0:
                raise ValueError("All costs must be scalar.")
            cost = cost + self._weights[i] * val

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
        for var, value in extra_vars_and_values.items():
            shared = aesara.shared(
                value, var.name + "_shared__", broadcastable=[s == 1 for s in value.shape]
            )
            self._extra_vars_shared[var.name] = shared
            givens.append((var, shared))

        if compute_grads:
            grads = grad(cost, grad_vars, disconnected_inputs="ignore")
            for grad_wrt, var in zip(grads, grad_vars):
                grad_wrt.name = f"{var.name}_grad"
            outputs = [cost] + grads
        else:
            outputs = [cost]

        inputs = grad_vars

        self._aesara_function = aesara.function(inputs, outputs, givens=givens, **kwargs)

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

    def __call__(self, grad_vars, grad_out=None, extra_vars=None):
        if extra_vars is not None:
            self.set_extra_values(extra_vars)

        if not self._extra_are_set:
            raise ValueError("Extra values are not set.")

        if isinstance(grad_vars, RaveledVars):
            grad_vars = list(DictToArrayBijection.rmap(grad_vars).values())

        cost, *grads = self._aesara_function(*grad_vars)

        if grads:
            grads_raveled = DictToArrayBijection.map(
                {v.name: gv for v, gv in zip(self._grad_vars, grads)}
            )

            if grad_out is None:
                return cost, grads_raveled.data
            else:
                np.copyto(grad_out, grads_raveled.data)
                return cost
        else:
            return cost

    @property
    def profile(self):
        """Profiling information of the underlying Aesara function."""
        return self._aesara_function.profile


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
    aesara_config: dict
        A dictionary of Aesara config values that should be set
        temporarily in the model context. See the documentation
        of Aesara for a complete list.
    check_bounds: bool
        Ensure that input parameters to distributions are in a valid
        range. If your model is built in a way where you know your
        parameters can only take on valid values you can set this to
        False for increased speed. This should not be used if your model
        contains discrete variables.

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
                Potential('p1', at.constant(1))

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
        instance._aesara_config = kwargs.get("aesara_config", {})
        return instance

    def __init__(self, name="", model=None, aesara_config=None, coords=None, check_bounds=True):
        self.name = name
        self.coords = {}
        self.RV_dims = {}
        self.add_coords(coords)
        self.check_bounds = check_bounds

        self.default_rng = aesara.shared(np.random.RandomState(), name="default_rng", borrow=True)
        self.default_rng.tag.is_rng = True
        self.default_rng.default_update = self.default_rng

        if self.parent is not None:
            self.named_vars = treedict(parent=self.parent.named_vars)
            self.values_to_rvs = treedict(parent=self.parent.values_to_rvs)
            self.rvs_to_values = treedict(parent=self.parent.rvs_to_values)
            self.free_RVs = treelist(parent=self.parent.free_RVs)
            self.observed_RVs = treelist(parent=self.parent.observed_RVs)
            self.auto_deterministics = treelist(parent=self.parent.auto_deterministics)
            self.deterministics = treelist(parent=self.parent.deterministics)
            self.potentials = treelist(parent=self.parent.potentials)
        else:
            self.named_vars = treedict()
            self.values_to_rvs = treedict()
            self.rvs_to_values = treedict()
            self.free_RVs = treelist()
            self.observed_RVs = treelist()
            self.auto_deterministics = treelist()
            self.deterministics = treelist()
            self.potentials = treelist()

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

    @property
    def ndim(self):
        return sum(var.ndim for var in self.value_vars)

    def logp_dlogp_function(self, grad_vars=None, tempered=False, **kwargs):
        """Compile an Aesara function that computes logp and gradient.

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
            grad_vars = [v.tag.value_var for v in typefilter(self.free_RVs, continuous_types)]
        else:
            for i, var in enumerate(grad_vars):
                if var.dtype not in continuous_types:
                    raise ValueError("Can only compute the gradient of continuous types: %s" % var)
                # We allow one to pass the random variable terms as arguments
                if hasattr(var.tag, "value_var"):
                    grad_vars[i] = var.tag.value_var

        if tempered:
            with self:
                # Convert random variables into their log-likelihood inputs and
                # apply their transforms, if any
                potentials, _ = rvs_to_value_vars(self.potentials, apply_transforms=True)

                free_RVs_logp = at.sum(
                    [at.sum(logpt(var, self.rvs_to_values.get(var, None))) for var in self.free_RVs]
                    + list(potentials)
                )
                observed_RVs_logp = at.sum(
                    [at.sum(logpt(obs, obs.tag.observations)) for obs in self.observed_RVs]
                )

            costs = [free_RVs_logp, observed_RVs_logp]
        else:
            costs = [self.logpt]

        input_vars = {i for i in graph_inputs(costs) if not isinstance(i, Constant)}
        extra_vars = [self.rvs_to_values.get(var, var) for var in self.free_RVs]
        extra_vars_and_values = {
            var: self.initial_point[var.name]
            for var in extra_vars
            if var in input_vars and var not in grad_vars
        }
        return ValueGradFunction(costs, grad_vars, extra_vars_and_values, **kwargs)

    @property
    def logpt(self):
        """Aesara scalar of log-probability of the model"""
        with self:
            factors = [logpt_sum(var, self.rvs_to_values.get(var, None)) for var in self.free_RVs]
            factors += [logpt_sum(obs, obs.tag.observations) for obs in self.observed_RVs]

            # Convert random variables into their log-likelihood inputs and
            # apply their transforms, if any
            potentials, _ = rvs_to_value_vars(self.potentials, apply_transforms=True)

            factors += potentials

            logp_var = at.sum([at.sum(factor) for factor in factors])
            if self.name:
                logp_var.name = "__logp_%s" % self.name
            else:
                logp_var.name = "__logp"
            return logp_var

    @property
    def logp_nojact(self):
        """Aesara scalar of log-probability of the model but without the jacobian
        if transformed Random Variable is presented.

        Note that if there is no transformed variable in the model, logp_nojact
        will be the same as logpt as there is no need for Jacobian correction.
        """
        with self:
            factors = [
                logpt_sum(var, getattr(var.tag, "value_var", None), jacobian=False)
                for var in self.free_RVs
            ]
            factors += [
                logpt_sum(obs, obs.tag.observations, jacobian=False) for obs in self.observed_RVs
            ]

            # Convert random variables into their log-likelihood inputs and
            # apply their transforms, if any
            potentials, _ = rvs_to_value_vars(self.potentials, apply_transforms=True)
            factors += potentials

            logp_var = at.sum([at.sum(factor) for factor in factors])

            if self.name:
                logp_var.name = "__logp_nojac_%s" % self.name
            else:
                logp_var.name = "__logp_nojac"
            return logp_var

    @property
    def varlogpt(self):
        """Aesara scalar of log-probability of the unobserved random variables
        (excluding deterministic)."""
        with self:
            factors = [logpt_sum(var, getattr(var.tag, "value_var", None)) for var in self.free_RVs]
            return at.sum(factors)

    @property
    def datalogpt(self):
        with self:
            factors = [logpt(obs, obs.tag.observations) for obs in self.observed_RVs]

            # Convert random variables into their log-likelihood inputs and
            # apply their transforms, if any
            potentials, _ = rvs_to_value_vars(self.potentials, apply_transforms=True)

            factors += [at.sum(factor) for factor in potentials]
            return at.sum(factors)

    @property
    def vars(self):
        warnings.warn(
            "Model.vars has been deprecated. Use Model.value_vars instead.",
            DeprecationWarning,
        )
        return self.value_vars

    @property
    def value_vars(self):
        """List of unobserved random variables used as inputs to the model's
        log-likelihood (which excludes deterministics).
        """
        return [self.rvs_to_values[v] for v in self.free_RVs]

    @property
    def unobserved_value_vars(self):
        """List of all random variables (including untransformed projections),
        as well as deterministics used as inputs and outputs of the the model's
        log-likelihood graph
        """
        vars = []
        for rv in self.free_RVs:
            value_var = self.rvs_to_values[rv]
            transform = getattr(value_var.tag, "transform", None)
            if transform is not None:
                # We need to create and add an un-transformed version of
                # each transformed variable
                untrans_value_var = transform.backward(rv, value_var)
                untrans_value_var.name = rv.name
                vars.append(untrans_value_var)
            vars.append(value_var)

        # Remove rvs from deterministics graph
        deterministics, _ = rvs_to_value_vars(self.deterministics, apply_transforms=True)

        return vars + deterministics

    @property
    def basic_RVs(self):
        """List of random variables the model is defined in terms of
        (which excludes deterministics).

        These are the actual random variable terms that make up the
        "sample-space" graph (i.e. you can sample these graphs by compiling them
        with `aesara.function`).  If you want the corresponding log-likelihood terms,
        use `var.tag.value_var`.
        """
        return self.free_RVs + self.observed_RVs

    @property
    def unobserved_RVs(self):
        """List of all random variables, including deterministic ones.

        These are the actual random variable terms that make up the
        "sample-space" graph (i.e. you can sample these graphs by compiling them
        with `aesara.function`).  If you want the corresponding log-likelihood terms,
        use `var.tag.value_var`.
        """
        return self.free_RVs + self.deterministics

    @property
    def independent_vars(self):
        """List of all variables that are non-stochastic inputs to the model.

        These are the actual random variable terms that make up the
        "sample-space" graph (i.e. you can sample these graphs by compiling them
        with `aesara.function`).  If you want the corresponding log-likelihood terms,
        use `var.tag.value_var`.
        """
        return inputvars(self.unobserved_RVs)

    @property
    def test_point(self):
        warnings.warn(
            "`Model.test_point` has been deprecated. Use `Model.initial_point` instead.",
            DeprecationWarning,
        )
        return self.initial_point

    @property
    def initial_point(self):
        points = []
        for rv_var in self.free_RVs:
            value_var = rv_var.tag.value_var
            var_value = getattr(value_var.tag, "test_value", None)

            if var_value is None:

                rv_var_value = getattr(rv_var.tag, "test_value", None)

                if rv_var_value is None:
                    try:
                        rv_var_value = rv_var.eval()
                    except MissingInputError:
                        raise MissingInputError(f"Couldn't generate an initial value for {rv_var}")

                transform = getattr(value_var.tag, "transform", None)

                if transform:
                    try:
                        rv_var_value = transform.forward(rv_var, rv_var_value).eval()
                    except MissingInputError:
                        raise MissingInputError(f"Couldn't generate an initial value for {rv_var}")

                var_value = rv_var_value
                value_var.tag.test_value = var_value

            points.append((value_var, var_value))

        return Point(points, model=self)

    @property
    def disc_vars(self):
        """All the discrete variables in the model"""
        return list(typefilter(self.value_vars, discrete_types))

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.value_vars, continuous_types))

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

    def register_rv(self, rv_var, name, data=None, total_size=None, dims=None, transform=UNSET):
        """Register an (un)observed random variable with the model.

        Parameters
        ----------
        rv_var: TensorVariable
        name: str
        data: array_like (optional)
            If data is provided, the variable is observed. If None,
            the variable is unobserved.
        total_size: scalar
            upscales logp of variable with ``coef = total_size/var.shape[0]``
        dims: tuple
            Dimension names for the variable.
        transform
            A transform for the random variable in log-likelihood space.

        Returns
        -------
        TensorVariable
        """
        name = self.name_for(name)
        rv_var.name = name
        rv_var.tag.total_size = total_size

        if data is None:
            self.free_RVs.append(rv_var)
            self.create_value_var(rv_var, transform)
            self.add_random_variable(rv_var, dims)
        else:
            if (
                isinstance(data, Variable)
                and not isinstance(data, (GenTensorVariable, Minibatch))
                and data.owner is not None
            ):
                raise TypeError("Observed data cannot consist of symbolic variables.")

            data = pandas_to_array(data)

            # `rv_var` is potentially a new variable (e.g. the original
            # variable could have its size changed to match the data, or be a
            # new graph that accounts for missing data)
            rv_var = self.make_obs_var(rv_var, data, dims, transform)

        return rv_var

    def make_obs_var(
        self, rv_var: TensorVariable, data: np.ndarray, dims, transform: Optional[Any]
    ) -> TensorVariable:
        """Create a `TensorVariable` for an observed random variable.

        Parameters
        ==========
        rv_var
            The random variable that is observed.
        data
            The observed data.
        dims: tuple
            Dimension names for the variable.
        transform
            A transform for the random variable in log-likelihood space.

        """
        name = rv_var.name
        data = pandas_to_array(data).astype(rv_var.dtype)

        # The shapes of the observed random variable and its data might not
        # match.  We need need to update the observed random variable's `size`
        # (i.e. number of samples) so that it matches the data.

        # Setting `size` produces a random variable with shape `size +
        # support_shape`, where `len(support_shape) == op.ndim_supp`, we need
        # to disregard the last `op.ndim_supp`-many dimensions when we
        # determine the appropriate `size` value from `data.shape`.
        ndim_supp = rv_var.owner.op.ndim_supp
        if ndim_supp > 0:
            new_size = data.shape[:-ndim_supp]
        else:
            new_size = data.shape

        rv_var = change_rv_size(rv_var, new_size)

        if aesara.config.compute_test_value != "off":
            test_value = getattr(rv_var.tag, "test_value", None)

            if test_value is not None:
                # We try to reuse the old test value
                rv_var.tag.test_value = np.broadcast_to(test_value, rv_var.tag.test_value.shape)
            else:
                rv_var.tag.test_value = data

        mask = getattr(data, "mask", None)
        if mask is not None:

            if mask.all():
                # If there are no observed values, this variable isn't really
                # observed.
                return rv_var

            impute_message = (
                f"Data in {rv_var} contains missing values and"
                " will be automatically imputed from the"
                " sampling distribution."
            )
            warnings.warn(impute_message, ImputationWarning)

            # We can get a random variable comprised of only the unobserved
            # entries by lifting the indices through the `RandomVariable` `Op`.

            masked_rv_var = rv_var[mask.nonzero()]

            fgraph = FunctionGraph(
                [i for i in graph_inputs((masked_rv_var,)) if not isinstance(i, Constant)],
                [masked_rv_var],
                clone=False,
            )

            (missing_rv_var,) = local_subtensor_rv_lift.transform(fgraph, fgraph.outputs[0].owner)

            self.register_rv(missing_rv_var, f"{name}_missing", transform=transform)

            # Now, we lift the non-missing observed values and produce a new
            # `rv_var` that contains only those.
            #
            # The end result is two disjoint distributions: one for the missing
            # values, and another for the non-missing values.

            antimask_idx = (~mask).nonzero()
            nonmissing_data = at.as_tensor_variable(data[antimask_idx])
            unmasked_rv_var = rv_var[antimask_idx]
            unmasked_rv_var = unmasked_rv_var.owner.clone().default_output()

            fgraph = FunctionGraph(
                [i for i in graph_inputs((unmasked_rv_var,)) if not isinstance(i, Constant)],
                [unmasked_rv_var],
                clone=False,
            )
            (observed_rv_var,) = local_subtensor_rv_lift.transform(fgraph, fgraph.outputs[0].owner)
            observed_rv_var.name = f"{name}_observed"

            observed_rv_var.tag.observations = nonmissing_data

            self.create_value_var(observed_rv_var, transform)
            self.add_random_variable(observed_rv_var, dims)
            self.observed_RVs.append(observed_rv_var)

            # Create deterministic that combines observed and missing
            rv_var = at.zeros(data.shape)
            rv_var = at.set_subtensor(rv_var[mask.nonzero()], missing_rv_var)
            rv_var = at.set_subtensor(rv_var[antimask_idx], observed_rv_var)
            rv_var = Deterministic(name, rv_var, self, dims, auto=True)

        elif sps.issparse(data):
            data = sparse.basic.as_sparse(data, name=name)
            rv_var.tag.observations = data
            self.create_value_var(rv_var, transform)
            self.add_random_variable(rv_var, dims)
            self.observed_RVs.append(rv_var)
        else:
            data = at.as_tensor_variable(data, name=name)
            rv_var.tag.observations = data
            self.create_value_var(rv_var, transform)
            self.add_random_variable(rv_var, dims)
            self.observed_RVs.append(rv_var)

        return rv_var

    def create_value_var(self, rv_var: TensorVariable, transform: Any) -> TensorVariable:
        """Create a ``TensorVariable`` that will be used as the random
        variable's "value" in log-likelihood graphs.

        In general, we'll call this type of variable the "value" variable.

        In all other cases, the role of the value variable is taken by
        observed data. That's why value variables are only referenced in
        this branch of the conditional.

        """
        value_var = rv_var.type()

        if aesara.config.compute_test_value != "off":
            value_var.tag.test_value = rv_var.tag.test_value

        value_var.name = rv_var.name

        rv_var.tag.value_var = value_var

        # Make the value variable a transformed value variable,
        # if there's an applicable transform
        if transform is UNSET and rv_var.owner:
            transform = logp_transform(rv_var.owner.op)

        if transform is not None and transform is not UNSET:
            value_var.tag.transform = transform
            value_var.name = f"{value_var.name}_{transform.name}__"
            if aesara.config.compute_test_value != "off":
                value_var.tag.test_value = transform.forward(rv_var, value_var).tag.test_value
            self.named_vars[value_var.name] = value_var

        self.rvs_to_values[rv_var] = value_var
        self.values_to_rvs[value_var] = rv_var

        return value_var

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
        """Compiles an Aesara function which returns ``outs`` and takes the variable
        ancestors of ``outs`` as inputs.

        Parameters
        ----------
        outs: Aesara variable or iterable of Aesara variables
        mode: Aesara compilation mode

        Returns
        -------
        Compiled Aesara function
        """
        with self:
            return aesara.function(
                self.value_vars,
                outs,
                allow_input_downcast=True,
                on_unused_input="ignore",
                accept_inplace=True,
                mode=mode,
                *args,
                **kwargs,
            )

    def fn(self, outs, mode=None, *args, **kwargs):
        """Compiles an Aesara function which returns the values of ``outs``
        and takes values of model vars as arguments.

        Parameters
        ----------
        outs: Aesara variable or iterable of Aesara variables
        mode: Aesara compilation mode

        Returns
        -------
        Compiled Aesara function
        """
        return LoosePointFunc(self.makefn(outs, mode, *args, **kwargs), self)

    def fastfn(self, outs, mode=None, *args, **kwargs):
        """Compiles an Aesara function which returns ``outs`` and takes values
        of model vars as a dict as an argument.

        Parameters
        ----------
        outs: Aesara variable or iterable of Aesara variables
        mode: Aesara compilation mode

        Returns
        -------
        Compiled Aesara function as point function.
        """
        f = self.makefn(outs, mode, *args, **kwargs)
        return FastPointFunc(f)

    def profile(self, outs, n=1000, point=None, profile=True, *args, **kwargs):
        """Compiles and profiles an Aesara function which returns ``outs`` and
        takes values of model vars as a dict as an argument.

        Parameters
        ----------
        outs: Aesara variable or iterable of Aesara variables
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
            point = self.initial_point

        for _ in range(n):
            f(**point)

        return f.profile

    def flatten(self, vars=None, order=None, inputvar=None):
        """Flattens model's input and returns:

        Parameters
        ----------
        vars: list of variables or None
            if None, then all model.free_RVs are used for flattening input
        order: list of variable names
            Optional, use predefined ordering
        inputvar: at.vector
            Optional, use predefined inputvar

        Returns
        -------
        flat_view
        """
        if vars is None:
            vars = self.value_vars
        if order is not None:
            var_map = {v.name: v for v in vars}
            vars = [var_map[n] for n in order]

        if inputvar is None:
            inputvar = at.vector("flat_view", dtype=aesara.config.floatX)
            if aesara.config.compute_test_value != "off":
                if vars:
                    inputvar.tag.test_value = flatten_list(vars).tag.test_value
                else:
                    inputvar.tag.test_value = np.asarray([], inputvar.dtype)

        replacements = {}
        last_idx = 0
        for var in vars:
            arr_len = at.prod(var.shape, dtype="int64")
            replacements[self.named_vars[var.name]] = (
                inputvar[last_idx : (last_idx + arr_len)].reshape(var.shape).astype(var.dtype)
            )
            last_idx += arr_len

        flat_view = FlatView(inputvar, replacements)

        return flat_view

    def update_start_vals(self, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]):
        r"""Update point `a` with `b`, without overwriting existing keys.

        Values specified for transformed variables in `a` will be recomputed
        conditional on the valures of `b` and stored in `b`.

        """
        # TODO FIXME XXX: If we're going to incrementally update transformed
        # variables, we should do it in topological order.
        for a_name, a_value in tuple(a.items()):
            # If the name is a random variable, get its value variable and
            # potentially transform it
            var = self.named_vars.get(a_name, None)
            value_var = self.rvs_to_values.get(var, None)
            if value_var:
                transform = getattr(value_var.tag, "transform", None)
                if transform:
                    fval_graph = transform.forward(var, a_value)
                    (fval_graph,), _ = rvs_to_value_vars((fval_graph,), apply_transforms=True)
                    fval_graph_inputs = {i: b[i.name] for i in inputvars(fval_graph) if i.name in b}
                    rv_var_value = fval_graph.eval(fval_graph_inputs)
                    # Why are these transformed values stored in `b`?  They're
                    # not going to be used to update `a`.
                    b[value_var.name] = rv_var_value

        a.update({k: v for k, v in b.items() if k not in a})

    def check_start_vals(self, start):
        r"""Check that the starting values for MCMC do not cause the relevant log probability
        to evaluate to something invalid (e.g. Inf or NaN)

        Parameters
        ----------
        start : dict, or array of dict
            Starting point in parameter space (or partial point)
            Defaults to ``trace.point(-1))`` if there is a trace provided and
            ``model.initial_point`` if not (defaults to empty dict). Initialization
            methods for NUTS (see ``init`` keyword) can overwrite the default.

        Raises
        ------
        ``KeyError`` if the parameters provided by `start` do not agree with the
        parameters contained within the model.

        ``pymc3.exceptions.SamplingError`` if the evaluation of the parameters
        in ``start`` leads to an invalid (i.e. non-finite) state

        Returns
        -------
        None
        """
        start_points = [start] if isinstance(start, dict) else start
        for elem in start_points:

            for k, v in elem.items():
                elem[k] = np.asarray(v, dtype=self[k].dtype)

            if not set(elem.keys()).issubset(self.named_vars.keys()):
                extra_keys = ", ".join(set(elem.keys()) - set(self.named_vars.keys()))
                valid_keys = ", ".join(self.named_vars.keys())
                raise KeyError(
                    "Some start parameters do not appear in the model!\n"
                    "Valid keys are: {}, but {} was supplied".format(valid_keys, extra_keys)
                )

            initial_eval = self.point_logps(point=elem)

            if not np.all(np.isfinite(initial_eval)):
                raise SamplingError(
                    "Initial evaluation of model at starting point failed!\n"
                    "Starting values:\n{}\n\n"
                    "Initial evaluation results:\n{}".format(elem, str(initial_eval))
                )

    def check_test_point(self, *args, **kwargs):
        warnings.warn(
            "`Model.check_test_point` has been deprecated. Use `Model.point_logps` instead.",
            DeprecationWarning,
        )
        return self.point_logps(*args, **kwargs)

    def point_logps(self, point=None, round_vals=2):
        """Computes the log probability of `point` for all random variables in the model.

        Parameters
        ----------
        point: Point
            Point to be evaluated.  If ``None``, then ``model.initial_point``
            is used.
        round_vals: int
            Number of decimals to round log-probabilities

        Returns
        -------
        Pandas Series
        """
        if point is None:
            point = self.initial_point

        return Series(
            {
                rv.name: np.round(
                    self.fn(logpt_sum(rv, getattr(rv.tag, "observations", None)))(point),
                    round_vals,
                )
                for rv in self.basic_RVs
            },
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
                rv_repr for rv_repr in rv_reprs if "TransformedDistribution()" not in rv_repr
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
    """Compiles an Aesara function which returns the values of ``outs`` and
    takes values of model vars as arguments.

    Parameters
    ----------
    outs: Aesara variable or iterable of Aesara variables
    mode: Aesara compilation mode

    Returns
    -------
    Compiled Aesara function
    """
    model = modelcontext(model)
    return model.fn(outs, mode, *args, **kwargs)


def fastfn(outs, mode=None, model=None):
    """Compiles an Aesara function which returns ``outs`` and takes values of model
    vars as a dict as an argument.

    Parameters
    ----------
    outs: Aesara variable or iterable of Aesara variables
    mode: Aesara compilation mode

    Returns
    -------
    Compiled Aesara function as point function.
    """
    model = modelcontext(model)
    return model.fastfn(outs, mode)


def Point(*args, filter_model_vars=False, **kwargs):
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
        if not filter_model_vars or (get_var_name(k) in map(get_var_name, model.value_vars))
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
        point = Point(model=self.model, *args, filter_model_vars=True, **kwargs)
        return self.f(**point)


compilef = fastfn


def Deterministic(name, var, model=None, dims=None, auto=False):
    """Create a named deterministic variable

    Parameters
    ----------
    name: str
    var: Aesara variables
    auto: bool
        Add automatically created deterministics (e.g., when imputing missing values)
        to a separate model.auto_deterministics list for filtering during sampling.


    Returns
    -------
    var: var, with name attribute
    """
    model = modelcontext(model)
    var = var.copy(model.name_for(name))
    if auto:
        model.auto_deterministics.append(var)
    else:
        model.deterministics.append(var)
    model.add_random_variable(var, dims)

    return var


def Potential(name, var, model=None):
    """Add an arbitrary factor potential to the model likelihood

    Parameters
    ----------
    name: str
    var: Aesara variables

    Returns
    -------
    var: var, with name attribute
    """
    model = modelcontext(model)
    var.name = model.name_for(name)
    var.tag.scaling = None
    model.potentials.append(var)
    model.add_random_variable(var)
    return var
