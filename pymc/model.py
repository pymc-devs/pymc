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
import functools
import threading
import types
import warnings

from sys import modules
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import aesara
import aesara.sparse as sparse
import aesara.tensor as at
import numpy as np
import scipy.sparse as sps

from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant, Variable, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.tensor.random.opt import local_subtensor_rv_lift
from aesara.tensor.random.var import RandomStateSharedVariable
from aesara.tensor.sharedvar import ScalarSharedVariable
from aesara.tensor.var import TensorVariable

from pymc.aesaraf import (
    compile_pymc,
    gradient,
    hessian,
    inputvars,
    pandas_to_array,
    rvs_to_value_vars,
)
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.data import GenTensorVariable, Minibatch
from pymc.distributions import joint_logpt
from pymc.distributions.logprob import _get_scaling
from pymc.distributions.transforms import _default_transform
from pymc.exceptions import ImputationWarning, SamplingError, ShapeError
from pymc.initial_point import make_initial_point_fn
from pymc.math import flatten_list
from pymc.util import (
    UNSET,
    WithMemoization,
    get_transformed_name,
    get_var_name,
    treedict,
    treelist,
)
from pymc.vartypes import continuous_types, discrete_types, typefilter

__all__ = [
    "Model",
    "modelcontext",
    "Deterministic",
    "Potential",
    "set_data",
    "Point",
    "compile_fn",
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

    def __new__(cls, name, bases, dct, **kwargs):  # pylint: disable=unused-argument
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
        # DO NOT send "**kwargs" to "type.__new__".  It won't catch them and
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
                raise TypeError(f"No {cls} on context stack")
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
        assert isinstance(
            context_class, type
        ), f"Name of context class, {context_class} was not resolvable to a class"
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
            raise ValueError(f"Cannot resolve context class {c}")

        assert cls is not None
        if isinstance(cls._context_class, str):
            cls._context_class = resolve_type(cls._context_class)
        if not isinstance(cls._context_class, (str, type)):
            raise ValueError(
                f"Context class for {cls.__name__}, {cls._context_class}, is not of the right type"
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
            grads = aesara.grad(cost, grad_vars, disconnected_inputs="ignore")
            for grad_wrt, var in zip(grads, grad_vars):
                grad_wrt.name = f"{var.name}_grad"
            outputs = [cost] + grads
        else:
            outputs = [cost]

        inputs = grad_vars

        self._aesara_function = compile_pymc(inputs, outputs, givens=givens, **kwargs)

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


class Model(WithMemoization, metaclass=ContextMeta):
    """Encapsulates the variables and likelihood factors of a model.

    Model class can be used for creating class based models. To create
    a class based model you should inherit from :class:`~pymc.Model` and
    override the `__init__` method with arbitrary definitions (do not
    forget to call base class :meth:`pymc.Model.__init__` first).

    Parameters
    ----------
    name: str
        name that will be used as prefix for names of all random
        variables defined within model
    check_bounds: bool
        Ensure that input parameters to distributions are in a valid
        range. If your model is built in a way where you know your
        parameters can only take on valid values you can set this to
        False for increased speed. This should not be used if your model
        contains discrete variables.
    rng_seeder: int or numpy.random.RandomState
        The ``numpy.random.RandomState`` used to seed the
        ``RandomStateSharedVariable`` sequence used by a model
        ``RandomVariable``s, or an int used to seed a new
        ``numpy.random.RandomState``.  If ``None``, a
        ``RandomStateSharedVariable`` will be generated and used.  Incremental
        access to the state sequence is provided by ``Model.next_rng``.

    Examples
    --------

    How to define a custom model

    .. code-block:: python

        class CustomModel(Model):
            # 1) override init
            def __init__(self, mean=0, sigma=1, name=''):
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
                # this will create variable named like '{prefix::}v1'
                # and assign attribute 'v1' to instance created
                # variable can be accessed with self.v1 or self['v1']

                # 4) this syntax will also work as we are in the
                # context of instance itself, names are given as usual
                Normal('v2', mu=mean, sigma=sd)

                # something more complex is allowed, too
                half_cauchy = HalfCauchy('sigma', beta=10, initval=1.)
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

        # variables inside both scopes will be named like `first::*`, `second::*`

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

    @staticmethod
    def _validate_name(name):
        if name.endswith(":"):
            raise KeyError("name should not end with `:`")
        return name

    def __init__(
        self,
        name="",
        coords=None,
        check_bounds=True,
        rng_seeder: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.name = self._validate_name(name)
        self.check_bounds = check_bounds

        if rng_seeder is None:
            self.rng_seeder = np.random.RandomState()
        elif isinstance(rng_seeder, int):
            self.rng_seeder = np.random.RandomState(rng_seeder)
        else:
            self.rng_seeder = rng_seeder

        # The sequence of model-generated RNGs
        self.rng_seq: List[SharedVariable] = []
        self._initial_values: Dict[TensorVariable, Optional[Union[np.ndarray, Variable, str]]] = {}

        if self.parent is not None:
            self.named_vars = treedict(parent=self.parent.named_vars)
            self.values_to_rvs = treedict(parent=self.parent.values_to_rvs)
            self.rvs_to_values = treedict(parent=self.parent.rvs_to_values)
            self.free_RVs = treelist(parent=self.parent.free_RVs)
            self.observed_RVs = treelist(parent=self.parent.observed_RVs)
            self.auto_deterministics = treelist(parent=self.parent.auto_deterministics)
            self.deterministics = treelist(parent=self.parent.deterministics)
            self.potentials = treelist(parent=self.parent.potentials)
            self._coords = self.parent._coords
            self._RV_dims = treedict(parent=self.parent._RV_dims)
            self._dim_lengths = self.parent._dim_lengths
        else:
            self.named_vars = treedict()
            self.values_to_rvs = treedict()
            self.rvs_to_values = treedict()
            self.free_RVs = treelist()
            self.observed_RVs = treelist()
            self.auto_deterministics = treelist()
            self.deterministics = treelist()
            self.potentials = treelist()
            self._coords = {}
            self._RV_dims = treedict()
            self._dim_lengths = {}
        self.add_coords(coords)

        from pymc.printing import str_for_model

        self.str_repr = types.MethodType(str_for_model, self)
        self._repr_latex_ = types.MethodType(
            functools.partial(str_for_model, formatting="latex"), self
        )

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
            grad_vars = [self.rvs_to_values[v] for v in typefilter(self.free_RVs, continuous_types)]
        else:
            for i, var in enumerate(grad_vars):
                if var.dtype not in continuous_types:
                    raise ValueError(f"Can only compute the gradient of continuous types: {var}")

        if tempered:
            costs = [self.varlogpt, self.datalogpt]
        else:
            costs = [self.logpt()]

        input_vars = {i for i in graph_inputs(costs) if not isinstance(i, Constant)}
        extra_vars = [self.rvs_to_values.get(var, var) for var in self.free_RVs]
        ip = self.compute_initial_point(0)
        extra_vars_and_values = {
            var: ip[var.name] for var in extra_vars if var in input_vars and var not in grad_vars
        }
        return ValueGradFunction(costs, grad_vars, extra_vars_and_values, **kwargs)

    def compile_logp(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
        sum: bool = True,
    ):
        """Compiled log probability density function.

        Parameters
        ----------
        vars: list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian:
            Whether to include jacobian terms in logprob graph. Defaults to True.
        sum:
            Whether to sum all logp terms or return elemwise logp for each variable.
            Defaults to True.
        """
        return self.model.compile_fn(self.logpt(vars=vars, jacobian=jacobian, sum=sum))

    def compile_dlogp(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
    ):
        """Compiled log probability density gradient function.

        Parameters
        ----------
        vars: list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian:
            Whether to include jacobian terms in logprob graph. Defaults to True.
        """
        return self.model.compile_fn(self.dlogpt(vars=vars, jacobian=jacobian))

    def compile_d2logp(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
    ):
        """Compiled log probability density hessian function.

        Parameters
        ----------
        vars: list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian:
            Whether to include jacobian terms in logprob graph. Defaults to True.
        """
        return self.model.compile_fn(self.d2logpt(vars=vars, jacobian=jacobian))

    def logpt(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
        sum: bool = True,
    ) -> Union[Variable, List[Variable]]:
        """Elemwise log-probability of the model.

        Parameters
        ----------
        vars: list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian:
            Whether to include jacobian terms in logprob graph. Defaults to True.
        sum:
            Whether to sum all logp terms or return elemwise logp for each variable.
            Defaults to True.

        Returns
        -------
        Logp graph(s)
        """
        varlist: List[TensorVariable]
        if vars is None:
            varlist = self.free_RVs + self.observed_RVs + self.potentials
        elif not isinstance(vars, (list, tuple)):
            varlist = [vars]
        else:
            varlist = cast(List[TensorVariable], vars)

        # We need to separate random variables from potential terms, and remember their
        # original order so that we can merge them together in the same order at the end
        rv_values = {}
        potentials = []
        rv_order, potential_order = [], []
        for i, var in enumerate(varlist):
            value_var = self.rvs_to_values.get(var)
            if value_var is not None:
                rv_values[var] = value_var
                rv_order.append(i)
            else:
                if var in self.potentials:
                    potentials.append(var)
                    potential_order.append(i)
                else:
                    raise ValueError(
                        f"Requested variable {var} not found among the model variables"
                    )

        rv_logps: List[TensorVariable] = []
        if rv_values:
            rv_logps = joint_logpt(list(rv_values.keys()), rv_values, sum=False, jacobian=jacobian)
            assert isinstance(rv_logps, list)

        # Replace random variables by their value variables in potential terms
        potential_logps = []
        if potentials:
            potential_logps, _ = rvs_to_value_vars(potentials, apply_transforms=True)

        logp_factors = [None] * len(varlist)
        for logp_order, logp in zip((rv_order + potential_order), (rv_logps + potential_logps)):
            logp_factors[logp_order] = logp

        if not sum:
            return logp_factors

        logp_scalar = at.sum([at.sum(factor) for factor in logp_factors])
        logp_scalar_name = "__logp" if jacobian else "__logp_nojac"
        if self.name:
            logp_scalar_name = f"{logp_scalar_name}_{self.name}"
        logp_scalar.name = logp_scalar_name
        return logp_scalar

    def dlogpt(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
    ) -> Variable:
        """Gradient of the models log-probability w.r.t. ``vars``.

        Parameters
        ----------
        vars: list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian:
            Whether to include jacobian terms in logprob graph. Defaults to True.

        Returns
        -------
        dlogp graph
        """
        if vars is None:
            value_vars = None
        else:
            if not isinstance(vars, (list, tuple)):
                vars = [vars]

            value_vars = []
            for i, var in enumerate(vars):
                value_var = self.rvs_to_values.get(var)
                if value_var is not None:
                    value_vars.append(value_var)
                else:
                    raise ValueError(
                        f"Requested variable {var} not found among the model variables"
                    )

        cost = self.logpt(jacobian=jacobian)
        return gradient(cost, value_vars)

    def d2logpt(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
    ) -> Variable:
        """Hessian of the models log-probability w.r.t. ``vars``.

        Parameters
        ----------
        vars: list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian:
            Whether to include jacobian terms in logprob graph. Defaults to True.

        Returns
        -------
        d²logp graph
        """
        if vars is None:
            value_vars = None
        else:
            if not isinstance(vars, (list, tuple)):
                vars = [vars]

            value_vars = []
            for i, var in enumerate(vars):
                value_var = self.rvs_to_values.get(var)
                if value_var is not None:
                    value_vars.append(value_var)
                else:
                    raise ValueError(
                        f"Requested variable {var} not found among the model variables"
                    )

        cost = self.logpt(jacobian=jacobian)
        return hessian(cost, value_vars)

    @property
    def datalogpt(self) -> Variable:
        """Aesara scalar of log-probability of the observed variables and
        potential terms"""
        return self.observedlogpt + self.potentiallogpt

    @property
    def varlogpt(self) -> Variable:
        """Aesara scalar of log-probability of the unobserved random variables
        (excluding deterministic)."""
        return self.logpt(vars=self.free_RVs)

    @property
    def varlogp_nojact(self) -> Variable:
        """Aesara scalar of log-probability of the unobserved random variables
        (excluding deterministic) without jacobian term."""
        return self.logpt(vars=self.free_RVs, jacobian=False)

    @property
    def observedlogpt(self) -> Variable:
        """Aesara scalar of log-probability of the observed variables"""
        return self.logpt(vars=self.observed_RVs)

    @property
    def potentiallogpt(self) -> Variable:
        """Aesara scalar of log-probability of the Potential terms"""
        # Convert random variables in Potential expression into their log-likelihood
        # inputs and apply their transforms, if any
        potentials, _ = rvs_to_value_vars(self.potentials, apply_transforms=True)
        if potentials:
            return at.sum([at.sum(factor) for factor in potentials])
        else:
            return at.constant(0.0)

    @property
    def vars(self):
        warnings.warn(
            "Model.vars has been deprecated. Use Model.value_vars instead.",
            FutureWarning,
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
        as well as deterministics used as inputs and outputs of the model's
        log-likelihood graph
        """
        vars = []
        untransformed_vars = []
        for rv in self.free_RVs:
            value_var = self.rvs_to_values[rv]
            transform = getattr(value_var.tag, "transform", None)
            if transform is not None:
                # We need to create and add an un-transformed version of
                # each transformed variable
                untrans_value_var = transform.backward(value_var, *rv.owner.inputs)
                untrans_value_var.name = rv.name
                untransformed_vars.append(untrans_value_var)
            vars.append(value_var)

        # Remove rvs from untransformed values graph
        untransformed_vars, _ = rvs_to_value_vars(untransformed_vars, apply_transforms=True)

        # Remove rvs from deterministics graph
        deterministics, _ = rvs_to_value_vars(self.deterministics, apply_transforms=True)

        return vars + untransformed_vars + deterministics

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
    def RV_dims(self) -> Dict[str, Tuple[Union[str, None], ...]]:
        """Tuples of dimension names for specific model variables.

        Entries in the tuples may be ``None``, if the RV dimension was not given a name.
        """
        return self._RV_dims

    @property
    def coords(self) -> Dict[str, Union[Tuple, None]]:
        """Coordinate values for model dimensions."""
        return self._coords

    @property
    def dim_lengths(self) -> Dict[str, Variable]:
        """The symbolic lengths of dimensions in the model.

        The values are typically instances of ``TensorVariable`` or ``ScalarSharedVariable``.
        """
        return self._dim_lengths

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
    def disc_vars(self):
        """All the discrete variables in the model"""
        return list(typefilter(self.value_vars, discrete_types))

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.value_vars, continuous_types))

    @property
    def test_point(self) -> Dict[str, np.ndarray]:
        """Deprecated alias for `Model.compute_initial_point(seed=None)`."""
        warnings.warn(
            "`Model.test_point` has been deprecated. Use `Model.compute_initial_point(seed=None)`.",
            FutureWarning,
        )
        return self.compute_initial_point()

    @property
    def initial_point(self) -> Dict[str, np.ndarray]:
        """Deprecated alias for `Model.compute_initial_point(seed=None)`."""
        warnings.warn(
            "`Model.initial_point` has been deprecated. Use `Model.compute_initial_point(seed=None)`.",
            FutureWarning,
        )
        return self.compute_initial_point()

    def compute_initial_point(self, seed=None) -> Dict[str, np.ndarray]:
        """Computes the initial point of the model.

        Returns
        -------
        ip : dict
            Maps names of transformed variables to numeric initial values in the transformed space.
        """
        if seed is None:
            seed = self.rng_seeder.randint(2**30, dtype=np.int64)
        fn = make_initial_point_fn(model=self, return_transformed=True)
        return Point(fn(seed), model=self)

    @property
    def initial_values(self) -> Dict[TensorVariable, Optional[Union[np.ndarray, Variable, str]]]:
        """Maps transformed variables to initial value placeholders.

        Keys are the random variables (as returned by e.g. ``pm.Uniform()``) and
        values are the numeric/symbolic initial values, strings denoting the strategy to get them, or None.
        """
        return self._initial_values

    def set_initval(self, rv_var, initval):
        """Sets an initial value (strategy) for a random variable."""
        if initval is not None and not isinstance(initval, (Variable, str)):
            # Convert scalars or array-like inputs to ndarrays
            initval = rv_var.type.filter(initval)

        self.initial_values[rv_var] = initval

    def next_rng(self) -> RandomStateSharedVariable:
        """Generate a new ``RandomStateSharedVariable``.

        The new ``RandomStateSharedVariable`` is also added to
        ``Model.rng_seq``.
        """
        new_seed = self.rng_seeder.randint(2**30, dtype=np.int64)
        next_rng = aesara.shared(np.random.RandomState(new_seed), borrow=True)
        next_rng.tag.is_rng = True

        self.rng_seq.append(next_rng)

        return next_rng

    def shape_from_dims(self, dims):
        shape = []
        if len(set(dims)) != len(dims):
            raise ValueError("Can not contain the same dimension name twice.")
        for dim in dims:
            if dim not in self.coords:
                raise ValueError(
                    f"Unknown dimension name '{dim}'. All dimension "
                    "names must be specified in the `coords` "
                    "argument of the model or through a pm.Data "
                    "variable."
                )
            shape.extend(np.shape(self.coords[dim]))
        return tuple(shape)

    def add_coord(
        self,
        name: str,
        values: Optional[Sequence] = None,
        *,
        length: Optional[Variable] = None,
    ):
        """Registers a dimension coordinate with the model.

        Parameters
        ----------
        name : str
            Name of the dimension.
            Forbidden: {"chain", "draw", "__sample__"}
        values : optional, array-like
            Coordinate values or ``None`` (for auto-numbering).
            If ``None`` is passed, a ``length`` must be specified.
        length : optional, scalar
            A symbolic scalar of the dimensions length.
            Defaults to ``aesara.shared(len(values))``.
        """
        if name in {"draw", "chain", "__sample__"}:
            raise ValueError(
                "Dimensions can not be named `draw`, `chain` or `__sample__`, "
                "as those are reserved for use in `InferenceData`."
            )
        if values is None and length is None:
            raise ValueError(
                f"Either `values` or `length` must be specified for the '{name}' dimension."
            )
        if length is not None and not isinstance(length, Variable):
            raise ValueError(
                f"The `length` passed for the '{name}' coord must be an Aesara Variable or None."
            )
        if values is not None:
            # Conversion to a tuple ensures that the coordinate values are immutable.
            # Also unlike numpy arrays the's tuple.index(...) which is handy to work with.
            values = tuple(values)
        if name in self.coords:
            if not np.array_equal(values, self.coords[name]):
                raise ValueError(f"Duplicate and incompatible coordinate: {name}.")
        else:
            self._coords[name] = values
            self._dim_lengths[name] = length or aesara.shared(len(values))

    def add_coords(
        self,
        coords: Dict[str, Optional[Sequence]],
        *,
        lengths: Optional[Dict[str, Union[Variable, None]]] = None,
    ):
        """Vectorized version of ``Model.add_coord``."""
        if coords is None:
            return
        lengths = lengths or {}

        for name, values in coords.items():
            self.add_coord(name, values, length=lengths.get(name, None))

    def set_data(
        self,
        name: str,
        values: Dict[str, Optional[Sequence]],
        coords: Optional[Dict[str, Sequence]] = None,
    ):
        """Changes the values of a data variable in the model.

        In contrast to pm.MutableData().set_value, this method can also
        update the corresponding coordinates.

        Parameters
        ----------
        name : str
            Name of a shared variable in the model.
        values : array-like
            New values for the shared variable.
        coords : optional, dict
            New coordinate values for dimensions of the shared variable.
            Must be provided for all named dimensions that change in length
            and already have coordinate values.
        """
        shared_object = self[name]
        if not isinstance(shared_object, SharedVariable):
            raise TypeError(
                f"The variable `{name}` must be a `SharedVariable`"
                " (created through `pm.MutableData()` or `pm.Data(mutable=True)`) to allow updating. "
                f"The current type is: {type(shared_object)}"
            )

        if isinstance(values, list):
            values = np.array(values)
        values = pandas_to_array(values)
        dims = self.RV_dims.get(name, None) or ()
        coords = coords or {}

        if values.ndim != shared_object.ndim:
            raise ValueError(
                f"New values for '{name}' must have {shared_object.ndim} dimensions, just like the original."
            )

        for d, dname in enumerate(dims):
            length_tensor = self.dim_lengths[dname]
            old_length = length_tensor.eval()
            new_length = values.shape[d]
            original_coords = self.coords.get(dname, None)
            new_coords = coords.get(dname, None)

            length_changed = new_length != old_length

            # Reject resizing if we already know that it would create shape problems.
            # NOTE: If there are multiple pm.MutableData containers sharing this dim, but the user only
            #       changes the values for one of them, they will run into shape problems nonetheless.
            length_belongs_to = length_tensor.owner.inputs[0].owner.inputs[0]
            if not isinstance(length_belongs_to, SharedVariable) and length_changed:
                raise ShapeError(
                    f"Resizing dimension '{dname}' with values of length {new_length} would lead to incompatibilities, "
                    f"because the dimension was initialized from '{length_belongs_to}' which is not a shared variable. "
                    f"Check if the dimension was defined implicitly before the shared variable '{name}' was created, "
                    f"for example by a model variable.",
                    actual=new_length,
                    expected=old_length,
                )
            if original_coords is not None and length_changed:
                if length_changed and new_coords is None:
                    raise ValueError(
                        f"The '{name}' variable already had {len(original_coords)} coord values defined for"
                        f"its {dname} dimension. With the new values this dimension changes to length "
                        f"{new_length}, so new coord values for the {dname} dimension are required."
                    )
            if new_coords is not None:
                # Update the registered coord values (also if they were None)
                if len(new_coords) != new_length:
                    raise ShapeError(
                        f"Length of new coordinate values for dimension '{dname}' does not match the provided values.",
                        actual=len(new_coords),
                        expected=new_length,
                    )
                self._coords[dname] = new_coords
            if isinstance(length_tensor, ScalarSharedVariable) and new_length != old_length:
                # Updating the shared variable resizes dependent nodes that use this dimension for their `size`.
                length_tensor.set_value(new_length)

        shared_object.set_value(values)

    def register_rv(
        self, rv_var, name, data=None, total_size=None, dims=None, transform=UNSET, initval=None
    ):
        """Register an (un)observed random variable with the model.

        Parameters
        ----------
        rv_var: TensorVariable
        name: str
            Intended name for the model variable.
        data: array_like (optional)
            If data is provided, the variable is observed. If None,
            the variable is unobserved.
        total_size: scalar
            upscales logp of variable with ``coef = total_size/var.shape[0]``
        dims: tuple
            Dimension names for the variable.
        transform
            A transform for the random variable in log-likelihood space.
        initval
            The initial value of the random variable.

        Returns
        -------
        TensorVariable
        """
        name = self.name_for(name)
        rv_var.name = name
        rv_var.tag.total_size = total_size
        rv_var.tag.scaling = _get_scaling(total_size, shape=rv_var.shape, ndim=rv_var.ndim)

        # Associate previously unknown dimension names with
        # the length of the corresponding RV dimension.
        if dims is not None:
            for d, dname in enumerate(dims):
                if not dname in self.dim_lengths:
                    self.add_coord(dname, values=None, length=rv_var.shape[d])

        if data is None:
            self.free_RVs.append(rv_var)
            self.create_value_var(rv_var, transform)
            self.add_random_variable(rv_var, dims)
            self.set_initval(rv_var, initval)
        else:
            if (
                isinstance(data, Variable)
                and not isinstance(data, (GenTensorVariable, Minibatch))
                and data.owner is not None
            ):
                raise TypeError(
                    "Variables that depend on other nodes cannot be used for observed data."
                    f"The data variable was: {data}"
                )

            # `rv_var` is potentially changed by `make_obs_var`,
            # for example into a new graph for imputation of missing data.
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
            Its dimensionality must be compatible with the data already.
        data
            The observed data.
        dims: tuple
            Dimension names for the variable.
        transform
            A transform for the random variable in log-likelihood space.

        """
        name = rv_var.name
        data = pandas_to_array(data).astype(rv_var.dtype)

        if data.ndim != rv_var.ndim:
            raise ShapeError(
                "Dimensionality of data and RV don't match.", actual=data.ndim, expected=rv_var.ndim
            )

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

            if rv_var.owner.op.ndim_supp > 0:
                raise NotImplementedError(
                    f"Automatic inputation is only supported for univariate RandomVariables, but {rv_var} is multivariate"
                )

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
            # Make a clone of the RV, but change the rng so that observed and missing
            # are not treated as equivalent nodes by aesara. This would happen if the
            # size of the masked and unmasked array happened to coincide
            _, size, _, *inps = observed_rv_var.owner.inputs
            rng = self.model.next_rng()
            observed_rv_var = observed_rv_var.owner.op(*inps, size=size, rng=rng)
            # Add default_update to new rng
            new_rng = observed_rv_var.owner.outputs[0]
            observed_rv_var.update = (rng, new_rng)
            rng.default_update = new_rng
            observed_rv_var.name = f"{name}_observed"

            observed_rv_var.tag.observations = nonmissing_data

            self.create_value_var(observed_rv_var, transform=None, value_var=nonmissing_data)
            self.add_random_variable(observed_rv_var, dims)
            self.observed_RVs.append(observed_rv_var)

            # Create deterministic that combines observed and missing
            rv_var = at.zeros(data.shape)
            rv_var = at.set_subtensor(rv_var[mask.nonzero()], missing_rv_var)
            rv_var = at.set_subtensor(rv_var[antimask_idx], observed_rv_var)
            rv_var = Deterministic(name, rv_var, self, dims, auto=True)

        else:
            if sps.issparse(data):
                data = sparse.basic.as_sparse(data, name=name)
            else:
                data = at.as_tensor_variable(data, name=name)
            rv_var.tag.observations = data
            self.create_value_var(rv_var, transform=None, value_var=data)
            self.add_random_variable(rv_var, dims)
            self.observed_RVs.append(rv_var)

        return rv_var

    def create_value_var(
        self, rv_var: TensorVariable, transform: Any, value_var: Optional[Variable] = None
    ) -> TensorVariable:
        """Create a ``TensorVariable`` that will be used as the random
        variable's "value" in log-likelihood graphs.

        In general, we'll call this type of variable the "value" variable.

        In all other cases, the role of the value variable is taken by
        observed data. That's why value variables are only referenced in
        this branch of the conditional.

        """
        if value_var is None:
            value_var = rv_var.type()
            value_var.name = rv_var.name

        if aesara.config.compute_test_value != "off":
            value_var.tag.test_value = rv_var.tag.test_value

        rv_var.tag.value_var = value_var

        # Make the value variable a transformed value variable,
        # if there's an applicable transform
        if transform is UNSET and rv_var.owner:
            transform = _default_transform(rv_var.owner.op, rv_var)

        if transform is not None and transform is not UNSET:
            value_var.tag.transform = transform
            value_var.name = f"{value_var.name}_{transform.name}__"
            if aesara.config.compute_test_value != "off":
                value_var.tag.test_value = transform.forward(
                    value_var, *rv_var.owner.inputs
                ).tag.test_value
            self.named_vars[value_var.name] = value_var

        self.rvs_to_values[rv_var] = value_var
        self.values_to_rvs[value_var] = rv_var

        return value_var

    def add_random_variable(self, var, dims: Optional[Tuple[Union[str, None], ...]] = None):
        """Add a random variable to the named variables of the model."""
        if self.named_vars.tree_contains(var.name):
            raise ValueError(f"Variable name {var.name} already exists.")

        if dims is not None:
            if isinstance(dims, str):
                dims = (dims,)
            for dim in dims:
                if dim not in self.coords and dim is not None:
                    raise ValueError(f"Dimension {dim} is not specified in `coords`.")
            if any(var.name == dim for dim in dims):
                raise ValueError(f"Variable `{var.name}` has the same name as its dimension label.")
            self._RV_dims[var.name] = dims

        self.named_vars[var.name] = var
        if not hasattr(self, self.name_of(var.name)):
            setattr(self, self.name_of(var.name), var)

    @property
    def prefix(self) -> str:
        if self.isroot or not self.parent.prefix:
            name = self.name
        else:
            name = f"{self.parent.prefix}::{self.name}"
        return name

    def name_for(self, name):
        """Checks if name has prefix and adds if needed"""
        name = self._validate_name(name)
        if self.prefix:
            if not name.startswith(self.prefix):
                return f"{self.prefix}::{name}"
            else:
                return name
        else:
            return name

    def name_of(self, name):
        """Checks if name has prefix and deletes if needed"""
        name = self._validate_name(name)
        if not self.prefix or not name:
            return name
        elif name.startswith(self.prefix + "::"):
            return name[len(self.prefix) + 2 :]
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

    def compile_fn(
        self,
        outs: Sequence[Variable],
        *,
        inputs: Optional[Sequence[Variable]] = None,
        mode=None,
        point_fn: bool = True,
        **kwargs,
    ) -> Union["PointFunc", Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]]:
        """Compiles an Aesara function

        Parameters
        ----------
        outs: Aesara variable or iterable of Aesara variables
        inputs: Aesara input variables, defaults to aesaraf.inputvars(outs).
        mode: Aesara compilation mode, default=None
        point_fn:
            Whether to wrap the compiled function in a PointFunc, which takes a Point
            dictionary with model variable names and values as input.

        Returns
        -------
        Compiled Aesara function
        """
        if inputs is None:
            inputs = inputvars(outs)

        with self:
            fn = compile_pymc(
                inputs,
                outs,
                allow_input_downcast=True,
                accept_inplace=True,
                mode=mode,
                **kwargs,
            )

        if point_fn:
            return PointFunc(fn)
        return fn

    def profile(self, outs, *, n=1000, point=None, profile=True, **kwargs):
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
        kwargs.setdefault("on_unused_input", "ignore")
        f = self.compile_fn(outs, inputs=self.value_vars, point_fn=False, profile=profile, **kwargs)
        if point is None:
            point = self.compute_initial_point()

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
        conditional on the values of `b` and stored in `b`.

        """
        raise FutureWarning(
            "The `Model.update_start_vals` method was removed."
            " To change initial values you may set the items of `Model.initial_values` directly."
        )

    def eval_rv_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Evaluates shapes of untransformed AND transformed free variables.

        Returns
        -------
        shapes : dict
            Maps untransformed and transformed variable names to shape tuples.
        """
        names = []
        outputs = []
        for rv in self.free_RVs:
            rv_var = self.rvs_to_values[rv]
            transform = getattr(rv_var.tag, "transform", None)
            if transform is not None:
                names.append(get_transformed_name(rv.name, transform))
                outputs.append(transform.forward(rv, *rv.owner.inputs).shape)
            names.append(rv.name)
            outputs.append(rv.shape)
        f = aesara.function(
            inputs=[],
            outputs=outputs,
            givens=[(obs, obs.tag.observations) for obs in self.observed_RVs],
            mode=aesara.compile.mode.FAST_COMPILE,
            on_unused_input="ignore",
        )
        return {name: tuple(shape) for name, shape in zip(names, f())}

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

        ``pymc.exceptions.SamplingError`` if the evaluation of the parameters
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
                    f"Valid keys are: {valid_keys}, but {extra_keys} was supplied"
                )

            initial_eval = self.point_logps(point=elem)

            if not all(np.isfinite(v) for v in initial_eval.values()):
                raise SamplingError(
                    "Initial evaluation of model at starting point failed!\n"
                    f"Starting values:\n{elem}\n\n"
                    f"Initial evaluation results:\n{initial_eval}"
                )

    def check_test_point(self, *args, **kwargs):
        warnings.warn(
            "`Model.check_test_point` has been deprecated. Use `Model.point_logps` instead.",
            FutureWarning,
        )
        return self.point_logps(*args, **kwargs)

    def point_logps(self, point=None, round_vals=2):
        """Computes the log probability of `point` for all random variables in the model.

        Parameters
        ----------
        point: Point, optional
            Point to be evaluated.  If ``None``, then ``model.initial_point``
            is used.
        round_vals: int, default 2
            Number of decimals to round log-probabilities.

        Returns
        -------
        log_probability_of_point : dict
            Log probability of `point`.
        """
        if point is None:
            point = self.compute_initial_point()

        factors = self.basic_RVs + self.potentials
        factor_logps_fn = [at.sum(factor) for factor in self.logpt(factors, sum=False)]
        return {
            factor.name: np.round(np.asarray(factor_logp), round_vals)
            for factor, factor_logp in zip(
                factors,
                self.compile_fn(factor_logps_fn)(point),
            )
        }


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

        >>> import pymc as pm
        >>> with pm.Model() as model:
        ...     x = pm.MutableData('x', [1., 2., 3.])
        ...     y = pm.MutableData('y', [1., 2., 3.])
        ...     beta = pm.Normal('beta', 0, 1)
        ...     obs = pm.Normal('obs', x * beta, 1, observed=y)
        ...     idata = pm.sample(1000, tune=1000)

    Set the value of `x` to predict on new data.

    .. code:: ipython

        >>> with model:
        ...     pm.set_data({'x': [5., 6., 9.]})
        ...     y_test = pm.sample_posterior_predictive(idata)
        >>> y_test['obs'].mean(axis=0)
        array([4.6088569 , 5.54128318, 8.32953844])
    """
    model = modelcontext(model)

    for variable_name, new_value in new_data.items():
        model.set_data(variable_name, new_value)


def compile_fn(outs, mode=None, point_fn=True, model=None, **kwargs):
    """Compiles an Aesara function which returns ``outs`` and takes values of model
    vars as a dict as an argument.
    Parameters
    ----------
    outs: Aesara variable or iterable of Aesara variables
    mode: Aesara compilation mode
    point_fn:
        Whether to wrap the compiled function in a PointFunc, which takes a Point
        dictionary with model variable names and values as input.
    Returns
    -------
    Compiled Aesara function as point function.
    """
    model = modelcontext(model)
    return model.compile_fn(outs, mode, point_fn=point_fn, **kwargs)


def Point(*args, filter_model_vars=False, **kwargs) -> Dict[str, np.ndarray]:
    """Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
    args, kwargs
        arguments to build a dict
    filter_model_vars : bool
        If `True`, only model variables are included in the result.
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


class PointFunc:
    """Wraps so a function so it takes a dict of arguments instead of arguments."""

    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)


def Deterministic(name, var, model=None, dims=None, auto=False):
    """Create a named deterministic variable

    Notes
    -----
    Deterministic nodes are ones that given all the inputs are not random variables

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

    from pymc.printing import str_for_potential_or_deterministic

    var.str_repr = types.MethodType(
        functools.partial(str_for_potential_or_deterministic, dist_name="Deterministic"), var
    )
    var._repr_latex_ = types.MethodType(
        functools.partial(
            str_for_potential_or_deterministic, dist_name="Deterministic", formatting="latex"
        ),
        var,
    )

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
    var.tag.scaling = 1.0
    model.potentials.append(var)
    model.add_random_variable(var)

    from pymc.printing import str_for_potential_or_deterministic

    var.str_repr = types.MethodType(
        functools.partial(str_for_potential_or_deterministic, dist_name="Potential"), var
    )
    var._repr_latex_ = types.MethodType(
        functools.partial(
            str_for_potential_or_deterministic, dist_name="Potential", formatting="latex"
        ),
        var,
    )

    return var
