#   Copyright 2024 The PyMC Developers
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
from __future__ import annotations

import functools
import sys
import threading
import types
import warnings

from collections.abc import Iterable, Sequence
from typing import (
    Literal,
    cast,
    overload,
)

import numpy as np
import pytensor
import pytensor.sparse as sparse
import pytensor.tensor as pt
import scipy.sparse as sps

from pytensor.compile import DeepCopyOp, Function, get_mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant, Variable, graph_inputs
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.variable import TensorConstant, TensorVariable

from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.data import is_valid_observed
from pymc.exceptions import (
    BlockModelAccessError,
    ImputationWarning,
    SamplingError,
    ShapeError,
    ShapeWarning,
)
from pymc.initial_point import make_initial_point_fn
from pymc.logprob.basic import transformed_conditional_logp
from pymc.logprob.transforms import Transform
from pymc.logprob.utils import ParameterValueError, replace_rvs_by_values
from pymc.model_graph import model_to_graphviz
from pymc.pytensorf import (
    PointFunc,
    SeedSequenceSeed,
    compile_pymc,
    convert_observed_data,
    gradient,
    hessian,
    inputvars,
    rewrite_pregrad,
)
from pymc.util import (
    UNSET,
    VarName,
    WithMemoization,
    _add_future_warning_tag,
    _UnsetType,
    get_transformed_name,
    get_value_vars_from_user_vars,
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


class ModelManager(threading.local):
    """Keeps track of currently active model contexts.

    A global instance of this is created in this module on import.
    Use that instance, `MODEL_MANAGER` to inspect current contexts.

    It inherits from threading.local so is thread-safe, if models
    can be entered/exited within individual threads.
    """

    def __init__(self):
        self.active_contexts: list[Model] = []

    @property
    def current_context(self) -> Model | None:
        """Return the innermost context of any current contexts."""
        return self.active_contexts[-1] if self.active_contexts else None

    @property
    def parent_context(self) -> Model | None:
        """Return the parent context to the active context, if any."""
        return self.active_contexts[-2] if len(self.active_contexts) > 1 else None


# MODEL_MANAGER is instantiated at import, and serves as a truth for
# what any currently active model contexts are.
MODEL_MANAGER = ModelManager()


def modelcontext(model: Model | None) -> Model:
    """Return the given model or, if None was supplied, try to find one in the context stack."""
    if model is None:
        model = Model.get_context(error_if_none=False)

        if model is None:
            # TODO: This should be a ValueError, but that breaks
            # ArviZ (and others?), so might need a deprecation.
            raise TypeError("No model on context stack.")
    return model


class ValueGradFunction:
    """Create a PyTensor function that computes a value and its gradient.

    Parameters
    ----------
    costs: list of PyTensor variables
        We compute the weighted sum of the specified PyTensor values, and the gradient
        of that sum. The weights can be specified with `ValueGradFunction.set_weights`.
    grad_vars: list of named PyTensor variables or None
        The arguments with respect to which the gradient is computed.
    extra_vars_and_values: dict of PyTensor variables and their initial values
        Other arguments of the function that are assumed constant and their
        values. They are stored in shared variables and can be set using
        `set_extra_values`.
    dtype: str, default=pytensor.config.floatX
        The dtype of the arrays.
    casting: {'no', 'equiv', 'save', 'same_kind', 'unsafe'}, default='no'
        Casting rule for casting `grad_args` to the array dtype.
        See `numpy.can_cast` for a description of the options.
        Keep in mind that we cast the variables to the array *and*
        back from the array dtype to the variable dtype.
    compute_grads: bool, default=True
        If False, return only the logp, not the gradient.
    kwargs
        Extra arguments are passed on to `pytensor.function`.

    Attributes
    ----------
    profile: PyTensor profiling object or None
        The profiling object of the PyTensor function that computes value and
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
            dtype = pytensor.config.floatX
        self.dtype = dtype

        self._n_costs = len(costs)
        if self._n_costs == 0:
            raise ValueError("At least one cost is required.")
        weights = np.ones(self._n_costs - 1, dtype=self.dtype)
        self._weights = pytensor.shared(weights, "__weights")

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
            shared = pytensor.shared(
                value, var.name + "_shared__", shape=[1 if s == 1 else None for s in value.shape]
            )
            self._extra_vars_shared[var.name] = shared
            givens.append((var, shared))

        cost = rewrite_pregrad(cost)

        if compute_grads:
            grads = pytensor.grad(cost, grad_vars, disconnected_inputs="ignore")
            for grad_wrt, var in zip(grads, grad_vars):
                grad_wrt.name = f"{var.name}_grad"
            outputs = [cost, *grads]
        else:
            outputs = [cost]

        inputs = grad_vars

        self._pytensor_function = compile_pymc(inputs, outputs, givens=givens, **kwargs)

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

        cost, *grads = self._pytensor_function(*grad_vars)

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
        """Profiling information of the underlying PyTensor function."""
        return self._pytensor_function.profile


class ContextMeta(type):
    """A metaclass in order to apply a model's context during `Model.__init__``."""

    # We want the Model's context to be active during __init__. In order for this
    # to apply to subclasses of Model as well, we need to use a metaclass.
    def __call__(cls: type[Model], *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        with instance:  # applies context
            instance.__init__(*args, **kwargs)
        return instance


class Model(WithMemoization, metaclass=ContextMeta):
    """Encapsulates the variables and likelihood factors of a model.

    Model class can be used for creating class based models. To create
    a class based model you should inherit from :class:`~pymc.Model` and
    override the `__init__` method with arbitrary definitions (do not
    forget to call base class :meth:`pymc.Model.__init__` first).

    Parameters
    ----------
    name : str
        name that will be used as prefix for names of all random
        variables defined within model
    coords : dict
        Xarray-like coordinate keys and values. These coordinates can be used
        to specify the shape of random variables and to label (but not specify)
        the shape of Determinsitic, Potential and Data objects.
        Other than specifying the shape of random variables, coordinates have no
        effect on the model. They can't be used for label-based broadcasting or indexing.
        You must use numpy-like operations for those behaviors.
    check_bounds : bool
        Ensure that input parameters to distributions are in a valid
        range. If your model is built in a way where you know your
        parameters can only take on valid values you can set this to
        False for increased speed. This should not be used if your model
        contains discrete variables.
    model : PyMC model, optional
        A parent model that this model belongs to. If not specified and the current model
        is created inside another model's context, the parent model will be set to that model.
        If `None` the model will not have a parent.

    Examples
    --------
    Use context manager to define model and respective variables

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Normal("x")


    Use object API to define model and respective variables

    .. code-block:: python

        import pymc as pm

        model = pm.Model()
        x = pm.Normal("x", model=model)


    Use coords for defining the shape of random variables and labeling other model variables

    .. code-block:: python

        import pymc as pm
        import numpy as np

        coords = {
            "feature",
            ["A", "B", "C"],
            "trial",
            [1, 2, 3, 4, 5],
        }

        with pm.Model(coords=coords) as model:
            # Variable will have default dim label `intercept__dim_0`
            intercept = pm.Normal("intercept", shape=(3,))
            # Variable will have shape (3,) and dim label `feature`
            beta = pm.Normal("beta", dims=("feature",))

            # Dims below are only used for labeling, they have no effect on shape
            # Variable will have default dim label `idx__dim_0`
            idx = pm.Data("idx", np.array([0, 1, 1, 2, 2]))
            x = pm.Data("x", np.random.normal(size=(5, 3)), dims=("trial", "feature"))
            # single dim can be passed as string
            mu = pm.Deterministic("mu", intercept[idx] + beta @ x, dims="trial")

            # Dims controls the shape of the variable
            # If not specified, it would be inferred from the shape of the observations
            y = pm.Normal("y", mu=mu, observed=[-1, 0, 0, 1, 1], dims=("trial",))


    Define nested models, and provide name for variable name prefixing

    .. code-block:: python

        import pymc as pm

        with pm.Model(name="root") as root:
            x = pm.Normal("x")  # Variable wil be named "root::x"

            with pm.Model(name="first") as first:
                # Variable will belong to root and first
                y = pm.Normal("y", mu=x)  # Variable wil be named "root::first::y"

            # Can pass parent model explicitly
            with pm.Model(name="second", model=root) as second:
                # Variable will belong to root and second
                z = pm.Normal("z", mu=y)  # Variable wil be named "root::second::z"

            # Set None for standalone model
            with pm.Model(name="third", model=None) as third:
                # Variable will belong to third only
                w = pm.Normal("w")  # Variable wil be named "third::w"


    Set `check_bounds` to False for models with only continuous variables and default transformers
    PyMC will remove the bounds check from the model logp which can speed up sampling

    .. code-block:: python

        import pymc as pm

        with pm.Model(check_bounds=False) as model:
            sigma = pm.HalfNormal("sigma")
            x = pm.Normal("x", sigma=sigma)  # No bounds check will be performed on `sigma`


    """

    def __enter__(self):
        """Enter the context manager."""
        MODEL_MANAGER.active_contexts.append(self)
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        """Exit the context manager."""
        _ = MODEL_MANAGER.active_contexts.pop()

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
        *,
        coords_mutable=None,
        model: _UnsetType | None | Model = UNSET,
    ):
        self.name = self._validate_name(name)
        self.check_bounds = check_bounds
        self._parent = model if not isinstance(model, _UnsetType) else MODEL_MANAGER.parent_context

        if coords_mutable is not None:
            warnings.warn(
                "All coords are now mutable by default. coords_mutable will be removed in a future release.",
                FutureWarning,
            )

        if self.parent is not None:
            self.named_vars = treedict(parent=self.parent.named_vars)
            self.named_vars_to_dims = treedict(parent=self.parent.named_vars_to_dims)
            self.values_to_rvs = treedict(parent=self.parent.values_to_rvs)
            self.rvs_to_values = treedict(parent=self.parent.rvs_to_values)
            self.rvs_to_transforms = treedict(parent=self.parent.rvs_to_transforms)
            self.rvs_to_initial_values = treedict(parent=self.parent.rvs_to_initial_values)
            self.free_RVs = treelist(parent=self.parent.free_RVs)
            self.observed_RVs = treelist(parent=self.parent.observed_RVs)
            self.deterministics = treelist(parent=self.parent.deterministics)
            self.potentials = treelist(parent=self.parent.potentials)
            self.data_vars = treelist(parent=self.parent.data_vars)
            self._coords = self.parent._coords
            self._dim_lengths = self.parent._dim_lengths
        else:
            self.named_vars = treedict()
            self.named_vars_to_dims = treedict()
            self.values_to_rvs = treedict()
            self.rvs_to_values = treedict()
            self.rvs_to_transforms = treedict()
            self.rvs_to_initial_values = treedict()
            self.free_RVs = treelist()
            self.observed_RVs = treelist()
            self.deterministics = treelist()
            self.potentials = treelist()
            self.data_vars = treelist()
            self._coords = {}
            self._dim_lengths = {}
        self.add_coords(coords)
        if coords_mutable is not None:
            for name, values in coords_mutable.items():
                self.add_coord(name, values, mutable=True)

        from pymc.printing import str_for_model

        self.str_repr = types.MethodType(str_for_model, self)
        self._repr_latex_ = types.MethodType(
            functools.partial(str_for_model, formatting="latex"), self
        )

    @classmethod
    def get_context(
        cls, error_if_none: bool = True, allow_block_model_access: bool = False
    ) -> Model | None:
        model = MODEL_MANAGER.current_context
        if isinstance(model, BlockModelAccess) and not allow_block_model_access:
            raise BlockModelAccessError(model.error_msg_on_access)
        if model is None and error_if_none:
            raise TypeError("No model on context stack")
        return model

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

    def logp_dlogp_function(self, grad_vars=None, tempered=False, **kwargs):
        """Compile a PyTensor function that computes logp and gradient.

        Parameters
        ----------
        grad_vars : list of random variables, optional
            Compute the gradient with respect to those variables. If None,
            use all free random variables of this model.
        tempered : bool
            Compute the tempered logp `free_logp + alpha * observed_logp`.
            `alpha` can be changed using `ValueGradFunction.set_weights([alpha])`.
        """
        if grad_vars is None:
            grad_vars = self.continuous_value_vars
        else:
            grad_vars = get_value_vars_from_user_vars(grad_vars, self)
            for i, var in enumerate(grad_vars):
                if var.dtype not in continuous_types:
                    raise ValueError(f"Can only compute the gradient of continuous types: {var}")

        if tempered:
            costs = [self.varlogp, self.datalogp]
        else:
            costs = [self.logp()]

        input_vars = {i for i in graph_inputs(costs) if not isinstance(i, Constant)}
        ip = self.initial_point(0)
        extra_vars_and_values = {
            var: ip[var.name]
            for var in self.value_vars
            if var in input_vars and var not in grad_vars
        }
        return ValueGradFunction(costs, grad_vars, extra_vars_and_values, **kwargs)

    def compile_logp(
        self,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
        sum: bool = True,
        **compile_kwargs,
    ) -> PointFunc:
        """Compiled log probability density function.

        Parameters
        ----------
        vars : list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian : bool
            Whether to include jacobian terms in logprob graph. Defaults to True.
        sum : bool
            Whether to sum all logp terms or return elemwise logp for each variable.
            Defaults to True.
        """
        return self.compile_fn(self.logp(vars=vars, jacobian=jacobian, sum=sum), **compile_kwargs)

    def compile_dlogp(
        self,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
        **compile_kwargs,
    ) -> PointFunc:
        """Compiled log probability density gradient function.

        Parameters
        ----------
        vars : list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian : bool
            Whether to include jacobian terms in logprob graph. Defaults to True.
        """
        return self.compile_fn(self.dlogp(vars=vars, jacobian=jacobian), **compile_kwargs)

    def compile_d2logp(
        self,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
        negate_output=True,
        **compile_kwargs,
    ) -> PointFunc:
        """Compiled log probability density hessian function.

        Parameters
        ----------
        vars : list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian : bool
            Whether to include jacobian terms in logprob graph. Defaults to True.
        """
        return self.compile_fn(
            self.d2logp(vars=vars, jacobian=jacobian, negate_output=negate_output),
            **compile_kwargs,
        )

    def logp(
        self,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
        sum: bool = True,
    ) -> Variable | list[Variable]:
        """Elemwise log-probability of the model.

        Parameters
        ----------
        vars : list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian : bool
            Whether to include jacobian terms in logprob graph. Defaults to True.
        sum : bool
            Whether to sum all logp terms or return elemwise logp for each variable.
            Defaults to True.

        Returns
        -------
        Logp graph(s)
        """
        varlist: list[TensorVariable]
        if vars is None:
            varlist = self.free_RVs + self.observed_RVs + self.potentials
        elif not isinstance(vars, list | tuple):
            varlist = [vars]
        else:
            varlist = cast(list[TensorVariable], vars)

        # We need to separate random variables from potential terms, and remember their
        # original order so that we can merge them together in the same order at the end
        rvs = []
        potentials = []
        rv_order, potential_order = [], []
        for i, var in enumerate(varlist):
            rv = self.values_to_rvs.get(var, var)
            if rv in self.basic_RVs:
                rvs.append(rv)
                rv_order.append(i)
            else:
                if var in self.potentials:
                    potentials.append(var)
                    potential_order.append(i)
                else:
                    raise ValueError(
                        f"Requested variable {var} not found among the model variables"
                    )

        rv_logps: list[TensorVariable] = []
        if rvs:
            rv_logps = transformed_conditional_logp(
                rvs=rvs,
                rvs_to_values=self.rvs_to_values,
                rvs_to_transforms=self.rvs_to_transforms,
                jacobian=jacobian,
            )
            assert isinstance(rv_logps, list)

        # Replace random variables by their value variables in potential terms
        potential_logps = []
        if potentials:
            potential_logps = self.replace_rvs_by_values(potentials)

        logp_factors = [None] * len(varlist)
        for logp_order, logp in zip((rv_order + potential_order), (rv_logps + potential_logps)):
            logp_factors[logp_order] = logp

        if not sum:
            return logp_factors

        logp_scalar = pt.sum([pt.sum(factor) for factor in logp_factors])
        logp_scalar_name = "__logp" if jacobian else "__logp_nojac"
        if self.name:
            logp_scalar_name = f"{logp_scalar_name}_{self.name}"
        logp_scalar.name = logp_scalar_name
        return logp_scalar

    def dlogp(
        self,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
    ) -> Variable:
        """Gradient of the models log-probability w.r.t. ``vars``.

        Parameters
        ----------
        vars : list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian : bool
            Whether to include jacobian terms in logprob graph. Defaults to True.

        Returns
        -------
        dlogp graph
        """
        if vars is None:
            value_vars = None
        else:
            if not isinstance(vars, list | tuple):
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

        cost = self.logp(jacobian=jacobian)
        cost = rewrite_pregrad(cost)
        return gradient(cost, value_vars)

    def d2logp(
        self,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
        negate_output=True,
    ) -> Variable:
        """Hessian of the models log-probability w.r.t. ``vars``.

        Parameters
        ----------
        vars : list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables, as well as potential terms in model.
        jacobian : bool
            Whether to include jacobian terms in logprob graph. Defaults to True.

        Returns
        -------
        d²logp graph
        """
        if vars is None:
            value_vars = None
        else:
            if not isinstance(vars, list | tuple):
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

        cost = self.logp(jacobian=jacobian)
        cost = rewrite_pregrad(cost)
        return hessian(cost, value_vars, negate_output=negate_output)

    @property
    def datalogp(self) -> Variable:
        """PyTensor scalar of log-probability of the observed variables and potential terms."""
        return self.observedlogp + self.potentiallogp

    @property
    def varlogp(self) -> Variable:
        """PyTensor scalar of log-probability of the unobserved random variables (excluding deterministic)."""
        return self.logp(vars=self.free_RVs)

    @property
    def varlogp_nojac(self) -> Variable:
        """PyTensor scalar of log-probability of the unobserved random variables (excluding deterministic) without jacobian term."""
        return self.logp(vars=self.free_RVs, jacobian=False)

    @property
    def observedlogp(self) -> Variable:
        """PyTensor scalar of log-probability of the observed variables."""
        return self.logp(vars=self.observed_RVs)

    @property
    def potentiallogp(self) -> Variable:
        """PyTensor scalar of log-probability of the Potential terms."""
        # Convert random variables in Potential expression into their log-likelihood
        # inputs and apply their transforms, if any
        potentials = self.replace_rvs_by_values(self.potentials)
        if potentials:
            return pt.sum([pt.sum(factor) for factor in potentials])
        else:
            return pt.constant(0.0)

    @property
    def value_vars(self):
        """List of unobserved random variables used as inputs to the model's log-likelihood (which excludes deterministics)."""
        return [self.rvs_to_values[v] for v in self.free_RVs]

    @property
    def unobserved_value_vars(self):
        """List of all random variables (including untransformed projections), as well as deterministics used as inputs and outputs of the model's log-likelihood graph."""
        vars = []
        transformed_rvs = []
        for rv in self.free_RVs:
            value_var = self.rvs_to_values[rv]
            transform = self.rvs_to_transforms[rv]
            if transform is not None:
                transformed_rvs.append(rv)
            vars.append(value_var)

        # Remove rvs from untransformed values graph
        untransformed_vars = self.replace_rvs_by_values(transformed_rvs)

        # Remove rvs from deterministics graph
        deterministics = self.replace_rvs_by_values(self.deterministics)

        return vars + untransformed_vars + deterministics

    @property
    def discrete_value_vars(self):
        """All the discrete value variables in the model."""
        return list(typefilter(self.value_vars, discrete_types))

    @property
    def continuous_value_vars(self):
        """All the continuous value variables in the model."""
        return list(typefilter(self.value_vars, continuous_types))

    @property
    def basic_RVs(self):
        """List of random variables the model is defined in terms of.

        This excludes deterministics.

        These are the actual random variable terms that make up the
        "sample-space" graph (i.e. you can sample these graphs by compiling them
        with `pytensor.function`).  If you want the corresponding log-likelihood terms,
        use `model.value_vars` instead.
        """
        return self.free_RVs + self.observed_RVs

    @property
    def unobserved_RVs(self):
        """List of all random variables, including deterministic ones.

        These are the actual random variable terms that make up the
        "sample-space" graph (i.e. you can sample these graphs by compiling them
        with `pytensor.function`).  If you want the corresponding log-likelihood terms,
        use `var.unobserved_value_vars` instead.
        """
        return self.free_RVs + self.deterministics

    @property
    def coords(self) -> dict[str, tuple | None]:
        """Coordinate values for model dimensions."""
        return self._coords

    @property
    def dim_lengths(self) -> dict[str, Variable]:
        """The symbolic lengths of dimensions in the model.

        The values are typically instances of ``TensorVariable`` or ``ScalarSharedVariable``.
        """
        return self._dim_lengths

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
        values: Sequence | np.ndarray | None = None,
        mutable: bool | None = None,
        *,
        length: int | Variable | None = None,
    ):
        """Register a dimension coordinate with the model.

        Parameters
        ----------
        name : str
            Name of the dimension.
            Forbidden: {"chain", "draw", "__sample__"}
        values : optional, array_like
            Coordinate values or ``None`` (for auto-numbering).
            If ``None`` is passed, a ``length`` must be specified.
        mutable : bool
            Whether the created dimension should be resizable.
            Default is False.
        length : optional, scalar
            A scalar of the dimensions length.
            Defaults to ``pytensor.tensor.constant(len(values))``.
        """
        if mutable is not None:
            warnings.warn(
                "Coords are now always mutable. Specifying `mutable` will raise an error in a future release",
                FutureWarning,
            )

        if name in {"draw", "chain", "__sample__"}:
            raise ValueError(
                "Dimensions can not be named `draw`, `chain` or `__sample__`, "
                "as those are reserved for use in `InferenceData`."
            )
        if values is None and length is None:
            raise ValueError(
                f"Either `values` or `length` must be specified for the '{name}' dimension."
            )
        if values is not None:
            # Conversion to a tuple ensures that the coordinate values are immutable.
            # Also unlike numpy arrays the's tuple.index(...) which is handy to work with.
            values = tuple(values)
        if name in self.coords:
            if not np.array_equal(values, self.coords[name]):
                raise ValueError(f"Duplicate and incompatible coordinate: {name}.")
        if length is not None and not isinstance(length, int | Variable):
            raise ValueError(
                f"The `length` passed for the '{name}' coord must be an int, PyTensor Variable or None."
            )
        if length is None:
            length = len(values)
        if not isinstance(length, Variable):
            length = pytensor.shared(length, name=name)
        assert length.type.ndim == 0
        self._dim_lengths[name] = length
        self._coords[name] = values

    def add_coords(
        self,
        coords: dict[str, Sequence | None],
        *,
        lengths: dict[str, int | Variable | None] | None = None,
    ):
        """Vectorized version of ``Model.add_coord``."""
        if coords is None:
            return
        lengths = lengths or {}

        for name, values in coords.items():
            self.add_coord(name, values, length=lengths.get(name, None))

    def set_dim(self, name: str, new_length: int, coord_values: Sequence | None = None):
        """Update a mutable dimension.

        Parameters
        ----------
        name : str
            Name of the dimension.
        new_length : int
            New length of the dimension.
        coord_values : array_like, optional
            Optional sequence of coordinate values.
        """
        if coord_values is None and self.coords.get(name, None) is not None:
            raise ValueError(
                f"'{name}' has coord values. Pass `set_dim(..., coord_values=...)` to update them."
            )
        if coord_values is not None:
            len_cvals = len(coord_values)
            if len_cvals != new_length:
                raise ShapeError(
                    "Length of new coordinate values does not match the new dimension length.",
                    actual=len_cvals,
                    expected=new_length,
                )
            self._coords[name] = tuple(coord_values)
        dim_length = self.dim_lengths[name]
        if not isinstance(dim_length, SharedVariable):
            raise TypeError(
                f"The dim_length of `{name}` must be a `SharedVariable` "
                "(created through `coords` to allow updating). "
                f"The current type is: {type(dim_length)}"
            )
        dim_length.set_value(new_length)
        return

    def initial_point(self, random_seed: SeedSequenceSeed = None) -> dict[str, np.ndarray]:
        """Compute the initial point of the model.

        Parameters
        ----------
        random_seed : SeedSequenceSeed, default None
            Seed(s) for generating initial point from the model. Passed into :func:`pymc.pytensorf.reseed_rngs`

        Returns
        -------
        ip : dict of {str : array_like}
            Maps names of transformed variables to numeric initial values in the transformed space.
        """
        fn = make_initial_point_fn(model=self, return_transformed=True)
        return Point(fn(random_seed), model=self)

    def set_initval(self, rv_var, initval):
        """Set an initial value (strategy) for a random variable."""
        if initval is not None and not isinstance(initval, Variable | str):
            # Convert scalars or array-like inputs to ndarrays
            initval = rv_var.type.filter(initval)

        self.rvs_to_initial_values[rv_var] = initval

    def set_data(
        self,
        name: str,
        values: Sequence | np.ndarray,
        coords: dict[str, Sequence] | None = None,
    ):
        """Change the values of a data variable in the model.

        In contrast to pm.Data().set_value, this method can also
        update the corresponding coordinates.

        Parameters
        ----------
        name : str
            Name of a shared variable in the model.
        values : array_like
            New values for the shared variable.
        coords : optional, dict
            New coordinate values for dimensions of the shared variable.
            Must be provided for all named dimensions that change in length
            and already have coordinate values.
        """
        shared_object = self[name]
        if not isinstance(shared_object, SharedVariable):
            raise TypeError(
                f"The variable `{name}` must be a `SharedVariable` "
                "(created through `pm.Data()` to allow updating.) "
                f"The current type is: {type(shared_object)}"
            )

        if isinstance(values, list):
            values = np.array(values)
        values = convert_observed_data(values)
        dims = self.named_vars_to_dims.get(name, None) or ()
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
            # NOTE: If there are multiple pm.Data containers sharing this dim, but the user only
            #       changes the values for one of them, they will run into shape problems nonetheless.
            if length_changed:
                if original_coords is not None:
                    if new_coords is None:
                        raise ValueError(
                            f"The '{name}' variable already had {len(original_coords)} coord values defined for "
                            f"its {dname} dimension. With the new values this dimension changes to length "
                            f"{new_length}, so new coord values for the {dname} dimension are required."
                        )
                if isinstance(length_tensor, TensorConstant):
                    # The dimension was fixed in length.
                    # Resizing a data variable in this dimension would
                    # definitely lead to shape problems.
                    raise ShapeError(
                        f"Resizing dimension '{dname}' is impossible, because "
                        "a `TensorConstant` stores its length. To be able "
                        "to change the dimension length, create data with "
                        "pm.Data() instead."
                    )
                elif length_tensor.owner is not None:
                    # The dimension was created from another variable:
                    length_tensor_origin = length_tensor.owner.inputs[0]
                    # Get a handle on the tensor from which this dimension length was
                    # obtained by doing subindexing on the shape as in `.shape[i]`.
                    if isinstance(length_tensor_origin, TensorConstant):
                        raise ShapeError(
                            f"Resizing dimension '{dname}' with values of length {new_length} would lead to incompatibilities, "
                            f"because the dimension length is tied to a TensorConstant. "
                            f"Check if the dimension was defined implicitly before the shared variable '{name}' was created, "
                            f"for example by another model variable.",
                            actual=new_length,
                            expected=old_length,
                        )

                    # The shape entry this dimension is tied to is not a TensorConstant.
                    # Whether the dimension can be resized depends on the kind of Variable the shape belongs to.
                    # TODO: Consider checking the graph is what we are assuming it is
                    # isinstance(length_tensor.owner.op, Subtensor)
                    # isinstance(length_tensor.owner.inputs[0].owner.op, Shape)
                    length_belongs_to = length_tensor_origin.owner.inputs[0]

                    if length_belongs_to is shared_object:
                        # This is the shared variable that's being updated!
                        # No surprise it's changing.
                        pass
                    elif isinstance(length_belongs_to, SharedVariable):
                        # The dimension is mutable through a SharedVariable other than the one being modified.
                        # But the other variable was not yet re-sized! Warn the user to do that!
                        warnings.warn(
                            f"You are resizing a variable with dimension '{dname}' which was initialized "
                            f"as a mutable dimension by another variable ('{length_belongs_to}')."
                            " Remember to update that variable with the correct shape to avoid shape issues.",
                            ShapeWarning,
                            stacklevel=2,
                        )
                    else:
                        # The dimension is immutable.
                        raise ShapeError(
                            f"Resizing dimension '{dname}' with values of length {new_length} would lead to incompatibilities, "
                            f"because the dimension was initialized from '{length_belongs_to}' which is not a shared variable. "
                            f"Check if the dimension was defined implicitly before the shared variable '{name}' was created, "
                            f"for example by another model variable.",
                            actual=new_length,
                            expected=old_length,
                        )
                if isinstance(length_tensor, SharedVariable):
                    # The dimension is mutable, but was defined without being linked
                    # to a shared variable. This is allowed, but a little less robust.
                    self.set_dim(dname, new_length, coord_values=new_coords)

            if new_coords is not None:
                # Update the registered coord values (also if they were None)
                if len(new_coords) != new_length:
                    raise ShapeError(
                        f"Length of new coordinate values for dimension '{dname}' does not match the provided values.",
                        actual=len(new_coords),
                        expected=new_length,
                    )
                # store it as tuple for immutability as in add_coord
                self._coords[dname] = tuple(new_coords)

        shared_object.set_value(values)

    def register_rv(
        self,
        rv_var: RandomVariable,
        name: str,
        *,
        observed=None,
        total_size=None,
        dims=None,
        default_transform=UNSET,
        transform=UNSET,
        initval=None,
    ) -> TensorVariable:
        """Register an (un)observed random variable with the model.

        Parameters
        ----------
        rv_var : TensorVariable
        name : str
            Intended name for the model variable.
        observed : array_like, optional
            Data values for observed variables.
        total_size : scalar
            upscales logp of variable with ``coef = total_size/var.shape[0]``
        dims : tuple
            Dimension names for the variable.
        default_transform
            A default transform for the random variable in log-likelihood space.
        transform
            Additional transform which may be applied after default transform.
        initval
            The initial value of the random variable.

        Returns
        -------
        TensorVariable
        """
        name = self.name_for(name)
        rv_var.name = name
        _add_future_warning_tag(rv_var)

        # Associate previously unknown dimension names with
        # the length of the corresponding RV dimension.
        if dims is not None:
            for d, dname in enumerate(dims):
                if not isinstance(dname, str):
                    raise TypeError(f"Dims must be string. Got {dname} of type {type(dname)}")
                if dname not in self.dim_lengths:
                    self.add_coord(dname, values=None, length=rv_var.shape[d])

        if observed is None:
            if total_size is not None:
                raise ValueError("total_size can only be passed to observed RVs")
            self.free_RVs.append(rv_var)
            self.create_value_var(rv_var, transform=transform, default_transform=default_transform)
            self.add_named_variable(rv_var, dims)
            self.set_initval(rv_var, initval)
        else:
            if not is_valid_observed(observed):
                raise TypeError(
                    "Variables that depend on other nodes cannot be used for observed data."
                    f"The data variable was: {observed}"
                )

            # `rv_var` is potentially changed by `make_obs_var`,
            # for example into a new graph for imputation of missing data.
            rv_var = self.make_obs_var(
                rv_var, observed, dims, default_transform, transform, total_size
            )

        return rv_var

    def make_obs_var(
        self,
        rv_var: TensorVariable,
        data: np.ndarray,
        dims,
        default_transform: Transform | None,
        transform: Transform | None,
        total_size: int | None,
    ) -> TensorVariable:
        """Create a `TensorVariable` for an observed random variable.

        Parameters
        ----------
        rv_var : TensorVariable
            The random variable that is observed.
            Its dimensionality must be compatible with the data already.
        data : array_like
            The observed data.
        dims : tuple
            Dimension names for the variable.
        default_transform
            A transform for the random variable in log-likelihood space.
        transform
            Additional transform which may be applied after default transform.

        Returns
        -------
        TensorVariable
        """
        name = rv_var.name
        data = convert_observed_data(data).astype(rv_var.dtype)

        if data.ndim != rv_var.ndim:
            raise ShapeError(
                "Dimensionality of data and RV don't match.", actual=data.ndim, expected=rv_var.ndim
            )

        mask = getattr(data, "mask", None)
        if mask is not None:
            impute_message = (
                f"Data in {rv_var} contains missing values and"
                " will be automatically imputed from the"
                " sampling distribution."
            )
            warnings.warn(impute_message, ImputationWarning)

            if total_size is not None:
                raise ValueError("total_size is not compatible with imputed variables")

            from pymc.distributions.distribution import create_partial_observed_rv

            (
                (observed_rv, observed_mask),
                (unobserved_rv, _),
                joined_rv,
            ) = create_partial_observed_rv(rv_var, mask)
            observed_data = pt.as_tensor(data.data[observed_mask])

            # Register ObservedRV corresponding to observed component
            observed_rv.name = f"{name}_observed"
            self.create_value_var(
                observed_rv, transform=transform, default_transform=None, value_var=observed_data
            )
            self.add_named_variable(observed_rv)
            self.observed_RVs.append(observed_rv)

            # Register FreeRV corresponding to unobserved components
            self.register_rv(
                unobserved_rv,
                f"{name}_unobserved",
                transform=transform,
                default_transform=default_transform,
            )

            # Register Deterministic that combines observed and missing
            # Note: This can widely increase memory consumption during sampling for large datasets
            rv_var = Deterministic(name, joined_rv, self, dims)

        else:
            if sps.issparse(data):
                data = sparse.basic.as_sparse(data, name=name)
            else:
                data = pt.as_tensor_variable(data, name=name)

            if total_size:
                from pymc.variational.minibatch_rv import create_minibatch_rv

                rv_var = create_minibatch_rv(rv_var, total_size)
                rv_var.name = name

            rv_var.tag.observations = data
            self.create_value_var(
                rv_var, transform=transform, default_transform=None, value_var=data
            )
            self.add_named_variable(rv_var, dims)
            self.observed_RVs.append(rv_var)

        return rv_var

    def create_value_var(
        self,
        rv_var: TensorVariable,
        *,
        default_transform: Transform,
        transform: Transform,
        value_var: Variable | None = None,
    ) -> TensorVariable:
        """Create a ``TensorVariable`` that will be used as the random variable's "value" in log-likelihood graphs.

        In general, we'll call this type of variable the "value" variable.

        In all other cases, the role of the value variable is taken by
        observed data. That's why value variables are only referenced in
        this branch of the conditional.

        Parameters
        ----------
        rv_var : TensorVariable

        default_transform: Transform
            A transform for the random variable in log-likelihood space.

        transform: Transform
            Additional transform which may be applied after default transform.

        value_var : Variable, optional

        Returns
        -------
        TensorVariable
        """
        from pymc.distributions.transforms import ChainedTransform, _default_transform

        # Make the value variable a transformed value variable,
        # if there's an applicable transform
        if transform is None and default_transform is UNSET:
            default_transform = None
            warnings.warn(
                "To disable default transform, please use default_transform=None"
                " instead of transform=None. Setting transform to None will"
                " not have any effect in future.",
                UserWarning,
            )

        if default_transform is UNSET:
            if rv_var.owner is None:
                default_transform = None
            else:
                default_transform = _default_transform(rv_var.owner.op, rv_var)

        if transform is UNSET:
            transform = default_transform
        elif transform is not None and default_transform is not None:
            transform = ChainedTransform([default_transform, transform])

        if value_var is None:
            if transform is None:
                # Create value variable with the same type as the RV
                value_var = rv_var.type()
                value_var.name = rv_var.name
                if pytensor.config.compute_test_value != "off":
                    value_var.tag.test_value = rv_var.tag.test_value
            else:
                # Create value variable with the same type as the transformed RV
                value_var = transform.forward(rv_var, *rv_var.owner.inputs).type()
                value_var.name = f"{rv_var.name}_{transform.name}__"
                value_var.tag.transform = transform
                if pytensor.config.compute_test_value != "off":
                    value_var.tag.test_value = transform.forward(
                        rv_var, *rv_var.owner.inputs
                    ).tag.test_value

        _add_future_warning_tag(value_var)
        rv_var.tag.value_var = value_var

        self.rvs_to_transforms[rv_var] = transform
        self.rvs_to_values[rv_var] = value_var
        self.values_to_rvs[value_var] = rv_var

        return value_var

    def register_data_var(self, data, dims=None):
        """Register a data variable with the model."""
        self.data_vars.append(data)
        self.add_named_variable(data, dims=dims)

    def add_named_variable(self, var, dims: tuple[str | None, ...] | None = None):
        """Add a random graph variable to the named variables of the model.

        This can include several types of variables such basic_RVs, Data, Deterministics,
        and Potentials.

        Parameters
        ----------
        var

        dims : tuple, optional

        """
        if var.name is None:
            raise ValueError("Variable is unnamed.")
        if self.named_vars.tree_contains(var.name):
            raise ValueError(f"Variable name {var.name} already exists.")

        if dims is not None:
            if isinstance(dims, str):
                dims = (dims,)
            for dim in dims:
                if dim not in self.coords and dim is not None:
                    raise ValueError(f"Dimension {dim} is not specified in `coords`.")
            if any(var.name == dim for dim in dims if dim is not None):
                raise ValueError(f"Variable `{var.name}` has the same name as its dimension label.")
            # This check implicitly states that only vars with .ndim attribute can have dims
            if var.ndim != len(dims):
                raise ValueError(
                    f"{var} has {var.ndim} dims but {len(dims)} dim labels were provided."
                )
            self.named_vars_to_dims[var.name] = dims

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
        """Check if name has prefix and adds if needed."""
        name = self._validate_name(name)
        if self.prefix:
            if not name.startswith(self.prefix + "::"):
                return f"{self.prefix}::{name}"
            else:
                return name
        else:
            return name

    def name_of(self, name):
        """Check if name has prefix and deletes if needed."""
        name = self._validate_name(name)
        if not self.prefix or not name:
            return name
        elif name.startswith(self.prefix + "::"):
            return name[len(self.prefix) + 2 :]
        else:
            return name

    def __getitem__(self, key):
        """Get the variable named `key`."""
        try:
            return self.named_vars[key]
        except KeyError as e:
            try:
                return self.named_vars[self.name_for(key)]
            except KeyError:
                raise e

    def __contains__(self, key):
        """Check if the model contains a variable named `key`."""
        return key in self.named_vars or self.name_for(key) in self.named_vars

    def __copy__(self):
        """Clone the model."""
        return self.copy()

    def __deepcopy__(self, _):
        """Clone the model."""
        return self.copy()

    def copy(self):
        """
        Clone the model.

        To access variables in the cloned model use `cloned_model["var_name"]`.

        Examples
        --------
        .. code-block:: python

            import pymc as pm
            import copy

            with pm.Model() as m:
                p = pm.Beta("p", 1, 1)
                x = pm.Bernoulli("x", p=p, shape=(3,))

            clone_m = copy.copy(m)

            # Access cloned variables by name
            clone_x = clone_m["x"]

            # z will be part of clone_m but not m
            z = pm.Deterministic("z", clone_x + 1)
        """
        from pymc.model.fgraph import clone_model

        return clone_model(self)

    def replace_rvs_by_values(
        self,
        graphs: Sequence[TensorVariable],
        **kwargs,
    ) -> list[TensorVariable]:
        """Clone and replace random variables in graphs with their value variables.

        This will *not* recompute test values in the resulting graphs.

        Parameters
        ----------
        graphs : array_like
            The graphs in which to perform the replacements.

        Returns
        -------
        array_like
        """
        return replace_rvs_by_values(
            graphs,
            rvs_to_values=self.rvs_to_values,
            rvs_to_transforms=self.rvs_to_transforms,
        )

    @overload
    def compile_fn(
        self,
        outs: Variable | Sequence[Variable],
        *,
        inputs: Sequence[Variable] | None = None,
        mode=None,
        point_fn: Literal[True] = True,
        **kwargs,
    ) -> PointFunc: ...

    @overload
    def compile_fn(
        self,
        outs: Variable | Sequence[Variable],
        *,
        inputs: Sequence[Variable] | None = None,
        mode=None,
        point_fn: Literal[False],
        **kwargs,
    ) -> Function: ...

    def compile_fn(
        self,
        outs: Variable | Sequence[Variable],
        *,
        inputs: Sequence[Variable] | None = None,
        mode=None,
        point_fn: bool = True,
        **kwargs,
    ) -> PointFunc | Function:
        """Compiles a PyTensor function.

        Parameters
        ----------
        outs : Variable or sequence of Variables
            PyTensor variable or iterable of PyTensor variables.
        inputs : sequence of Variables, optional
            PyTensor input variables, defaults to pytensorf.inputvars(outs).
        mode
            PyTensor compilation mode, default=None.
        point_fn : bool
            Whether to wrap the compiled function in a PointFunc, which takes a Point
            dictionary with model variable names and values as input.
        Other keyword arguments :
            Any other keyword argument is sent to :py:func:`pymc.pytensorf.compile_pymc`.

        Returns
        -------
        Compiled PyTensor function
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
        """Compile and profile a PyTensor function which returns ``outs`` and takes values of model vars as a dict as an argument.

        Parameters
        ----------
        outs : PyTensor variable or iterable of PyTensor variables
        n : int, default 1000
            Number of iterations to run
        point : Point
            Point to pass to the function
        profile : True or ProfileStats
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
            point = self.initial_point()

        for _ in range(n):
            f(**point)

        return f.profile

    def update_start_vals(self, a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
        r"""Update point `a` with `b`, without overwriting existing keys.

        Values specified for transformed variables in `a` will be recomputed
        conditional on the values of `b` and stored in `b`.

        Parameters
        ----------
        a : dict

        b : dict
        """
        raise FutureWarning(
            "The `Model.update_start_vals` method was removed."
            " To change initial values you may set the items of `Model.initial_values` directly."
        )

    def eval_rv_shapes(self) -> dict[str, tuple[int, ...]]:
        """Evaluate shapes of untransformed AND transformed free variables.

        Returns
        -------
        shapes : dict
            Maps untransformed and transformed variable names to shape tuples.
        """
        names = []
        outputs = []
        for rv in self.free_RVs:
            transform = self.rvs_to_transforms[rv]
            if transform is not None:
                names.append(get_transformed_name(rv.name, transform))
                outputs.append(transform.forward(rv, *rv.owner.inputs).shape)
            names.append(rv.name)
            outputs.append(rv.shape)
        f = pytensor.function(
            inputs=[],
            outputs=outputs,
            givens=[(obs, self.rvs_to_values[obs]) for obs in self.observed_RVs],
            mode=pytensor.compile.mode.FAST_COMPILE,
            on_unused_input="ignore",
        )
        return {name: tuple(shape) for name, shape in zip(names, f())}

    def check_start_vals(self, start, **kwargs):
        r"""Check that the logp is defined and finite at the starting point.

        Parameters
        ----------
        start : dict, or array of dict
            Starting point in parameter space (or partial point)
            Defaults to ``trace.point(-1))`` if there is a trace provided and
            ``model.initial_point`` if not (defaults to empty dict). Initialization
            methods for NUTS (see ``init`` keyword) can overwrite the default.
        Other keyword arguments :
            Any other keyword argument is sent to :py:meth:`~pymc.model.core.Model.point_logps`.

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

        value_names_to_dtypes = {value.name: value.dtype for value in self.value_vars}
        value_names_set = set(value_names_to_dtypes.keys())
        for elem in start_points:
            for k, v in elem.items():
                elem[k] = np.asarray(v, dtype=value_names_to_dtypes[k])

            if not set(elem.keys()).issubset(value_names_set):
                extra_keys = ", ".join(set(elem.keys()) - value_names_set)
                valid_keys = ", ".join(value_names_set)
                raise KeyError(
                    "Some start parameters do not appear in the model!\n"
                    f"Valid keys are: {valid_keys}, but {extra_keys} was supplied"
                )

            initial_eval = self.point_logps(point=elem, **kwargs)

            if not all(np.isfinite(v) for v in initial_eval.values()):
                raise SamplingError(
                    "Initial evaluation of model at starting point failed!\n"
                    f"Starting values:\n{elem}\n\n"
                    f"Logp initial evaluation results:\n{initial_eval}\n"
                    "You can call `model.debug()` for more details."
                )

    def point_logps(self, point=None, round_vals=2, **kwargs):
        """Compute the log probability of `point` for all random variables in the model.

        Parameters
        ----------
        point : Point, optional
            Point to be evaluated.  If ``None``, then ``model.initial_point``
            is used.
        round_vals : int, default 2
            Number of decimals to round log-probabilities.
        Other keyword arguments :
            Any other keyword argument are sent provided to :py:meth:`~pymc.model.core.Model.compile_fn`

        Returns
        -------
        log_probability_of_point : dict
            Log probability of `point`.
        """
        if point is None:
            point = self.initial_point()

        factors = self.basic_RVs + self.potentials
        factor_logps_fn = [pt.sum(factor) for factor in self.logp(factors, sum=False)]
        return {
            factor.name: np.round(np.asarray(factor_logp), round_vals)
            for factor, factor_logp in zip(
                factors,
                self.compile_fn(factor_logps_fn, **kwargs)(point),
            )
        }

    def debug(
        self,
        point: dict[str, np.ndarray] | None = None,
        fn: Literal["logp", "dlogp", "random"] = "logp",
        verbose: bool = False,
    ):
        """Debug model function at point.

        The method will evaluate the `fn` for each variable at a time.
        When an evaluation fails or produces a non-finite value we print:
         1. The graph of the parameters
         2. The value of the parameters (if those can be evaluated)
         3. The output of `fn` (if it can be evaluated)

        This function should help to quickly narrow down invalid parametrizations.

        Parameters
        ----------
        point : Point, optional
            Point at which model function should be evaluated
        fn : str, default "logp"
            Function to be used for debugging. Can be one of [logp, dlogp, random].
        verbose : bool, default False
            Whether to show a more verbose PyTensor output when function cannot be evaluated
        """
        print_ = functools.partial(print, file=sys.stdout)

        def first_line(exc):
            return exc.args[0].split("\n")[0]

        def debug_parameters(rv):
            if isinstance(rv.owner.op, RandomVariable):
                inputs = rv.owner.op.dist_params(rv.owner)
            else:
                inputs = [inp for inp in rv.owner.inputs if not isinstance(inp.type, RandomType)]
            rv_inputs = pytensor.function(
                self.value_vars,
                self.replace_rvs_by_values(inputs),
                on_unused_input="ignore",
                mode=get_mode(None).excluding("inplace", "fusion"),
            )

            print_(f"The variable {rv} has the following parameters:")
            # done and used_ids are used to keep the same ids across distinct dprint calls
            done = {}
            used_ids = {}
            for i, out in enumerate(rv_inputs.maker.fgraph.outputs):
                print_(f"{i}: ", end="")
                # Don't print useless deepcopys
                if out.owner and isinstance(out.owner.op, DeepCopyOp):
                    out = out.owner.inputs[0]
                pytensor.dprint(out, print_type=True, done=done, used_ids=used_ids)

            try:
                print_("The parameters evaluate to:")
                for i, rv_input_eval in enumerate(rv_inputs(**point)):
                    print_(f"{i}: {rv_input_eval}")
            except Exception as exc:
                print_(
                    f"The parameters of the variable {rv} cannot be evaluated: {first_line(exc)}"
                )
                if verbose:
                    print_(exc, "\n")

        if fn not in ("logp", "dlogp", "random"):
            raise ValueError(f"fn must be one of [logp, dlogp, random], got {fn}")

        if point is None:
            point = self.initial_point()
        print_(f"point={point}\n")

        rvs_to_check = list(self.basic_RVs)
        if fn in ("logp", "dlogp"):
            rvs_to_check += [self.replace_rvs_by_values(p) for p in self.potentials]

        found_problem = False
        for rv in rvs_to_check:
            if fn == "logp":
                rv_fn = pytensor.function(
                    self.value_vars, self.logp(vars=rv, sum=False)[0], on_unused_input="ignore"
                )
            elif fn == "dlogp":
                rv_fn = pytensor.function(
                    self.value_vars, self.dlogp(vars=rv), on_unused_input="ignore"
                )
            else:
                [rv_inputs_replaced] = replace_rvs_by_values(
                    [rv],
                    # Don't include itself, or the function will just the the value variable
                    rvs_to_values={
                        rv_key: value
                        for rv_key, value in self.rvs_to_values.items()
                        if rv_key is not rv
                    },
                    rvs_to_transforms=self.rvs_to_transforms,
                )
                rv_fn = pytensor.function(
                    self.value_vars, rv_inputs_replaced, on_unused_input="ignore"
                )

            try:
                rv_fn_eval = rv_fn(**point)
            except ParameterValueError as exc:
                found_problem = True
                debug_parameters(rv)
                print_(
                    f"This does not respect one of the following constraints: {first_line(exc)}\n"
                )
                if verbose:
                    print_(exc)
            except Exception as exc:
                found_problem = True
                debug_parameters(rv)
                print_(
                    f"The variable {rv} {fn} method raised the following exception: {first_line(exc)}\n"
                )
                if verbose:
                    print_(exc)
            else:
                if not np.all(np.isfinite(rv_fn_eval)):
                    found_problem = True
                    debug_parameters(rv)
                    if fn == "random" or rv is self.potentials:
                        print_("This combination seems able to generate non-finite values")
                    else:
                        # Find which values are associated with non-finite evaluation
                        values = self.rvs_to_values[rv]
                        if rv in self.observed_RVs:
                            values = values.eval()
                        else:
                            values = point[values.name]

                        observed = " observed " if rv in self.observed_RVs else " "
                        print_(
                            f"Some of the{observed}values of variable {rv} are associated with a non-finite {fn}:"
                        )
                        mask = ~np.isfinite(rv_fn_eval)
                        for value, fn_eval in zip(values[mask], rv_fn_eval[mask]):
                            print_(f" value = {value} -> {fn} = {fn_eval}")
                        print_()

        if not found_problem:
            print_("No problems found")
        elif not verbose:
            print_("You can set `verbose=True` for more details")

    def to_graphviz(
        self,
        *,
        var_names: Iterable[VarName] | None = None,
        formatting: str = "plain",
        save: str | None = None,
        figsize: tuple[int, int] | None = None,
        dpi: int = 300,
    ):
        """Produce a graphviz Digraph from a PyMC model.

        Requires graphviz, which may be installed most easily with
            conda install -c conda-forge python-graphviz

        Alternatively, you may install the `graphviz` binaries yourself,
        and then `pip install graphviz` to get the python bindings.  See
        http://graphviz.readthedocs.io/en/stable/manual.html
        for more information.

        Parameters
        ----------
        var_names : iterable of variable names, optional
            Subset of variables to be plotted that identify a subgraph with respect to the entire model graph
        formatting : str, optional
            one of { "plain" }
        save : str, optional
            If provided, an image of the graph will be saved to this location. The format is inferred from
            the file extension.
        figsize : tuple[int, int], optional
            Width and height of the figure in inches. If not provided, uses the default figure size. It only affect
            the size of the saved figure.
        dpi : int, optional
            Dots per inch. It only affects the resolution of the saved figure. The default is 300.

        Examples
        --------
        How to plot the graph of the model.

        .. code-block:: python

            import numpy as np
            from pymc import HalfCauchy, Model, Normal

            J = 8
            y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
            sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

            with Model() as schools:
                eta = Normal("eta", 0, 1, shape=J)
                mu = Normal("mu", 0, sigma=1e6)
                tau = HalfCauchy("tau", 25)

                theta = mu + tau * eta

                obs = Normal("obs", theta, sigma=sigma, observed=y)

            schools.to_graphviz()

        Note that this code automatically plots the graph if executed in a Jupyter notebook.
        If executed non-interactively, such as in a script or python console, the graph
        needs to be rendered explicitly:

        .. code-block:: python

            # creates the file `schools.pdf`
            schools.to_graphviz().render("schools")
        """
        return model_to_graphviz(
            model=self,
            var_names=var_names,
            formatting=formatting,
            save=save,
            figsize=figsize,
            dpi=dpi,
        )


class BlockModelAccess(Model):
    """Can be used to prevent user access to Model contexts."""

    def __init__(self, *args, error_msg_on_access="Model access is blocked", **kwargs):
        self.error_msg_on_access = error_msg_on_access


def new_or_existing_block_model_access(*args, **kwargs):
    """Return a BlockModelAccess in the stack or create a new one if none is found."""
    model = Model.get_context(error_if_none=False, allow_block_model_access=True)
    if isinstance(model, BlockModelAccess):
        return model
    return BlockModelAccess(*args, **kwargs)


def set_data(new_data, model=None, *, coords=None):
    """Set the value of one or more data container variables.

    Note that the shape is also dynamic, it is updated when the value is
    changed.  See the examples below for two common use-cases that take
    advantage of this behavior.

    Parameters
    ----------
    new_data: dict
        New values for the data containers. The keys of the dictionary are
        the variables' names in the model and the values are the objects
        with which to update.
    model: Model (optional if in `with` context)

    Examples
    --------
    This example shows how to change the shape of the likelihood to correspond automatically with
    `x`, the predictor in a regression model.

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y, shape=x.shape)
            idata = pm.sample()

    Then change the value of `x` to predict on new data.

    .. code-block:: python

        with model:
            pm.set_data({'x': [5., 6., 9., 12., 15.]})
            y_test = pm.sample_posterior_predictive(idata)

        print(y_test.posterior_predictive['obs'].mean(('chain', 'draw')))

        >>> array([4.6088569 , 5.54128318, 8.32953844, 11.14044852, 13.94178173])

    This example shows how to reuse the same model without recompiling on a new data set.  The
    shape of the likelihood, `obs`, automatically tracks the shape of the observed data, `y`.

    .. code-block:: python

        import numpy as np
        import pymc as pm

        rng = np.random.default_rng()
        data = rng.normal(loc=1.0, scale=2.0, size=100)

        with pm.Model() as model:
            y = pm.Data("y", data)
            theta = pm.Normal("theta", mu=0.0, sigma=10.0)
            obs = pm.Normal("obs", theta, 2.0, observed=y, shape=y.shape)
            idata = pm.sample()

    Now update the model with a new data set.

    .. code-block:: python

        with model:
            pm.set_data({"y": rng.normal(loc=1.0, scale=2.0, size=200)})
            idata = pm.sample()
    """
    model = modelcontext(model)

    for variable_name, new_value in new_data.items():
        model.set_data(variable_name, new_value, coords=coords)


def compile_fn(
    outs: Variable | Sequence[Variable],
    *,
    inputs: Sequence[Variable] | None = None,
    mode=None,
    point_fn: bool = True,
    model: Model | None = None,
    **kwargs,
) -> PointFunc | Function:
    """Compiles a PyTensor function.

    Parameters
    ----------
    outs
        PyTensor variable or iterable of PyTensor variables.
    inputs
        PyTensor input variables, defaults to pytensorf.inputvars(outs).
    mode
        PyTensor compilation mode, default=None.
    point_fn : bool
        Whether to wrap the compiled function in a PointFunc, which takes a Point
        dictionary with model variable names and values as input.
    model : Model, optional
        Current model on stack.

    Returns
    -------
    Compiled PyTensor function
    """
    model = modelcontext(model)
    return model.compile_fn(
        outs,
        inputs=inputs,
        mode=mode,
        point_fn=point_fn,
        **kwargs,
    )


def Point(*args, filter_model_vars=False, **kwargs) -> dict[VarName, np.ndarray]:
    """Build a point.

    Uses same args as dict() does.
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


def Deterministic(name, var, model=None, dims=None):
    """Create a named deterministic variable.

    Deterministic nodes are only deterministic given all of their inputs, i.e.
    they don't add randomness to the model.  They are generally used to record
    an intermediary result.

    Parameters
    ----------
    name : str
        Name of the deterministic variable to be registered in the model.
    var : tensor_like
        Expression for the calculation of the variable.
    model : Model, optional
        The model object to which the Deterministic variable is added.
        If ``None`` is provided, the current model in the context stack is used.
    dims : str or tuple of str, optional
        Dimension names for the variable.

    Returns
    -------
    var : tensor_like
        The registered, named variable wrapped in Deterministic.

    Examples
    --------
    Indeed, PyMC allows for arbitrary combinations of random variables, for
    example in the case of a logistic regression

    .. code:: python

        with pm.Model():
            alpha = pm.Normal("alpha", 0, 1)
            intercept = pm.Normal("intercept", 0, 1)
            p = pm.math.invlogit(alpha * x + intercept)
            outcome = pm.Bernoulli("outcome", p, observed=outcomes)


    but doesn't memorize the fact that the expression ``pm.math.invlogit(alpha *
    x + intercept)`` has been affected to the variable ``p``.  If the quantity
    ``p`` is important and one would like to track its value in the sampling
    trace, then one can use a deterministic node:

    .. code:: python

        with pm.Model():
            alpha = pm.Normal("alpha", 0, 1)
            intercept = pm.Normal("intercept", 0, 1)
            p = pm.Deterministic("p", pm.math.invlogit(alpha * x + intercept))
            outcome = pm.Bernoulli("outcome", p, observed=outcomes)

    These two models are strictly equivalent from a mathematical point of view.
    However, in the first case, the inference data will only contain values for
    the variables ``alpha``, ``intercept`` and ``outcome``.  In the second, it
    will also contain sampled values of ``p`` for each of the observed points.

    Notes
    -----
    Even though adding a Deterministic node forces PyMC to compute this
    expression, which could have been optimized away otherwise, this doesn't come
    with a performance cost.  Indeed, Deterministic nodes are computed outside
    the main computation graph, which can be optimized as though there was no
    Deterministic nodes.  Whereas the optimized graph can be evaluated thousands
    of times during a NUTS step, the Deterministic quantities are just
    computeed once at the end of the step, with the final values of the other
    random variables.
    """
    model = modelcontext(model)
    var = var.copy(model.name_for(name))
    model.deterministics.append(var)
    model.add_named_variable(var, dims)

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


def Potential(name, var: TensorVariable, model=None, dims=None) -> TensorVariable:
    """Add an arbitrary term to the model log-probability.

    Warnings
    --------
    Potential terms only influence probability-based sampling, such as ``pm.sample``, but not forward sampling like
    ``pm.sample_prior_predictive`` or ``pm.sample_posterior_predictive``. A warning is raised when doing forward
    sampling with models containing Potential terms.

    Parameters
    ----------
    name : str
        Name of the potential variable to be registered in the model.
    var : tensor_like
        Expression to be added to the model joint logp.
    model : Model, optional
        The model object to which the potential function is added.
        If ``None`` is provided, the current model in the context stack is used.
    dims : str or tuple of str, optional
        Dimension names for the variable.

    Returns
    -------
    var : tensor_like
        The registered, named model variable.

    Examples
    --------
    In this example, we define a constraint on ``x`` to be greater or equal to 0.
    The statement ``pm.math.log(pm.math.switch(constraint, 0, 1))`` adds either 0 or -inf to the model logp,
    depending on whether the constraint is met. During sampling, any proposals where ``x`` is negative will be rejected.

    .. code:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)

            constraint = x >= 0
            potential = pm.Potential("x_constraint", pm.math.log(pm.math.switch(constraint, 1, 0)))


    Instead, with a soft constraint like ``pm.math.log(pm.math.switch(constraint, 1, 0.5))``,
    the sampler will be less likely, but not forbidden, from accepting negative values for `x`.

    .. code:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)

            constraint = x >= 0
            potential = pm.Potential("x_constraint", pm.math.log(pm.math.switch(constraint, 1.0, 0.5)))

    A Potential term can depend on multiple variables.
    In the following example, the ``soft_sum_constraint`` potential encourages ``x`` and ``y`` to have a small sum.
    The more the sum deviates from zero, the more negative the penalty value of ``(-((x + y)**2))``.

    .. code:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=10)
            y = pm.Normal("y", mu=0, sigma=10)
            soft_sum_constraint = pm.Potential("soft_sum_constraint", -((x + y)**2))

    A Potential can be used to define a specific prior term.
    The following example imposes a power law prior on `max_items`, under the form ``log(1/max_items)``,
    which penalizes very large values of `max_items`.

    .. code:: python

        import pymc as pm

        with pm.Model() as model:
            # p(max_items) = 1 / max_items
            max_items = pm.Uniform("max_items", lower=1, upper=100)
            pm.Potential("power_prior", pm.math.log(1/max_items))

            n_items = pm.Uniform("n_items", lower=1, upper=max_items, observed=60)

    A Potential can be used to define a specific likelihood term.
    In the following example, a normal likelihood term is added to fixed data.
    The same result would be obtained by using an observed `Normal` variable.

    .. code:: python

        import pymc as pm

        def normal_logp(value, mu, sigma):
            return -0.5 * ((value - mu) / sigma) ** 2 - pm.math.log(sigma)

        with pm.Model() as model:
            mu = pm.Normal("x")
            sigma = pm.HalfNormal("sigma")

            data = [0.1, 0.5, 0.9]
            llike = pm.Potential("llike", normal_logp(data, mu, sigma))


    """
    model = modelcontext(model)
    var.name = model.name_for(name)
    model.potentials.append(var)
    model.add_named_variable(var, dims)

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
