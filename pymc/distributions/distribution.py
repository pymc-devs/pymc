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
import contextvars
import functools
import sys
import types
import warnings

from abc import ABCMeta
from functools import singledispatch
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import opcode

from aesara import tensor as at
from aesara.compile.builders import OpFromGraph
from aesara.graph import node_rewriter
from aesara.graph.basic import Node, clone_replace
from aesara.graph.rewriting.basic import in2out
from aesara.graph.utils import MetaType
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.type import RandomType
from aesara.tensor.var import TensorVariable
from typing_extensions import TypeAlias

from pymc.aesaraf import convert_observed_data
from pymc.distributions.shape_utils import (
    Dims,
    Shape,
    convert_dims,
    convert_shape,
    convert_size,
    find_size,
    shape_from_dims,
)
from pymc.logprob.abstract import (
    MeasurableVariable,
    _get_measurable_outputs,
    _icdf,
    _logcdf,
    _logprob,
)
from pymc.logprob.rewriting import logprob_rewrites_db
from pymc.printing import str_for_dist
from pymc.util import UNSET, _add_future_warning_tag
from pymc.vartypes import string_types

__all__ = [
    "DensityDistRV",
    "DensityDist",
    "Distribution",
    "Continuous",
    "Discrete",
    "SymbolicRandomVariable",
]

DIST_PARAMETER_TYPES: TypeAlias = Union[np.ndarray, int, float, TensorVariable]

vectorized_ppc: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar(
    "vectorized_ppc", default=None
)

PLATFORM = sys.platform


class _Unpickling:
    pass


class DistributionMeta(ABCMeta):
    """
    DistributionMeta class


    Notes
    -----
    DistributionMeta currently performs many functions, and will likely be refactored soon.
    See issue below for more details
    https://github.com/pymc-devs/pymc/issues/5308
    """

    def __new__(cls, name, bases, clsdict):

        # Forcefully deprecate old v3 `Distribution`s
        if "random" in clsdict:

            def _random(*args, **kwargs):
                warnings.warn(
                    "The old `Distribution.random` interface is deprecated.",
                    FutureWarning,
                    stacklevel=2,
                )
                return clsdict["random"](*args, **kwargs)

            clsdict["random"] = _random

        rv_op = clsdict.setdefault("rv_op", None)
        rv_type = None

        if isinstance(rv_op, RandomVariable):
            rv_type = type(rv_op)
            clsdict["rv_type"] = rv_type

        new_cls = super().__new__(cls, name, bases, clsdict)

        if rv_type is not None:
            # Create dispatch functions

            class_logp = clsdict.get("logp")
            if class_logp:

                @_logprob.register(rv_type)
                def logp(op, values, *dist_params, **kwargs):
                    dist_params = dist_params[3:]
                    (value,) = values
                    return class_logp(value, *dist_params)

            class_logcdf = clsdict.get("logcdf")
            if class_logcdf:

                @_logcdf.register(rv_type)
                def logcdf(op, value, *dist_params, **kwargs):
                    dist_params = dist_params[3:]
                    return class_logcdf(value, *dist_params)

            class_icdf = clsdict.get("icdf")
            if class_icdf:

                @_icdf.register(rv_type)
                def icdf(op, value, *dist_params, **kwargs):
                    dist_params = dist_params[3:]
                    return class_icdf(value, *dist_params)

            class_moment = clsdict.get("moment")
            if class_moment:

                @_moment.register(rv_type)
                def moment(op, rv, rng, size, dtype, *dist_params):
                    return class_moment(rv, size, *dist_params)

            # Register the Aesara `RandomVariable` type as a subclass of this
            # `Distribution` type.
            new_cls.register(rv_type)

        return new_cls


def _make_nice_attr_error(oldcode: str, newcode: str):
    def fn(*args, **kwargs):
        raise AttributeError(f"The `{oldcode}` method was removed. Instead use `{newcode}`.`")

    return fn


# Helper function from pyprob
def _extract_target_of_assignment(depth):
    frame = sys._getframe(depth)
    code = frame.f_code
    next_instruction = code.co_code[frame.f_lasti + 2]
    instruction_arg = code.co_code[frame.f_lasti + 3]
    instruction_name = opcode.opname[next_instruction]
    if instruction_name == "STORE_FAST":
        return code.co_varnames[instruction_arg]
    elif instruction_name in ["STORE_NAME", "STORE_GLOBAL"]:
        return code.co_names[instruction_arg]
    elif (
        instruction_name in ["LOAD_FAST", "LOAD_NAME", "LOAD_GLOBAL"]
        and opcode.opname[code.co_code[frame.f_lasti + 4]] in ["LOAD_CONST", "LOAD_FAST"]
        and opcode.opname[code.co_code[frame.f_lasti + 6]] == "STORE_SUBSCR"
    ):
        if instruction_name == "LOAD_FAST":
            base_name = code.co_varnames[instruction_arg]
        else:
            base_name = code.co_names[instruction_arg]

        second_instruction = opcode.opname[code.co_code[frame.f_lasti + 4]]
        second_arg = code.co_code[frame.f_lasti + 5]
        if second_instruction == "LOAD_CONST":
            value = code.co_consts[second_arg]
        elif second_instruction == "LOAD_FAST":
            var_name = code.co_varnames[second_arg]
            value = frame.f_locals[var_name]
        else:
            value = None
        if value is not None:
            index_name = repr(value)
            return base_name + "[" + index_name + "]"
        else:
            return None
    else:
        return None


class SymbolicRandomVariable(OpFromGraph):
    """Symbolic Random Variable

    This is a subclasse of `OpFromGraph` which is used to encapsulate the symbolic
    random graph of complex distributions which are built on top of pure
    `RandomVariable`s.

    These graphs may vary structurally based on the inputs (e.g., their dimensionality),
    and usually require that random inputs have specific shapes for correct outputs
    (e.g., avoiding broadcasting of random inputs). Due to this, most distributions that
    return SymbolicRandomVariable create their these graphs at runtime via the
    classmethod `cls.rv_op`, taking care to clone and resize random inputs, if needed.
    """

    ndim_supp: int = None
    """Number of support dimensions as in RandomVariables
    (0 for scalar, 1 for vector, ...)
     """

    inline_logprob: bool = False
    """Specifies whether the logprob function is derived automatically by introspection
    of the inner graph.

    If `False`, a logprob function must be dispatched directly to the subclass type.
    """

    _print_name: Tuple[str, str] = ("Unknown", "\\operatorname{Unknown}")
    """Tuple of (name, latex name) used for for pretty-printing variables of this type"""

    def __init__(self, *args, ndim_supp, **kwargs):
        self.ndim_supp = ndim_supp
        kwargs.setdefault("inline", True)
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        """Symbolic update expression for input random state variables

        Returns a dictionary with the symbolic expressions required for correct updating
        of random state input variables repeated function evaluations. This is used by
        `aesaraf.compile_pymc`.
        """
        return {}


class Distribution(metaclass=DistributionMeta):
    """Statistical distribution"""

    rv_op: [RandomVariable, SymbolicRandomVariable] = None
    rv_type: MetaType = None

    def __new__(
        cls,
        *args,
        rng=None,
        dims: Optional[Dims] = None,
        initval=None,
        observed=None,
        total_size=None,
        transform=UNSET,
        **kwargs,
    ) -> TensorVariable:
        """Adds a tensor variable corresponding to a PyMC distribution to the current model.

        Note that all remaining kwargs must be compatible with ``.dist()``

        Parameters
        ----------
        cls : type
            A PyMC distribution.
        rng : optional
            Random number generator to use with the RandomVariable.
        dims : tuple, optional
            A tuple of dimension names known to the model. When shape is not provided,
            the shape of dims is used to define the shape of the variable.
        initval : optional
            Numeric or symbolic untransformed initial value of matching shape,
            or one of the following initial value strategies: "moment", "prior".
            Depending on the sampler's settings, a random jitter may be added to numeric, symbolic
            or moment-based initial values in the transformed space.
        observed : optional
            Observed data to be passed when registering the random variable in the model.
            When neither shape nor dims is provided, the shape of observed is used to
            define the shape of the variable.
            See ``Model.register_rv``.
        total_size : float, optional
            See ``Model.register_rv``.
        transform : optional
            See ``Model.register_rv``.
        **kwargs
            Keyword arguments that will be forwarded to ``.dist()`` or the Aesara RV Op.
            Most prominently: ``shape`` for ``.dist()`` or ``dtype`` for the Op.

        Returns
        -------
        rv : TensorVariable
            The created random variable tensor, registered in the Model.
        """

        try:
            from pymc.model import Model

            model = Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to "
                "instantiate distributions. Add variable inside "
                "a 'with model:' block, or use the '.dist' syntax "
                "for a standalone distribution."
            )

        if "name" in kwargs:
            name = kwargs.pop("name")
        elif len(args) > 0 and isinstance(args[0], string_types):
            name = args[0]
            args = args[1:]
        else:
            name = _extract_target_of_assignment(2)
            if name is None:
                raise TypeError(
                    "Name could not be inferred for variable from surrounding "
                    "context. Pass a name explicitly as the first argument to "
                    "the Distribution."
                )

        if not isinstance(name, string_types):
            raise TypeError(f"Name needs to be a string but got: {name}")

        if "testval" in kwargs:
            initval = kwargs.pop("testval")
            warnings.warn(
                "The `testval` argument is deprecated; use `initval`.",
                FutureWarning,
                stacklevel=2,
            )

        dims = convert_dims(dims)
        if observed is not None:
            observed = convert_observed_data(observed)

        # Preference is given to size or shape. If not specified, we rely on dims and
        # finally, observed, to determine the shape of the variable.
        if kwargs.get("size") is None and kwargs.get("shape") is None:
            if dims is not None:
                kwargs["shape"] = shape_from_dims(dims, model)
            elif observed is not None:
                kwargs["shape"] = tuple(observed.shape)

        rv_out = cls.dist(*args, **kwargs)

        rv_out = model.register_rv(
            rv_out,
            name,
            observed,
            total_size,
            dims=dims,
            transform=transform,
            initval=initval,
        )

        # add in pretty-printing support
        rv_out.str_repr = types.MethodType(str_for_dist, rv_out)
        rv_out._repr_latex_ = types.MethodType(
            functools.partial(str_for_dist, formatting="latex"), rv_out
        )

        rv_out.logp = _make_nice_attr_error("rv.logp(x)", "pm.logp(rv, x)")
        rv_out.logcdf = _make_nice_attr_error("rv.logcdf(x)", "pm.logcdf(rv, x)")
        rv_out.random = _make_nice_attr_error("rv.random()", "pm.draw(rv)")
        return rv_out

    @classmethod
    def dist(
        cls,
        dist_params,
        *,
        shape: Optional[Shape] = None,
        **kwargs,
    ) -> TensorVariable:
        """Creates a tensor variable corresponding to the `cls` distribution.

        Parameters
        ----------
        dist_params : array-like
            The inputs to the `RandomVariable` `Op`.
        shape : int, tuple, Variable, optional
            A tuple of sizes for each dimension of the new RV.
        **kwargs
            Keyword arguments that will be forwarded to the Aesara RV Op.
            Most prominently: ``size`` or ``dtype``.

        Returns
        -------
        rv : TensorVariable
            The created random variable tensor.
        """
        if "testval" in kwargs:
            kwargs.pop("testval")
            warnings.warn(
                "The `.dist(testval=...)` argument is deprecated and has no effect. "
                "Initial values for sampling/optimization can be specified with `initval` in a modelcontext. "
                "For using Aesara's test value features, you must assign the `.tag.test_value` yourself.",
                FutureWarning,
                stacklevel=2,
            )
        if "initval" in kwargs:
            raise TypeError(
                "Unexpected keyword argument `initval`. "
                "This argument is not available for the `.dist()` API."
            )

        if "dims" in kwargs:
            raise NotImplementedError("The use of a `.dist(dims=...)` API is not supported.")
        size = kwargs.pop("size", None)
        if shape is not None and size is not None:
            raise ValueError(
                f"Passing both `shape` ({shape}) and `size` ({size}) is not supported!"
            )

        shape = convert_shape(shape)
        size = convert_size(size)

        # SymbolicRVs don't have `ndim_supp` until they are created
        ndim_supp = getattr(cls.rv_op, "ndim_supp", None)
        if ndim_supp is None:
            ndim_supp = cls.rv_op(*dist_params, **kwargs).owner.op.ndim_supp
        create_size = find_size(shape=shape, size=size, ndim_supp=ndim_supp)
        rv_out = cls.rv_op(*dist_params, size=create_size, **kwargs)

        rv_out.logp = _make_nice_attr_error("rv.logp(x)", "pm.logp(rv, x)")
        rv_out.logcdf = _make_nice_attr_error("rv.logcdf(x)", "pm.logcdf(rv, x)")
        rv_out.random = _make_nice_attr_error("rv.random()", "pm.draw(rv)")
        _add_future_warning_tag(rv_out)
        return rv_out


# Let PyMC know that the SymbolicRandomVariable has a logprob.
MeasurableVariable.register(SymbolicRandomVariable)


@_get_measurable_outputs.register(SymbolicRandomVariable)
def _get_measurable_outputs_symbolic_random_variable(op, node):
    # This tells PyMC that any non RandomType outputs are measurable

    # Assume that if there is one default_output, that's the only one that is measurable
    # In the rare case this is not what one wants, a specialized _get_measuarable_outputs
    # can dispatch for a subclassed Op
    if op.default_output is not None:
        return [node.default_output()]

    # Otherwise assume that any outputs that are not of RandomType are measurable
    return [out for out in node.outputs if not isinstance(out.type, RandomType)]


@node_rewriter([SymbolicRandomVariable])
def inline_symbolic_random_variable(fgraph, node):
    """
    This optimization expands the internal graph of a SymbolicRV when obtaining the logp
    graph, if the flag `inline_logprob` is True.
    """
    op = node.op
    if op.inline_logprob:
        return clone_replace(op.inner_outputs, {u: v for u, v in zip(op.inner_inputs, node.inputs)})


# Registered before pre-canonicalization which happens at position=-10
logprob_rewrites_db.register(
    "inline_SymbolicRandomVariable",
    in2out(inline_symbolic_random_variable),
    "basic",
    position=-20,
)


@singledispatch
def _moment(op, rv, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(f"Variable {rv} of type {op} has no moment implementation.")


def moment(rv: TensorVariable) -> TensorVariable:
    """Method for choosing a representative point/value
    that can be used to start optimization or MCMC sampling.

    The only parameter to this function is the RandomVariable
    for which the value is to be derived.
    """
    return _moment(rv.owner.op, rv, *rv.owner.inputs).astype(rv.dtype)


class Discrete(Distribution):
    """Base class for discrete distributions"""

    def __new__(cls, name, *args, **kwargs):

        if kwargs.get("transform", None):
            raise ValueError("Transformations for discrete distributions")

        return super().__new__(cls, name, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""


class DensityDistRV(RandomVariable):
    """
    Base class for DensityDistRV

    This should be subclassed when defining custom DensityDist objects.
    """

    name = "DensityDistRV"
    _print_name = ("DensityDist", "\\operatorname{DensityDist}")

    @classmethod
    def rng_fn(cls, rng, *args):
        args = list(args)
        size = args.pop(-1)
        return cls._random_fn(*args, rng=rng, size=size)


class DensityDist(Distribution):
    """A distribution that can be used to wrap black-box log density functions.

    Creates a Distribution and registers the supplied log density function to be used
    for inference. It is also possible to supply a `random` method in order to be able
    to sample from the prior or posterior predictive distributions.


    Parameters
    ----------
    name : str
    dist_params : Tuple
        A sequence of the distribution's parameter. These will be converted into
        Aesara tensors internally. These parameters could be other ``TensorVariable``
        instances created from , optionally created via ``RandomVariable`` ``Op``s.
    class_name : str
        Name for the RandomVariable class which will wrap the DensityDist methods.
        When not specified, it will be given the name of the variable.

        .. warning:: New DensityDists created with the same class_name will override the
            methods dispatched onto the previous classes. If using DensityDists with
            different methods across separate models, be sure to use distinct
            class_names.

    logp : Optional[Callable]
        A callable that calculates the log density of some given observed ``value``
        conditioned on certain distribution parameter values. It must have the
        following signature: ``logp(value, *dist_params)``, where ``value`` is
        an Aesara tensor that represents the observed value, and ``dist_params``
        are the tensors that hold the values of the distribution parameters.
        This function must return an Aesara tensor. If ``None``, a ``NotImplemented``
        error will be raised when trying to compute the distribution's logp.
    logcdf : Optional[Callable]
        A callable that calculates the log cummulative probability of some given observed
        ``value`` conditioned on certain distribution parameter values. It must have the
        following signature: ``logcdf(value, *dist_params)``, where ``value`` is
        an Aesara tensor that represents the observed value, and ``dist_params``
        are the tensors that hold the values of the distribution parameters.
        This function must return an Aesara tensor. If ``None``, a ``NotImplemented``
        error will be raised when trying to compute the distribution's logcdf.
    random : Optional[Callable]
        A callable that can be used to generate random draws from the distribution.
        It must have the following signature: ``random(*dist_params, rng=None, size=None)``.
        The distribution parameters are passed as positional arguments in the
        same order as they are supplied when the ``DensityDist`` is constructed.
        The keyword arguments are ``rnd``, which will provide the random variable's
        associated :py:class:`~numpy.random.Generator`, and ``size``, that will represent
        the desired size of the random draw. If ``None``, a ``NotImplemented``
        error will be raised when trying to draw random samples from the distribution's
        prior or posterior predictive.
    moment : Optional[Callable]
        A callable that can be used to compute the moments of the distribution.
        It must have the following signature: ``moment(rv, size, *rv_inputs)``.
        The distribution's :class:`~aesara.tensor.random.op.RandomVariable` is passed
        as the first argument ``rv``. ``size`` is the random variable's size implied
        by the ``dims``, ``size`` and parameters supplied to the distribution. Finally,
        ``rv_inputs`` is the sequence of the distribution parameters, in the same order
        as they were supplied when the DensityDist was created. If ``None``, a default
        ``moment`` function will be assigned that will always return 0, or an array
        of zeros.
    ndim_supp : int
        The number of dimensions in the support of the distribution. Defaults to assuming
        a scalar distribution, i.e. ``ndim_supp = 0``.
    ndims_params : Optional[Sequence[int]]
        The list of number of dimensions in the support of each of the distribution's
        parameters. If ``None``, it is assumed that all parameters are scalars, hence
        the number of dimensions of their support will be 0.
    dtype : str
        The dtype of the distribution. All draws and observations passed into the distribution
        will be casted onto this dtype.
    kwargs :
        Extra keyword arguments are passed to the parent's class ``__new__`` method.

    Examples
    --------
        .. code-block:: python

            def logp(value, mu):
                return -(value - mu)**2

            with pm.Model():
                mu = pm.Normal('mu',0,1)
                pm.DensityDist(
                    'density_dist',
                    mu,
                    logp=logp,
                    observed=np.random.randn(100),
                )
                idata = pm.sample(100)

        .. code-block:: python

            def logp(value, mu):
                return -(value - mu)**2

            def random(mu, rng=None, size=None):
                return rng.normal(loc=mu, scale=1, size=size)

            with pm.Model():
                mu = pm.Normal('mu', 0 , 1)
                dens = pm.DensityDist(
                    'density_dist',
                    mu,
                    logp=logp,
                    random=random,
                    observed=np.random.randn(100, 3),
                    size=(100, 3),
                )
                prior = pm.sample_prior_predictive(10).prior_predictive['density_dist']
            assert prior.shape == (1, 10, 100, 3)

    """

    rv_type = DensityDistRV

    def __new__(cls, name, *args, **kwargs):
        kwargs.setdefault("class_name", name)
        if isinstance(kwargs.get("observed", None), dict):
            raise TypeError(
                "Since ``v4.0.0`` the ``observed`` parameter should be of type"
                " ``pd.Series``, ``np.array``, or ``pm.Data``."
                " Previous versions allowed passing distribution parameters as"
                " a dictionary in ``observed``, in the current version these "
                "parameters are positional arguments."
            )
        return super().__new__(cls, name, *args, **kwargs)

    @classmethod
    def dist(
        cls,
        *dist_params,
        class_name: str,
        logp: Optional[Callable] = None,
        logcdf: Optional[Callable] = None,
        random: Optional[Callable] = None,
        moment: Optional[Callable] = None,
        ndim_supp: int = 0,
        ndims_params: Optional[Sequence[int]] = None,
        dtype: str = "floatX",
        **kwargs,
    ):

        if dist_params is None:
            dist_params = []
        elif len(dist_params) > 0 and callable(dist_params[0]):
            raise TypeError(
                "The DensityDist API has changed, you are using the old API "
                "where logp was the first positional argument. In the current API, "
                "the logp is a keyword argument, amongst other changes. Please refer "
                "to the API documentation for more information on how to use the "
                "new DensityDist API."
            )
            dist_params = [as_tensor_variable(param) for param in dist_params]

        # Assume scalar ndims_params
        if ndims_params is None:
            ndims_params = [0] * len(dist_params)

        if logp is None:
            logp = default_not_implemented(class_name, "logp")

        if logcdf is None:
            logcdf = default_not_implemented(class_name, "logcdf")

        if moment is None:
            moment = functools.partial(
                default_moment,
                rv_name=class_name,
                has_fallback=random is not None,
                ndim_supp=ndim_supp,
            )

        if random is None:
            random = default_not_implemented(class_name, "random")

        return super().dist(
            dist_params,
            class_name=class_name,
            logp=logp,
            logcdf=logcdf,
            random=random,
            moment=moment,
            ndim_supp=ndim_supp,
            ndims_params=ndims_params,
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        *dist_params,
        class_name: str,
        logp: Optional[Callable],
        logcdf: Optional[Callable],
        random: Optional[Callable],
        moment: Optional[Callable],
        ndim_supp: int,
        ndims_params: Optional[Sequence[int]],
        dtype: str,
        **kwargs,
    ):
        rv_op = type(
            f"DensityDist_{class_name}",
            (DensityDistRV,),
            dict(
                name=f"DensityDist_{class_name}",
                inplace=False,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                dtype=dtype,
                # Specifc to DensityDist
                _random_fn=random,
            ),
        )()

        # Register custom logp
        rv_type = type(rv_op)

        @_logprob.register(rv_type)
        def density_dist_logp(op, value_var_list, *dist_params, **kwargs):
            _dist_params = dist_params[3:]
            value_var = value_var_list[0]
            return logp(value_var, *_dist_params)

        @_logcdf.register(rv_type)
        def density_dist_logcdf(op, var, rvs_to_values, *dist_params, **kwargs):
            value_var = rvs_to_values.get(var, var)
            return logcdf(value_var, *dist_params, **kwargs)

        @_moment.register(rv_type)
        def density_dist_get_moment(op, rv, rng, size, dtype, *dist_params):
            return moment(rv, size, *dist_params)

        return rv_op(*dist_params, **kwargs)


def default_not_implemented(rv_name, method_name):
    message = (
        f"Attempted to run {method_name} on the DensityDist '{rv_name}', "
        f"but this method had not been provided when the distribution was "
        f"constructed. Please re-build your model and provide a callable "
        f"to '{rv_name}'s {method_name} keyword argument.\n"
    )

    def func(*args, **kwargs):
        raise NotImplementedError(message)

    return func


def default_moment(rv, size, *rv_inputs, rv_name=None, has_fallback=False, ndim_supp=0):
    if ndim_supp == 0:
        return at.zeros(size, dtype=rv.dtype)
    elif has_fallback:
        return at.zeros_like(rv)
    else:
        raise TypeError(
            "Cannot safely infer the size of a multivariate random variable's moment. "
            f"Please provide a moment function when instantiating the {rv_name} "
            "random variable."
        )
