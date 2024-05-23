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
import contextvars
import functools
import re
import sys
import types
import warnings

from abc import ABCMeta
from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import TypeAlias

import numpy as np

from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import FunctionGraph, clone_replace, node_rewriter
from pytensor.graph.basic import Apply, Variable, io_toposort
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.rewriting.basic import GraphRewriter, in2out
from pytensor.graph.utils import MetaType
from pytensor.scan.op import Scan
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import safe_signature
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.rewriting import local_subtensor_rv_lift
from pytensor.tensor.random.type import RandomGeneratorType, RandomType
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.utils import _parse_gufunc_signature
from pytensor.tensor.variable import TensorVariable

from pymc.distributions.shape_utils import (
    Dims,
    Shape,
    _change_dist_size,
    convert_dims,
    convert_shape,
    convert_size,
    find_size,
    rv_size_is_none,
    shape_from_dims,
)
from pymc.exceptions import BlockModelAccessError
from pymc.logprob.abstract import MeasurableVariable, _icdf, _logcdf, _logprob
from pymc.logprob.basic import logp
from pymc.logprob.rewriting import logprob_rewrites_db
from pymc.model.core import new_or_existing_block_model_access
from pymc.printing import str_for_dist
from pymc.pytensorf import (
    collect_default_updates,
    collect_default_updates_inner_fgraph,
    constant_fold,
    convert_observed_data,
    floatX,
)
from pymc.util import UNSET, _add_future_warning_tag
from pymc.vartypes import continuous_types, string_types

__all__ = [
    "CustomDist",
    "DensityDist",
    "DiracDelta",
    "Distribution",
    "Continuous",
    "Discrete",
    "SymbolicRandomVariable",
]

DIST_PARAMETER_TYPES: TypeAlias = np.ndarray | int | float | TensorVariable

vectorized_ppc: contextvars.ContextVar[Callable | None] = contextvars.ContextVar(
    "vectorized_ppc", default=None
)

PLATFORM = sys.platform


class FiniteLogpPointRewrite(GraphRewriter):
    def rewrite_support_point_scan_node(self, node):
        if not isinstance(node.op, Scan):
            return

        node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs
        op = node.op

        local_fgraph_topo = io_toposort(node_inputs, node_outputs)

        replace_with_support_point = []
        to_replace_set = set()

        for nd in local_fgraph_topo:
            if nd not in to_replace_set and isinstance(
                nd.op, RandomVariable | SymbolicRandomVariable
            ):
                replace_with_support_point.append(nd.out)
                to_replace_set.add(nd)
        givens = {}
        if len(replace_with_support_point) > 0:
            for item in replace_with_support_point:
                givens[item] = support_point(item)
        else:
            return
        op_outs = clone_replace(node_outputs, replace=givens)

        nwScan = Scan(
            node_inputs,
            op_outs,
            op.info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            name=op.name,
            allow_gc=op.allow_gc,
        )
        nw_node = nwScan(*(node.inputs), return_list=True)[0].owner
        return nw_node

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    def apply(self, fgraph):
        for node in fgraph.toposort():
            if isinstance(node.op, RandomVariable | SymbolicRandomVariable):
                fgraph.replace(node.out, support_point(node.out))
            elif isinstance(node.op, Scan):
                new_node = self.rewrite_support_point_scan_node(node)
                if new_node is not None:
                    fgraph.replace_all(tuple(zip(node.outputs, new_node.outputs)))


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
        rv_type = clsdict.setdefault("rv_type", None)

        if isinstance(rv_op, RandomVariable):
            if rv_type is not None:
                assert isinstance(rv_op, rv_type)
            else:
                rv_type = type(rv_op)
                clsdict["rv_type"] = rv_type

        new_cls = super().__new__(cls, name, bases, clsdict)

        if rv_type is not None:
            # Create dispatch functions

            signature = getattr(rv_type, "signature", None)
            size_idx: int | None = None
            params_idxs: tuple[int] | None = None
            if signature is not None:
                _, size_idx, params_idxs = SymbolicRandomVariable.get_idxs(signature)

            class_change_dist_size = clsdict.get("change_dist_size")
            if class_change_dist_size:

                @_change_dist_size.register(rv_type)
                def change_dist_size(op, rv, new_size, expand):
                    return class_change_dist_size(rv, new_size, expand)

            class_logp = clsdict.get("logp")
            if class_logp:

                @_logprob.register(rv_type)
                def logp(op, values, *dist_params, **kwargs):
                    if isinstance(op, RandomVariable):
                        rng, size, dtype, *dist_params = dist_params
                    elif params_idxs:
                        dist_params = [dist_params[i] for i in params_idxs]
                    [value] = values
                    return class_logp(value, *dist_params)

            class_logcdf = clsdict.get("logcdf")
            if class_logcdf:

                @_logcdf.register(rv_type)
                def logcdf(op, value, *dist_params, **kwargs):
                    if isinstance(op, RandomVariable):
                        rng, size, dtype, *dist_params = dist_params
                    elif params_idxs:
                        dist_params = [dist_params[i] for i in params_idxs]
                    return class_logcdf(value, *dist_params)

            class_icdf = clsdict.get("icdf")
            if class_icdf:

                @_icdf.register(rv_type)
                def icdf(op, value, *dist_params, **kwargs):
                    if isinstance(op, RandomVariable):
                        rng, size, dtype, *dist_params = dist_params
                    elif params_idxs:
                        dist_params = [dist_params[i] for i in params_idxs]
                    return class_icdf(value, *dist_params)

            class_moment = clsdict.get("moment")
            if class_moment:
                warnings.warn(
                    "The moment() method is deprecated. Use support_point() instead.",
                    DeprecationWarning,
                )

                clsdict["support_point"] = class_moment

            class_support_point = clsdict.get("support_point")

            if class_support_point:

                @_support_point.register(rv_type)
                def support_point(op, rv, *dist_params):
                    if isinstance(op, RandomVariable):
                        rng, size, dtype, *dist_params = dist_params
                        return class_support_point(rv, size, *dist_params)
                    elif params_idxs and size_idx is not None:
                        size = dist_params[size_idx]
                        dist_params = [dist_params[i] for i in params_idxs]
                        return class_support_point(rv, size, *dist_params)
                    else:
                        return class_support_point(rv, *dist_params)

            # Register the PyTensor rv_type as a subclass of this PyMC Distribution type.
            new_cls.register(rv_type)

        return new_cls


def _make_nice_attr_error(oldcode: str, newcode: str):
    def fn(*args, **kwargs):
        raise AttributeError(f"The `{oldcode}` method was removed. Instead use `{newcode}`.`")

    return fn


class _class_or_instancemethod(classmethod):
    """Allow a method to be called both as a classmethod and an instancemethod,
    giving priority to the instancemethod.

    This is used to allow extracting information from the signature of a SymbolicRandomVariable
    which may be provided either as a class attribute or as an instance attribute.

    Adapted from https://stackoverflow.com/a/28238047
    """

    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


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

    signature: str = None
    """Numpy-like vectorized signature of the distribution.

    It allows tokens [rng], [size] to identify the special inputs.

    The signature of a Normal RV with mu and scale scalar params looks like
    `"[rng],[size],(),()->[rng],()"`
    """

    inline_logprob: bool = False
    """Specifies whether the logprob function is derived automatically by introspection
    of the inner graph.

    If `False`, a logprob function must be dispatched directly to the subclass type.
    """

    _print_name: tuple[str, str] = ("Unknown", "\\operatorname{Unknown}")
    """Tuple of (name, latex name) used for for pretty-printing variables of this type"""

    @staticmethod
    def _parse_signature(signature: str) -> tuple[str, str]:
        """Parse signature as if special tokens were vector elements"""
        # Regex to split across commas not inside parenthesis
        # Copied from https://stackoverflow.com/a/26634150
        fake_signature = signature.replace("[rng]", "(rng)").replace("[size]", "(size)")
        return _parse_gufunc_signature(fake_signature)

    @staticmethod
    def _parse_params_signature(signature):
        """Parse the signature of the distribution's parameters, ignoring rng and size tokens."""
        special_tokens = r"|".join((r"\[rng\],?", r"\[size\],?"))
        params_signature = re.sub(special_tokens, "", signature)
        # Remove dandling commas
        params_signature = re.sub(r",(?=[->])|,$", "", params_signature)

        # Numpy gufunc signature doesn't accept empty inputs
        if params_signature.startswith("->"):
            # Pretent there was at least one scalar input and then discard that
            return [], _parse_gufunc_signature("()" + params_signature)[1]
        else:
            return _parse_gufunc_signature(params_signature)

    @_class_or_instancemethod
    @property
    def ndims_params(cls_or_self) -> Sequence[int] | None:
        """Number of core dimensions of the distribution's parameters."""
        signature = cls_or_self.signature
        if signature is None:
            return None
        inputs_signature, _ = cls_or_self._parse_params_signature(signature)
        return [len(sig) for sig in inputs_signature]

    @_class_or_instancemethod
    @property
    def ndim_supp(cls_or_self) -> int | None:
        """Number of support dimensions of the RandomVariable

        (0 for scalar, 1 for vector, ...)
        """
        signature = cls_or_self.signature
        if signature is None:
            return None
        _, outputs_params_signature = cls_or_self._parse_params_signature(signature)
        return max(len(out_sig) for out_sig in outputs_params_signature)

    @_class_or_instancemethod
    @property
    def default_output(cls_or_self) -> int | None:
        signature = cls_or_self.signature
        if signature is None:
            return None

        _, outputs_signature = cls_or_self._parse_signature(signature)

        # If there is a single non `[rng]` outputs, that is the default one!
        candidate_default_output = [
            i for i, out_sig in enumerate(outputs_signature) if out_sig != ("rng",)
        ]
        if len(candidate_default_output) == 1:
            return candidate_default_output[0]
        else:
            return None

    @staticmethod
    def get_idxs(signature: str) -> tuple[tuple[int], int | None, tuple[int]]:
        """Parse signature and return indexes for *[rng], [size] and parameters"""
        inputs_signature, outputs_signature = SymbolicRandomVariable._parse_signature(signature)
        rng_idxs = []
        size_idx = None
        params_idxs = []
        for i, inp_sig in enumerate(inputs_signature):
            if inp_sig == ("size",):
                size_idx = i
            elif inp_sig == ("rng",):
                rng_idxs.append(i)
            else:
                params_idxs.append(i)
        return tuple(rng_idxs), size_idx, tuple(params_idxs)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize a SymbolicRandomVariable class."""
        if "signature" in kwargs:
            self.signature = kwargs.pop("signature")

        if "ndim_supp" in kwargs:
            # For backwards compatibility we allow passing ndim_supp without signature
            # This is the only variable that PyMC absolutely needs to work with SymbolicRandomVariables
            self.ndim_supp = kwargs.pop("ndim_supp")

        if self.ndim_supp is None:
            raise ValueError("ndim_supp or signature must be provided")

        kwargs.setdefault("inline", True)
        kwargs.setdefault("strict", True)
        super().__init__(*args, **kwargs)

    def update(self, node: Apply) -> dict[Variable, Variable]:
        """Symbolic update expression for input random state variables

        Returns a dictionary with the symbolic expressions required for correct updating
        of random state input variables repeated function evaluations. This is used by
        `pytensorf.compile_pymc`.
        """
        return collect_default_updates_inner_fgraph(node)

    def batch_ndim(self, node: Apply) -> int:
        """Number of dimensions of the distribution's batch shape."""
        out_ndim = max(getattr(out.type, "ndim", 0) for out in node.outputs)
        return out_ndim - self.ndim_supp


@_change_dist_size.register(SymbolicRandomVariable)
def change_symbolic_rv_size(op, rv, new_size, expand) -> TensorVariable:
    if op.signature is None:
        raise NotImplementedError(
            f"SymbolicRandomVariable {op} without signature requires custom `_change_dist_size` implementation."
        )
    inputs_signature = op.signature.split("->")[0].split(",")
    if "[size]" not in inputs_signature:
        raise NotImplementedError(
            f"SymbolicRandomVariable {op} without [size] in signature requires custom `_change_dist_size` implementation."
        )
    size_arg_idx = inputs_signature.index("[size]")
    size = rv.owner.inputs[size_arg_idx]

    if expand:
        new_size = tuple(new_size) + tuple(size)

    numerical_inputs = [
        inp for inp, sig in zip(rv.owner.inputs, inputs_signature) if sig not in ("[size]", "[rng]")
    ]
    return op.rv_op(*numerical_inputs, size=new_size)


class Distribution(metaclass=DistributionMeta):
    """Statistical distribution"""

    rv_op: [RandomVariable, SymbolicRandomVariable] = None
    rv_type: MetaType = None

    def __new__(
        cls,
        name: str,
        *args,
        rng=None,
        dims: Dims | None = None,
        initval=None,
        observed=None,
        total_size=None,
        transform=UNSET,
        default_transform=UNSET,
        **kwargs,
    ) -> TensorVariable:
        """Adds a tensor variable corresponding to a PyMC distribution to the current model.

        Note that all remaining kwargs must be compatible with ``.dist()``

        Parameters
        ----------
        cls : type
            A PyMC distribution.
        name : str
            Name for the new model variable.
        rng : optional
            Random number generator to use with the RandomVariable.
        dims : tuple, optional
            A tuple of dimension names known to the model. When shape is not provided,
            the shape of dims is used to define the shape of the variable.
        initval : optional
            Numeric or symbolic untransformed initial value of matching shape,
            or one of the following initial value strategies: "support_point", "prior".
            Depending on the sampler's settings, a random jitter may be added to numeric, symbolic
            or support_point-based initial values in the transformed space.
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
            Keyword arguments that will be forwarded to ``.dist()`` or the PyTensor RV Op.
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

        if "testval" in kwargs:
            initval = kwargs.pop("testval")
            warnings.warn(
                "The `testval` argument is deprecated; use `initval`.",
                FutureWarning,
                stacklevel=2,
            )

        if not isinstance(name, string_types):
            raise TypeError(f"Name needs to be a string but got: {name}")

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
            observed=observed,
            total_size=total_size,
            dims=dims,
            transform=transform,
            default_transform=default_transform,
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
        shape: Shape | None = None,
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
            Keyword arguments that will be forwarded to the PyTensor RV Op.
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
                "For using PyTensor's test value features, you must assign the `.tag.test_value` yourself.",
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

        # SymbolicRVs don't always have `ndim_supp` until they are created
        ndim_supp = getattr(cls.rv_type, "ndim_supp", None)
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


@node_rewriter([SymbolicRandomVariable])
def inline_symbolic_random_variable(fgraph, node):
    """
    Optimization that expands the internal graph of a SymbolicRV when obtaining the logp
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
def _support_point(op, rv, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(f"Variable {rv} of type {op} has no support_point implementation.")


def support_point(rv: TensorVariable) -> TensorVariable:
    """Method for choosing a representative point/value
    that can be used to start optimization or MCMC sampling.

    The only parameter to this function is the RandomVariable
    for which the value is to be derived.
    """
    return _support_point(rv.owner.op, rv, *rv.owner.inputs).astype(rv.dtype)


def _moment(op, rv, *rv_inputs) -> TensorVariable:
    warnings.warn(
        "The moment() method is deprecated. Use support_point() instead.",
        DeprecationWarning,
    )
    return _support_point(op, rv, *rv_inputs)


def moment(rv: TensorVariable) -> TensorVariable:
    warnings.warn(
        "The moment() method is deprecated. Use support_point() instead.",
        DeprecationWarning,
    )
    return support_point(rv)


class Discrete(Distribution):
    """Base class for discrete distributions"""

    def __new__(cls, name, *args, **kwargs):
        if kwargs.get("transform", None):
            raise ValueError("Transformations for discrete distributions")

        return super().__new__(cls, name, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""


class CustomDistRV(RandomVariable):
    """
    Base class for CustomDistRV

    This should be subclassed when defining CustomDist objects.
    """

    name = "CustomDistRV"
    _print_name = ("CustomDist", "\\operatorname{CustomDist}")

    @classmethod
    def rng_fn(cls, rng, *args):
        args = list(args)
        size = args.pop(-1)
        return cls._random_fn(*args, rng=rng, size=size)


class _CustomDist(Distribution):
    """A distribution that returns a subclass of CustomDistRV"""

    rv_type = CustomDistRV

    @classmethod
    def dist(
        cls,
        *dist_params,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        random: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        class_name: str = "CustomDist",
        **kwargs,
    ):
        if ndim_supp is None or ndims_params is None:
            if signature is None:
                ndim_supp = 0
                ndims_params = [0] * len(dist_params)
            else:
                inputs, outputs = _parse_gufunc_signature(signature)
                ndim_supp = max(len(out) for out in outputs)
                ndims_params = [len(inp) for inp in inputs]

        if ndim_supp > 0:
            raise NotImplementedError(
                "CustomDist with ndim_supp > 0 and without a `dist` function are not supported."
            )

        dist_params = [as_tensor_variable(param) for param in dist_params]

        if logp is None:
            logp = default_not_implemented(class_name, "logp")

        if logcdf is None:
            logcdf = default_not_implemented(class_name, "logcdf")

        if support_point is None:
            support_point = functools.partial(
                default_support_point,
                rv_name=class_name,
                has_fallback=random is not None,
                ndim_supp=ndim_supp,
            )

        if random is None:
            random = default_not_implemented(class_name, "random")

        return super().dist(
            dist_params,
            logp=logp,
            logcdf=logcdf,
            random=random,
            support_point=support_point,
            ndim_supp=ndim_supp,
            ndims_params=ndims_params,
            dtype=dtype,
            class_name=class_name,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        *dist_params,
        logp: Callable | None,
        logcdf: Callable | None,
        random: Callable | None,
        support_point: Callable | None,
        ndim_supp: int,
        ndims_params: Sequence[int],
        dtype: str,
        class_name: str,
        **kwargs,
    ):
        rv_type = type(
            class_name,
            (CustomDistRV,),
            dict(
                name=class_name,
                inplace=False,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                dtype=dtype,
                _print_name=(class_name, f"\\operatorname{{{class_name}}}"),
                # Specific to CustomDist
                _random_fn=random,
            ),
        )

        # Dispatch custom methods
        @_logprob.register(rv_type)
        def custom_dist_logp(op, values, rng, size, dtype, *dist_params, **kwargs):
            return logp(values[0], *dist_params)

        @_logcdf.register(rv_type)
        def custom_dist_logcdf(op, value, rng, size, dtype, *dist_params, **kwargs):
            return logcdf(value, *dist_params, **kwargs)

        @_support_point.register(rv_type)
        def custom_dist_support_point(op, rv, rng, size, dtype, *dist_params):
            return support_point(rv, size, *dist_params)

        rv_op = rv_type()
        return rv_op(*dist_params, **kwargs)


class CustomSymbolicDistRV(SymbolicRandomVariable):
    """
    Base class for CustomSymbolicDist

    This should be subclassed when defining custom CustomDist objects that have
    symbolic random methods.
    """

    default_output = 0

    _print_name = ("CustomSymbolicDist", "\\operatorname{CustomSymbolicDist}")


@_support_point.register(CustomSymbolicDistRV)
def dist_support_point(op, rv, *args):
    node = rv.owner
    rv_out_idx = node.outputs.index(rv)

    fgraph = op.fgraph.clone()
    replace_support_point = FiniteLogpPointRewrite()
    replace_support_point.rewrite(fgraph)
    # Replace dummy inner inputs by outer inputs
    fgraph.replace_all(tuple(zip(op.inner_inputs, args)), import_missing=True)
    support_point = fgraph.outputs[rv_out_idx]
    return support_point


class _CustomSymbolicDist(Distribution):
    rv_type = CustomSymbolicDistRV

    @classmethod
    def dist(
        cls,
        *dist_params,
        dist: Callable,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        class_name: str = "CustomDist",
        **kwargs,
    ):
        dist_params = [as_tensor_variable(param) for param in dist_params]

        if logcdf is None:
            logcdf = default_not_implemented(class_name, "logcdf")

        if signature is None:
            if ndim_supp is None:
                ndim_supp = 0
            if ndims_params is None:
                ndims_params = [0] * len(dist_params)
            signature = safe_signature(
                core_inputs=[pt.tensor(shape=(None,) * ndim_param) for ndim_param in ndims_params],
                core_outputs=[pt.tensor(shape=(None,) * ndim_supp)],
            )

        return super().dist(
            dist_params,
            class_name=class_name,
            logp=logp,
            logcdf=logcdf,
            dist=dist,
            support_point=support_point,
            signature=signature,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        *dist_params,
        dist: Callable,
        logp: Callable | None,
        logcdf: Callable | None,
        support_point: Callable | None,
        size=None,
        signature: str,
        class_name: str,
    ):
        size = normalize_size_param(size)
        dummy_size_param = size.type()
        dummy_dist_params = [dist_param.type() for dist_param in dist_params]
        with new_or_existing_block_model_access(
            error_msg_on_access="Model variables cannot be created in the dist function. Use the `.dist` API"
        ):
            dummy_rv = dist(*dummy_dist_params, dummy_size_param)
        dummy_params = [dummy_size_param, *dummy_dist_params]
        dummy_updates_dict = collect_default_updates(inputs=dummy_params, outputs=(dummy_rv,))

        rv_type = type(
            class_name,
            (CustomSymbolicDistRV,),
            # If logp is not provided, we try to infer it from the dist graph
            dict(
                inline_logprob=logp is None,
                _print_name=(class_name, f"\\operatorname{{{class_name}}}"),
            ),
        )

        # Dispatch custom methods
        if logp is not None:

            @_logprob.register(rv_type)
            def custom_dist_logp(op, values, size, *inputs, **kwargs):
                [value] = values
                rv_params = inputs[: len(dist_params)]
                return logp(value, *rv_params)

        if logcdf is not None:

            @_logcdf.register(rv_type)
            def custom_dist_logcdf(op, value, size, *inputs, **kwargs):
                rv_params = inputs[: len(dist_params)]
                return logcdf(value, *rv_params)

        if support_point is not None:

            @_support_point.register(rv_type)
            def custom_dist_support_point(op, rv, size, *params):
                return support_point(
                    rv,
                    size,
                    *[
                        p
                        for p in params
                        if not isinstance(p.type, RandomType | RandomGeneratorType)
                    ],
                )

        @_change_dist_size.register(rv_type)
        def change_custom_dist_size(op, rv, new_size, expand):
            node = rv.owner

            if expand:
                shape = tuple(rv.shape)
                old_size = shape[: len(shape) - node.op.ndim_supp]
                new_size = tuple(new_size) + tuple(old_size)
            new_size = pt.as_tensor(new_size, dtype="int64", ndim=1)

            old_size, *old_dist_params = node.inputs[: len(dist_params) + 1]

            # OpFromGraph has to be recreated if the size type changes!
            dummy_size_param = new_size.type()
            dummy_dist_params = [dist_param.type() for dist_param in old_dist_params]
            dummy_rv = dist(*dummy_dist_params, dummy_size_param)
            dummy_params = [dummy_size_param, *dummy_dist_params]
            updates_dict = collect_default_updates(inputs=dummy_params, outputs=(dummy_rv,))
            rngs = updates_dict.keys()
            rngs_updates = updates_dict.values()
            new_rv_op = rv_type(
                inputs=[*dummy_params, *rngs],
                outputs=[dummy_rv, *rngs_updates],
                signature=signature,
            )
            new_rv = new_rv_op(new_size, *dist_params, *rngs)

            return new_rv

        # RNGs are not passed as explicit inputs (because we usually don't know how many are needed)
        # We retrieve them here
        updates_dict = collect_default_updates(inputs=dummy_params, outputs=(dummy_rv,))
        rngs = updates_dict.keys()
        rngs_updates = updates_dict.values()

        inputs = [*dummy_params, *rngs]
        outputs = [dummy_rv, *rngs_updates]
        signature = cls._infer_final_signature(
            signature, n_inputs=len(inputs), n_outputs=len(outputs), n_rngs=len(rngs)
        )
        rv_op = rv_type(
            inputs=inputs,
            outputs=outputs,
            signature=signature,
        )
        return rv_op(size, *dist_params, *rngs)

    @staticmethod
    def _infer_final_signature(signature: str, n_inputs, n_outputs, n_rngs) -> str:
        """Add size and updates to user provided gufunc signature if they are missing."""

        # Regex to split across outer commas
        # Copied from https://stackoverflow.com/a/26634150
        outer_commas = re.compile(r",\s*(?![^()]*\))")

        input_sig, output_sig = signature.split("->")
        # It's valid to have a signature without params inputs, as in a Flat RV
        n_inputs_sig = len(outer_commas.split(input_sig)) if input_sig.strip() else 0
        n_outputs_sig = len(outer_commas.split(output_sig))

        if n_inputs_sig == n_inputs and n_outputs_sig == n_outputs:
            # User provided a signature with no missing parts
            return signature

        size_sig = "[size]"
        rngs_sig = ("[rng]",) * n_rngs
        if n_inputs_sig == (n_inputs - n_rngs - 1):
            # Assume size and rngs are missing
            if input_sig.strip():
                input_sig = ",".join((size_sig, input_sig, *rngs_sig))
            else:
                input_sig = ",".join((size_sig, *rngs_sig))
        if n_outputs_sig == (n_outputs - n_rngs):
            # Assume updates are missing
            output_sig = ",".join((output_sig, *rngs_sig))
        signature = "->".join((input_sig, output_sig))
        return signature


class CustomDist:
    """A helper class to create custom distributions

    This class can be used to wrap black-box random and logp methods for use in
    forward and mcmc sampling.

    A user can provide a `dist` function that returns a PyTensor graph built from
    simpler PyMC distributions, which represents the distribution. This graph is
    used to take random draws, and to infer the logp expression automatically
    when not provided by the user.

    Alternatively, a user can provide a `random` function that returns numerical
    draws (e.g., via NumPy routines), and a `logp` function that must return a
    PyTensor graph that represents the logp graph when evaluated. This is used for
    mcmc sampling.

    Additionally, a user can provide a `logcdf` and `support_point` functions that must return
    PyTensor graphs that computes those quantities. These may be used by other PyMC
    routines.

    Parameters
    ----------
    name : str
    dist_params : Tuple
        A sequence of the distribution's parameter. These will be converted into
        Pytensor tensor variables internally.
    dist: Optional[Callable]
        A callable that returns a PyTensor graph built from simpler PyMC distributions
        which represents the distribution. This can be used by PyMC to take random draws
        as well as to infer the logp of the distribution in some cases. In that case
        it's not necessary to implement ``random`` or ``logp`` functions.

        It must have the following signature: ``dist(*dist_params, size)``.
        The symbolic tensor distribution parameters are passed as positional arguments in
        the same order as they are supplied when the ``CustomDist`` is constructed.

    random : Optional[Callable]
        A callable that can be used to generate random draws from the distribution

        It must have the following signature: ``random(*dist_params, rng=None, size=None)``.
        The numerical distribution parameters are passed as positional arguments in the
        same order as they are supplied when the ``CustomDist`` is constructed.
        The keyword arguments are ``rng``, which will provide the random variable's
        associated :py:class:`~numpy.random.Generator`, and ``size``, that will represent
        the desired size of the random draw. If ``None``, a ``NotImplemented``
        error will be raised when trying to draw random samples from the distribution's
        prior or posterior predictive.

    logp : Optional[Callable]
        A callable that calculates the log probability of some given ``value``
        conditioned on certain distribution parameter values. It must have the
        following signature: ``logp(value, *dist_params)``, where ``value`` is
        an PyTensor tensor that represents the distribution value, and ``dist_params``
        are the tensors that hold the values of the distribution parameters.
        This function must return an PyTensor tensor.

        When the `dist` function is specified, PyMC will try to automatically
        infer the `logp` when this is not provided.

        Otherwise, a ``NotImplementedError`` will be raised when trying to compute the
        distribution's logp.
    logcdf : Optional[Callable]
        A callable that calculates the log cumulative log probability of some given
        ``value`` conditioned on certain distribution parameter values. It must have the
        following signature: ``logcdf(value, *dist_params)``, where ``value`` is
        an PyTensor tensor that represents the distribution value, and ``dist_params``
        are the tensors that hold the values of the distribution parameters.
        This function must return an PyTensor tensor. If ``None``, a ``NotImplementedError``
        will be raised when trying to compute the distribution's logcdf.
    support_point : Optional[Callable]
        A callable that can be used to compute the finete logp point of the distribution.
        It must have the following signature: ``support_point(rv, size, *rv_inputs)``.
        The distribution's variable is passed as the first argument ``rv``. ``size``
        is the random variable's size implied by the ``dims``, ``size`` and parameters
        supplied to the distribution. Finally, ``rv_inputs`` is the sequence of the
        distribution parameters, in the same order as they were supplied when the
        CustomDist was created. If ``None``, a default  ``support_point`` function will be
        assigned that will always return 0, or an array of zeros.
    ndim_supp : Optional[int]
        The number of dimensions in the support of the distribution.
        Inferred from signature, if provided. Defaults to assuming
        a scalar distribution, i.e. ``ndim_supp = 0``
    ndims_params : Optional[Sequence[int]]
        The list of number of dimensions in the support of each of the distribution's
        parameters. Inferred from signature, if provided. Defaults to assuming
        all parameters are scalars, i.e. ``ndims_params=[0, ...]``.
    signature : Optional[str]
        A numpy vectorize-like signature that indicates the number and core dimensionality
        of the input parameters and sample outputs of the CustomDist.
        When specified, `ndim_supp` and `ndims_params` are not needed. See examples below.
    dtype : str
        The dtype of the distribution. All draws and observations passed into the
        distribution will be cast onto this dtype. This is not needed if an PyTensor
        dist function is provided, which should already return the right dtype!
    class_name : str
        Name for the class which will wrap the CustomDist methods. When not specified,
        it will be given the name of the model variable.
    kwargs :
        Extra keyword arguments are passed to the parent's class ``__new__`` method.


    Examples
    --------
    Create a CustomDist that wraps a black-box logp function. This variable cannot be
    used in prior or posterior predictive sampling because no random function was provided

    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pytensor.tensor import TensorVariable

        def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
            return -(value - mu)**2

        with pm.Model():
            mu = pm.Normal('mu',0,1)
            pm.CustomDist(
                'custom_dist',
                mu,
                logp=logp,
                observed=np.random.randn(100),
            )
            idata = pm.sample(100)

    Provide a random function that return numerical draws. This allows one to use a
    CustomDist in prior and posterior predictive sampling.
    A gufunc signature was also provided, which may be used by other routines.

    .. code-block:: python

        from typing import Optional, Tuple

        import numpy as np
        import pymc as pm
        from pytensor.tensor import TensorVariable

        def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
            return -(value - mu)**2

        def random(
            mu: np.ndarray | float,
            rng: Optional[np.random.Generator] = None,
            size : Optional[Tuple[int]]=None,
        ) -> np.ndarray | float :
            return rng.normal(loc=mu, scale=1, size=size)

        with pm.Model():
            mu = pm.Normal('mu', 0 , 1)
            pm.CustomDist(
                'custom_dist',
                mu,
                logp=logp,
                random=random,
                signature="()->()",
                observed=np.random.randn(100, 3),
                size=(100, 3),
            )
            prior = pm.sample_prior_predictive(10)

    Provide a dist function that creates a PyTensor graph built from other
    PyMC distributions. PyMC can automatically infer that the logp of this
    variable corresponds to a shifted Exponential distribution.
    A gufunc signature was also provided, which may be used by other routines.

    .. code-block:: python

        import pymc as pm
        from pytensor.tensor import TensorVariable

        def dist(
            lam: TensorVariable,
            shift: TensorVariable,
            size: TensorVariable,
        ) -> TensorVariable:
            return pm.Exponential.dist(lam, size=size) + shift

        with pm.Model() as m:
            lam = pm.HalfNormal("lam")
            shift = -1
            pm.CustomDist(
                "custom_dist",
                lam,
                shift,
                dist=dist,
                signature="(),()->()",
                observed=[-1, -1, 0],
            )

            prior = pm.sample_prior_predictive()
            posterior = pm.sample()

    Provide a dist function that creates a PyTensor graph built from other
    PyMC distributions. PyMC can automatically infer that the logp of
    this variable corresponds to a modified-PERT distribution.

    .. code-block:: python

       import pymc as pm
       from pytensor.tensor import TensorVariable

        def pert(
            low: TensorVariable,
            peak: TensorVariable,
            high: TensorVariable,
            lmbda: TensorVariable,
            size: TensorVariable,
        ) -> TensorVariable:
            range = (high - low)
            s_alpha = 1 + lmbda * (peak - low) / range
            s_beta = 1 + lmbda * (high - peak) / range
            return pm.Beta.dist(s_alpha, s_beta, size=size) * range + low

        with pm.Model() as m:
            low = pm.Normal("low", 0, 10)
            peak = pm.Normal("peak", 50, 10)
            high = pm.Normal("high", 100, 10)
            lmbda = 4
            pm.CustomDist("pert", low, peak, high, lmbda, dist=pert, observed=[30, 35, 73])

        m.point_logps()

    """

    def __new__(
        cls,
        name,
        *dist_params,
        dist: Callable | None = None,
        random: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        # TODO: Deprecate ndim_supp / ndims_params in favor of signature?
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        **kwargs,
    ):
        if isinstance(kwargs.get("observed", None), dict):
            raise TypeError(
                "Since ``v4.0.0`` the ``observed`` parameter should be of type"
                " ``pd.Series``, ``np.array``, or ``pm.Data``."
                " Previous versions allowed passing distribution parameters as"
                " a dictionary in ``observed``, in the current version these "
                "parameters are positional arguments."
            )
        dist_params = cls.parse_dist_params(dist_params)
        cls.check_valid_dist_random(dist, random, dist_params)
        moment = kwargs.pop("moment", None)
        if moment is not None:
            warnings.warn(
                "`moment` argument is deprecated. Use `support_point` instead.",
                FutureWarning,
            )
            support_point = moment
        if dist is not None:
            kwargs.setdefault("class_name", f"CustomDist_{name}")
            return _CustomSymbolicDist(
                name,
                *dist_params,
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                **kwargs,
            )
        else:
            kwargs.setdefault("class_name", f"CustomDist_{name}")
            return _CustomDist(
                name,
                *dist_params,
                random=random,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                dtype=dtype,
                **kwargs,
            )

    @classmethod
    def dist(
        cls,
        *dist_params,
        dist: Callable | None = None,
        random: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        **kwargs,
    ):
        dist_params = cls.parse_dist_params(dist_params)
        cls.check_valid_dist_random(dist, random, dist_params)
        if dist is not None:
            return _CustomSymbolicDist.dist(
                *dist_params,
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                **kwargs,
            )
        else:
            return _CustomDist.dist(
                *dist_params,
                random=random,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                dtype=dtype,
                **kwargs,
            )

    @classmethod
    def parse_dist_params(cls, dist_params):
        if len(dist_params) > 0 and callable(dist_params[0]):
            raise TypeError(
                "The DensityDist API has changed, you are using the old API "
                "where logp was the first positional argument. In the current API, "
                "the logp is a keyword argument, amongst other changes. Please refer "
                "to the API documentation for more information on how to use the "
                "new DensityDist API."
            )
        return [as_tensor_variable(param) for param in dist_params]

    @classmethod
    def check_valid_dist_random(cls, dist, random, dist_params):
        if dist is not None and random is not None:
            raise ValueError("Cannot provide both dist and random functions")
        if random is not None and cls.is_symbolic_random(random, dist_params):
            raise TypeError(
                "API change: function passed to `random` argument should no longer return a PyTensor graph. "
                "Pass such function to the `dist` argument instead."
            )

    @classmethod
    def is_symbolic_random(self, random, dist_params):
        if random is None:
            return False
        # Try calling random with symbolic inputs
        try:
            size = normalize_size_param(None)
            with new_or_existing_block_model_access(
                error_msg_on_access="Model variables cannot be created in the random function. Use the `.dist` API to create such variables."
            ):
                out = random(*dist_params, size)
        except BlockModelAccessError:
            raise
        except Exception:
            # If it fails we assume it was not
            return False
        # Confirm the output is symbolic
        return isinstance(out, Variable)


DensityDist = CustomDist


def default_not_implemented(rv_name, method_name):
    message = (
        f"Attempted to run {method_name} on the CustomDist '{rv_name}', "
        f"but this method had not been provided when the distribution was "
        f"constructed. Please re-build your model and provide a callable "
        f"to '{rv_name}'s {method_name} keyword argument.\n"
    )

    def func(*args, **kwargs):
        raise NotImplementedError(message)

    return func


def default_support_point(rv, size, *rv_inputs, rv_name=None, has_fallback=False, ndim_supp=0):
    if ndim_supp == 0:
        return pt.zeros(size, dtype=rv.dtype)
    elif has_fallback:
        return pt.zeros_like(rv)
    else:
        raise TypeError(
            "Cannot safely infer the size of a multivariate random variable's support_point. "
            f"Please provide a support_point function when instantiating the {rv_name} "
            "random variable."
        )


class DiracDeltaRV(RandomVariable):
    name = "diracdelta"
    ndim_supp = 0
    ndims_params = [0]
    _print_name = ("DiracDelta", "\\operatorname{DiracDelta}")

    def make_node(self, rng, size, dtype, c):
        c = pt.as_tensor_variable(c)
        return super().make_node(rng, size, c.dtype, c)

    @classmethod
    def rng_fn(cls, rng, c, size=None):
        if size is None:
            return c.copy()
        return np.full(size, c)


diracdelta = DiracDeltaRV()


class DiracDelta(Discrete):
    r"""
    DiracDelta log-likelihood.

    Parameters
    ----------
    c : tensor_like of float or int
        Dirac Delta parameter. The dtype of `c` determines the dtype of the distribution.
        This can affect which sampler is assigned to DiracDelta variables, or variables
        that use DiracDelta, such as Mixtures.
    """

    rv_op = diracdelta

    @classmethod
    def dist(cls, c, *args, **kwargs):
        c = pt.as_tensor_variable(c)
        if c.dtype in continuous_types:
            c = floatX(c)
        return super().dist([c], **kwargs)

    def support_point(rv, size, c):
        if not rv_size_is_none(size):
            c = pt.full(size, c)
        return c

    def logp(value, c):
        return pt.switch(
            pt.eq(value, c),
            pt.zeros_like(value),
            -np.inf,
        )

    def logcdf(value, c):
        return pt.switch(
            pt.lt(value, c),
            -np.inf,
            0,
        )


class PartialObservedRV(SymbolicRandomVariable):
    """RandomVariable with partially observed subspace, as indicated by a boolean mask.

    See `create_partial_observed_rv` for more details.
    """


def create_partial_observed_rv(
    rv: TensorVariable,
    mask: np.ndarray | TensorVariable,
) -> tuple[
    tuple[TensorVariable, TensorVariable], tuple[TensorVariable, TensorVariable], TensorVariable
]:
    """Separate observed and unobserved components of a RandomVariable.

    This function may return two independent RandomVariables or, if not possible,
    two variables from a common `PartialObservedRV` node

    Parameters
    ----------
    rv : TensorVariable
    mask : tensor_like
        Constant or variable boolean mask. True entries correspond to components of the variable that are not observed.

    Returns
    -------
    observed_rv and mask : Tuple of TensorVariable
        The observed component of the RV and respective indexing mask
    unobserved_rv and mask : Tuple of TensorVariable
        The unobserved component of the RV and respective indexing mask
    joined_rv : TensorVariable
        The symbolic join of the observed and unobserved components.
    """
    if not mask.dtype == "bool":
        raise ValueError(
            f"mask must be an array or tensor of boolean dtype, got dtype: {mask.dtype}"
        )

    if mask.ndim > rv.ndim:
        raise ValueError(f"mask can't have more dims than rv, got ndim: {mask.ndim}")

    antimask = ~mask

    can_rewrite = False
    # Only pure RVs can be rewritten
    if isinstance(rv.owner.op, RandomVariable):
        ndim_supp = rv.owner.op.ndim_supp

        # All univariate RVs can be rewritten
        if ndim_supp == 0:
            can_rewrite = True

        # Multivariate RVs can be rewritten if masking does not split within support dimensions
        else:
            batch_dims = rv.type.ndim - ndim_supp
            constant_mask = getattr(as_tensor_variable(mask), "data", None)

            # Indexing does not overlap with core dimensions
            if mask.ndim <= batch_dims:
                can_rewrite = True

            # Try to handle special case where mask is constant across support dimensions,
            # TODO: This could be done by the rewrite itself
            elif constant_mask is not None:
                # We check if a constant_mask that only keeps the first entry of each support dim
                # is equivalent to the original one after re-expanding.
                trimmed_mask = constant_mask[(...,) + (0,) * ndim_supp]
                expanded_mask = np.broadcast_to(
                    np.expand_dims(trimmed_mask, axis=tuple(range(-ndim_supp, 0))),
                    shape=constant_mask.shape,
                )
                if np.array_equal(constant_mask, expanded_mask):
                    mask = trimmed_mask
                    antimask = ~trimmed_mask
                    can_rewrite = True

    if can_rewrite:
        masked_rv = rv[mask]
        fgraph = FunctionGraph(outputs=[masked_rv], clone=False, features=[ShapeFeature()])
        [unobserved_rv] = local_subtensor_rv_lift.transform(fgraph, fgraph.outputs[0].owner)

        antimasked_rv = rv[antimask]
        fgraph = FunctionGraph(outputs=[antimasked_rv], clone=False, features=[ShapeFeature()])
        [observed_rv] = local_subtensor_rv_lift.transform(fgraph, fgraph.outputs[0].owner)

        # Make a clone of the observedRV, with a distinct rng so that observed and
        # unobserved are never treated as equivalent (and mergeable) nodes by pytensor.
        _, size, _, *inps = observed_rv.owner.inputs
        observed_rv = observed_rv.owner.op(*inps, size=size)

    # For all other cases use the more general PartialObservedRV
    else:
        # The symbolic graph simply splits the observed and unobserved components,
        # so they can be given separate values.
        dist_, mask_ = rv.type(), as_tensor_variable(mask).type()
        observed_rv_, unobserved_rv_ = dist_[~mask_], dist_[mask_]

        observed_rv, unobserved_rv = PartialObservedRV(
            inputs=[dist_, mask_],
            outputs=[observed_rv_, unobserved_rv_],
            ndim_supp=rv.owner.op.ndim_supp,
        )(rv, mask)

    [rv_shape] = constant_fold([rv.shape], raise_not_constant=False)
    joined_rv = pt.empty(rv_shape, dtype=rv.type.dtype)
    joined_rv = pt.set_subtensor(joined_rv[mask], unobserved_rv)
    joined_rv = pt.set_subtensor(joined_rv[antimask], observed_rv)

    return (observed_rv, antimask), (unobserved_rv, mask), joined_rv


@_logprob.register(PartialObservedRV)
def partial_observed_rv_logprob(op, values, dist, mask, **kwargs):
    # For the logp, simply join the values
    [obs_value, unobs_value] = values
    antimask = ~mask
    # We don't need it to be completely folded, just to avoid any RVs in the graph of the shape
    [folded_shape] = constant_fold([dist.shape], raise_not_constant=False)
    joined_value = pt.empty(folded_shape)
    joined_value = pt.set_subtensor(joined_value[mask], unobs_value)
    joined_value = pt.set_subtensor(joined_value[antimask], obs_value)
    joined_logp = logp(dist, joined_value)

    # If we have a univariate RV we can split apart the logp terms
    if op.ndim_supp == 0:
        return joined_logp[antimask], joined_logp[mask]
    # Otherwise, we can't (always/ easily) split apart logp terms.
    # We return the full logp for the observed value, and a 0-nd array for the unobserved value
    else:
        return joined_logp.ravel(), pt.zeros((0,), dtype=joined_logp.type.dtype)


@_support_point.register(PartialObservedRV)
def partial_observed_rv_support_point(op, partial_obs_rv, rv, mask):
    # Unobserved output
    if partial_obs_rv.owner.outputs.index(partial_obs_rv) == 1:
        return support_point(rv)[mask]
    # Observed output
    else:
        return support_point(rv)[~mask]
