#   Copyright 2024 - present The PyMC Developers
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
from typing import Any, TypeAlias

import numpy as np

from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import FunctionGraph, graph_replace, node_rewriter
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.utils import MetaType
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.random.op import RandomVariable, RNGConsumerOp
from pytensor.tensor.random.rewriting import local_subtensor_rv_lift
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
from pymc.logprob.abstract import MeasurableOp, _icdf, _logccdf, _logcdf, _logprob
from pymc.logprob.basic import logp
from pymc.logprob.rewriting import logprob_rewrites_db
from pymc.printing import str_for_dist
from pymc.pytensorf import (
    collect_default_updates_inner_fgraph,
    constant_fold,
    convert_observed_data,
    floatX,
)
from pymc.util import UNSET
from pymc.vartypes import continuous_types, string_types

__all__ = [
    "Continuous",
    "DiracDelta",
    "Discrete",
    "Distribution",
    "SymbolicRandomVariable",
]

DIST_PARAMETER_TYPES: TypeAlias = np.ndarray | int | float | TensorVariable

vectorized_ppc: contextvars.ContextVar[Callable | None] = contextvars.ContextVar(
    "vectorized_ppc", default=None
)

PLATFORM = sys.platform


class _Unpickling:
    pass


class DistributionMeta(ABCMeta):
    """
    DistributionMeta class.

    Notes
    -----
    DistributionMeta currently performs many functions, and will likely be refactored soon.
    See issue below for more details
    https://github.com/pymc-devs/pymc/issues/5308
    """

    def __new__(cls, name, bases, clsdict):
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

            size_idx: int | None = None
            params_idxs: tuple[int] | None = None
            if issubclass(rv_type, SymbolicRandomVariable):
                extended_signature = getattr(rv_type, "extended_signature", None)
                if extended_signature is not None:
                    [_, size_idx, params_idxs], _ = (
                        SymbolicRandomVariable.get_input_output_type_idxs(extended_signature)
                    )

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
                        rng, size, *dist_params = dist_params
                    elif params_idxs:
                        dist_params = [dist_params[i] for i in params_idxs]
                    [value] = values
                    return class_logp(value, *dist_params)

            class_logcdf = clsdict.get("logcdf")
            if class_logcdf:

                @_logcdf.register(rv_type)
                def logcdf(op, value, *dist_params, **kwargs):
                    if isinstance(op, RandomVariable):
                        rng, size, *dist_params = dist_params
                    elif params_idxs:
                        dist_params = [dist_params[i] for i in params_idxs]
                    return class_logcdf(value, *dist_params)

            class_logccdf = clsdict.get("logccdf")
            if class_logccdf:

                @_logccdf.register(rv_type)
                def logccdf(op, value, *dist_params, **kwargs):
                    if isinstance(op, RandomVariable):
                        rng, size, *dist_params = dist_params
                    elif params_idxs:
                        dist_params = [dist_params[i] for i in params_idxs]
                    return class_logccdf(value, *dist_params)

            class_icdf = clsdict.get("icdf")
            if class_icdf:

                @_icdf.register(rv_type)
                def icdf(op, value, *dist_params, **kwargs):
                    if isinstance(op, RandomVariable):
                        rng, size, *dist_params = dist_params
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
                        rng, size, *dist_params = dist_params
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


class _class_or_instance_property(property):
    """Allow a property to be accessed from a class or an instance.

    Priority is given to the instance.

    This is used to allow extracting information from the signature of a SymbolicRandomVariable
    which may be available early as a class attribute or only later as an instance attribute.

    Adapted from https://stackoverflow.com/a/13624858
    """

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_self if owner_self is not None else owner_cls)


class SymbolicRandomVariable(MeasurableOp, RNGConsumerOp, OpFromGraph):
    """Symbolic Random Variable.

    This is a subclasse of `OpFromGraph` which is used to encapsulate the symbolic
    random graph of complex distributions which are built on top of pure
    `RandomVariable`s.

    These graphs may vary structurally based on the inputs (e.g., their dimensionality),
    and usually require that random inputs have specific shapes for correct outputs
    (e.g., avoiding broadcasting of random inputs). Due to this, most distributions that
    return SymbolicRandomVariable create their these graphs at runtime via the
    classmethod `cls.rv_op`, taking care to clone and resize random inputs, if needed.
    """

    extended_signature: str = None
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

    @_class_or_instance_property
    def signature(cls_or_self) -> None | str:
        # Convert "expanded" signature into "vanilla" signature that has no rng and size tokens
        extended_signature = cls_or_self.extended_signature
        if extended_signature is None:
            return None

        # Remove special tokens
        special_tokens = r"|".join((r"\[rng\],?", r"\[size\],?"))
        signature = re.sub(special_tokens, "", extended_signature)
        # Remove dandling commas
        signature = re.sub(r",(?=[->])|,$", "", signature)

        return signature

    @_class_or_instance_property
    def ndims_params(cls_or_self) -> Sequence[int] | None:
        """Return number of core dimensions of the distribution's parameters."""
        signature = cls_or_self.signature
        if signature is None:
            return None
        inputs_signature, _ = _parse_gufunc_signature(signature)
        return [len(sig) for sig in inputs_signature]

    @_class_or_instance_property
    def ndim_supp(cls_or_self) -> int | None:
        """Return number of support dimensions of the RandomVariable.

        (0 for scalar, 1 for vector, ...)
        """
        signature = cls_or_self.signature
        if signature is None:
            return getattr(cls_or_self, "_ndim_supp", None)
        _, outputs_params_signature = _parse_gufunc_signature(signature)
        return max(len(out_sig) for out_sig in outputs_params_signature)

    @_class_or_instance_property
    def default_output(cls_or_self) -> int | None:
        extended_signature = cls_or_self.extended_signature
        if extended_signature is None:
            return None

        _, [_, candidate_default_output] = cls_or_self.get_input_output_type_idxs(
            extended_signature
        )

        if len(candidate_default_output) == 1:
            return candidate_default_output[0]
        else:
            return None

    @staticmethod
    def get_input_output_type_idxs(
        extended_signature: str | None,
    ) -> tuple[
        tuple[tuple[int, ...], int | None, tuple[int, ...]],
        tuple[tuple[int, ...], tuple[int, ...]],
    ]:
        """Parse extended_signature and return indexes for *[rng], [size] and parameters as well as outputs."""
        if extended_signature is None:
            raise ValueError("extended_signature must be provided")

        fake_signature = extended_signature.replace("[rng]", "(rng)").replace("[size]", "(size)")
        inputs_signature, outputs_signature = _parse_gufunc_signature(fake_signature)

        input_rng_idxs = []
        size_idx = None
        input_params_idxs = []
        for i, inp_sig in enumerate(inputs_signature):
            if inp_sig == ("size",):
                size_idx = i
            elif inp_sig == ("rng",):
                input_rng_idxs.append(i)
            else:
                input_params_idxs.append(i)

        output_rng_idxs = []
        output_params_idxs = []
        for i, out_sig in enumerate(outputs_signature):
            if out_sig == ("rng",):
                output_rng_idxs.append(i)
            else:
                output_params_idxs.append(i)

        return (
            (tuple(input_rng_idxs), size_idx, tuple(input_params_idxs)),
            (tuple(output_rng_idxs), tuple(output_params_idxs)),
        )

    def rng_params(self, node) -> tuple[Variable, ...]:
        """Extract the rng parameters from the node's inputs."""
        [rng_args_idxs, _, _], _ = self.get_input_output_type_idxs(self.extended_signature)
        return tuple(node.inputs[i] for i in rng_args_idxs)

    def size_param(self, node) -> Variable | None:
        """Extract the size parameter from the node's inputs."""
        [_, size_arg_idx, _], _ = self.get_input_output_type_idxs(self.extended_signature)
        return node.inputs[size_arg_idx] if size_arg_idx is not None else None

    def dist_params(self, node) -> tuple[Variable, ...]:
        """Extract distribution parameters from the node's inputs."""
        [_, _, param_args_idxs], _ = self.get_input_output_type_idxs(self.extended_signature)
        return tuple(node.inputs[i] for i in param_args_idxs)

    def __init__(
        self,
        *args,
        extended_signature: str | None = None,
        **kwargs,
    ):
        """Initialize a SymbolicRandomVariable class."""
        if extended_signature is not None:
            self.extended_signature = extended_signature

        if "signature" in kwargs:
            self.extended_signature = kwargs.pop("signature")
            warnings.warn(
                "SymbolicRandomVariables signature argument was renamed to extended_signature."
            )

        if "ndim_supp" in kwargs:
            # For backwards compatibility we allow passing ndim_supp without signature
            # This is the only variable that PyMC absolutely needs to work with SymbolicRandomVariables
            self._ndim_supp = kwargs.pop("ndim_supp")

        if self.ndim_supp is None:
            raise ValueError("ndim_supp or signature must be provided")

        kwargs.setdefault("inline", True)
        kwargs.setdefault("strict", True)
        # Many RVS have a size argument, even when this is `None` and is therefore unused
        kwargs.setdefault("on_unused_input", "ignore")
        if hasattr(self, "name"):
            kwargs.setdefault("name", self.name)
        super().__init__(*args, **kwargs)

    def make_node(self, *inputs):
        # If we try to build the RV with a different size type (vector -> None or None -> vector)
        # We need to rebuild the Op with new size type in the inner graph
        if self.extended_signature is not None:
            (rng_arg_idxs, size_arg_idx, param_idxs), _ = self.get_input_output_type_idxs(
                self.extended_signature
            )
            if size_arg_idx is not None and len(rng_arg_idxs) == 1:
                new_size_type = normalize_size_param(inputs[size_arg_idx]).type
                if not self.input_types[size_arg_idx].in_same_class(new_size_type):
                    params = [inputs[idx] for idx in param_idxs]
                    size = inputs[size_arg_idx]
                    rng = inputs[rng_arg_idxs[0]]
                    return self.rebuild_rv(*params, size=size, rng=rng).owner

        return super().make_node(*inputs)

    def update(self, node: Apply) -> dict[Variable, Variable]:
        """Symbolic update expression for input random state variables.

        Returns a dictionary with the symbolic expressions required for correct updating
        of random state input variables repeated function evaluations. This is used by
        `pytensorf.compile_pymc`.
        """
        return collect_default_updates_inner_fgraph(node)

    def batch_ndim(self, node: Apply) -> int:
        """Return the number of dimensions of the distribution's batch shape."""
        out_ndim = max(getattr(out.type, "ndim", 0) for out in node.outputs)
        return out_ndim - self.ndim_supp

    def rebuild_rv(self, *args, **kwargs):
        """Rebuild the RandomVariable with new inputs."""
        if not hasattr(self, "rv_op"):
            raise NotImplementedError(
                f"SymbolicRandomVariable {self} without `rv_op` method cannot be rebuilt automatically."
            )
        return self.rv_op(*args, **kwargs)


@_change_dist_size.register(SymbolicRandomVariable)
def change_symbolic_rv_size(op: SymbolicRandomVariable, rv, new_size, expand) -> TensorVariable:
    extended_signature = op.extended_signature
    if extended_signature is None:
        raise NotImplementedError(
            f"SymbolicRandomVariable {op} without signature requires custom `_change_dist_size` implementation."
        )

    size = op.size_param(rv.owner)
    if size is None:
        raise NotImplementedError(
            f"SymbolicRandomVariable {op} without [size] in extended_signature requires custom `_change_dist_size` implementation."
        )

    params = op.dist_params(rv.owner)

    if expand and not rv_size_is_none(size):
        new_size = tuple(new_size) + tuple(size)

    return op.rebuild_rv(*params, size=new_size)


class Distribution(metaclass=DistributionMeta):
    """Statistical distribution."""

    # rv_op and _type are set to None via the DistributionMeta.__new__
    # if not specified as class attributes in subclasses of Distribution.
    # rv_op can either be a class (see the Normal class) or a method
    # (see the Censored class), both callable to return a TensorVariable.
    rv_op: Any = None
    rv_type: MetaType | None = None

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
        """Add a tensor variable corresponding to a PyMC distribution to the current model.

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
        return rv_out

    @classmethod
    def dist(
        cls,
        dist_params,
        *,
        shape: Shape | None = None,
        **kwargs,
    ) -> TensorVariable:
        """Create a tensor variable corresponding to the `cls` distribution.

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

        # `ndim_supp` may be available at the class level or at the instance level
        ndim_supp = getattr(cls.rv_op, "ndim_supp", getattr(cls.rv_type, "ndim_supp", None))
        if ndim_supp is None:
            # Initialize Ops and check the ndim_supp that is now required to exist
            ndim_supp = cls.rv_op(*dist_params, **kwargs).owner.op.ndim_supp

        create_size = find_size(shape=shape, size=size, ndim_supp=ndim_supp)
        return cls.rv_op(*dist_params, size=create_size, **kwargs)


@node_rewriter([SymbolicRandomVariable])
def inline_symbolic_random_variable(fgraph, node):
    """Expand a SymbolicRV when obtaining the logp graph if `inline_logprob` is True."""
    op = node.op
    if op.inline_logprob:
        return graph_replace(
            op.inner_outputs, dict(zip(op.inner_inputs, node.inputs)), strict=False
        )


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
    """Choose a representative point/value that can be used to start optimization or MCMC sampling.

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
    """Base class for discrete distributions."""

    def __new__(cls, name, *args, **kwargs):
        if kwargs.get("transform", None):
            raise ValueError("Transformations for discrete distributions")

        return super().__new__(cls, name, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions."""


class DiracDeltaRV(SymbolicRandomVariable):
    name = "diracdelta"
    extended_signature = "[size],()->()"
    _print_name = ("DiracDelta", "\\operatorname{DiracDelta}")

    def do_constant_folding(self, fgraph: "FunctionGraph", node: Apply) -> bool:
        # Because the distribution does not have RNGs we have to prevent constant-folding
        return False

    @classmethod
    def rv_op(cls, c, *, size=None, rng=None):
        size = normalize_size_param(size)
        c = pt.as_tensor(c)

        if rv_size_is_none(size):
            out = c.copy()
        else:
            out = pt.full(size, c)

        return cls(inputs=[size, c], outputs=[out])(size, c)


class DiracDelta(Discrete):
    r"""
    DiracDelta distribution.

    Parameters
    ----------
    c : tensor_like of float or int
        Dirac Delta parameter. The dtype of `c` determines the dtype of the distribution.
        This can affect which sampler is assigned to DiracDelta variables, or variables
        that use DiracDelta, such as Mixtures.
    """

    rv_type = DiracDeltaRV
    rv_op = DiracDeltaRV.rv_op

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
        unobserved_rv = local_subtensor_rv_lift.transform(fgraph, masked_rv.owner)[masked_rv]

        antimasked_rv = rv[antimask]
        fgraph = FunctionGraph(outputs=[antimasked_rv], clone=False, features=[ShapeFeature()])
        observed_rv = local_subtensor_rv_lift.transform(fgraph, antimasked_rv.owner)[antimasked_rv]

        # Make a clone of the observedRV, with a distinct rng so that observed and
        # unobserved are never treated as equivalent (and mergeable) nodes by pytensor.
        _, size, *inps = observed_rv.owner.inputs
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
