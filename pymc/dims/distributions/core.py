#   Copyright 2025 - present The PyMC Developers
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
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any

import pytensor.tensor as pt

from pytensor.graph.basic import Variable
from pytensor.tensor import TensorVariable
from pytensor.tensor import expand_dims as pt_expand_dims
from pytensor.tensor.elemwise import DimShuffle
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.basic import XTensorFromTensor, xtensor_from_tensor
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.type import XTensorVariable
from pytensor.xtensor.vectorization import XRV

from pymc import modelcontext
from pymc.dims.distributions.transforms import DimTransform, log_odds_transform, log_transform
from pymc.distributions.distribution import _support_point, support_point
from pymc.distributions.shape_utils import DimsWithEllipsis, convert_dims_with_ellipsis
from pymc.logprob.abstract import (
    _icdf,
    _logccdf,
    _logcdf,
    logprob_query,
)
from pymc.logprob.query import (
    bind_value,
    infer_support_axes,
    query_parts,
    rewrite_logprob_query,
    support_axes,
)
from pymc.util import UNSET


@_support_point.register(DimShuffle)
def dimshuffle_support_point(ds_op, _, rv):
    # We implement support point for DimShuffle because
    # DimDistribution can register a transposed version of a variable.

    return ds_op(support_point(rv))


@_support_point.register(XTensorFromTensor)
def xtensor_from_tensor_support_point(xtensor_op, _, rv):
    # We remove the xtensor_from_tensor operation, so initial_point doesn't have to do a further lowering
    return xtensor_op(support_point(rv))


def _to_tensor(op: XTensorFromTensor, value: XTensorVariable) -> TensorVariable:
    # Align dims that are shared between value and op to the right
    value_dims_set = set(value.dims)
    shared_dims = [dim for dim in op.dims if dim in value_dims_set]
    value = value.transpose(..., *shared_dims)
    # Add dummy broadcastable dimensions for dimensions present in the op but missing in the value
    n_value_unique_dims = len(value_dims_set) - len(shared_dims)
    missing_axis = [
        i for i, dim in enumerate(op.dims, start=n_value_unique_dims) if dim not in value_dims_set
    ]
    return pt_expand_dims(value.values, axis=missing_axis)


def _to_xtensor(
    op: XTensorFromTensor, value: XTensorVariable, var: TensorVariable, support_axes=()
) -> XTensorVariable:
    extra_value_dims = [dim for dim in value.dims if dim not in op.dims]
    # Dims that are unique to the value and not present in the op, are placed on the left by _align_value_dims
    all_dims = (*extra_value_dims, *op.dims)
    core_dims = {op.dims[axis] for axis in support_axes}
    var_dims = tuple(d for d in all_dims if d not in core_dims)
    return xtensor_from_tensor(var, dims=var_dims)


@_logcdf.register(XTensorFromTensor)
def measurable_xtensor_from_tensor_logcdf(op, value, rv):
    tensor_value = _to_tensor(op, value)
    rv_logcdf = _logcdf(rv.owner.op, tensor_value, *rv.owner.inputs)
    return _to_xtensor(op, value, rv_logcdf)


@_logccdf.register(XTensorFromTensor)
def measurable_xtensor_from_tensor_logccdf(op, value, rv):
    tensor_value = _to_tensor(op, value)
    rv_logcdf = _logccdf(rv.owner.op, tensor_value, *rv.owner.inputs)
    return _to_xtensor(op, value, rv_logcdf)


@_icdf.register(XTensorFromTensor)
def measurable_xtensor_from_tensor_icdf(op, value, rv):
    tensor_value = _to_tensor(op, value)
    icdf = _icdf(rv.owner.op, tensor_value, *rv.owner.inputs)
    return _to_xtensor(op, value, icdf)


def copy_docstring(regular_cls):
    # Copy docstring from regular distribution class to dims class
    def get_regular_docstring(dims_cls):
        if regular_cls and regular_cls.__doc__ and dims_cls.__doc__ is None:
            dims_cls.__doc__ = regular_cls.__doc__.replace("tensor_like", "xtensor_like")
        return dims_cls

    return get_regular_docstring


class DimDistribution:
    """Base class for PyMC distribution that wrap pytensor.xtensor.random operations, and follow xarray-like semantics."""

    xrv_op: Callable
    default_transform: DimTransform | None = None

    @staticmethod
    def _as_xtensor(x):
        try:
            return as_xtensor(x)
        except TypeError:
            raise ValueError(
                f"Variable {x} must have dims associated with it.\n"
                "To avoid subtle bugs, PyMC does not make any assumptions about the dims of parameters.\n"
                "Use `pymc.dims.as_xtensor(..., dims=...)` to specify the dims explicitly."
            )

    def __new__(
        cls,
        name: str,
        *dist_params,
        dims: DimsWithEllipsis | None = None,
        initval=None,
        observed=None,
        total_size=None,
        transform=UNSET,
        default_transform=UNSET,
        model=None,
        **kwargs,
    ):
        try:
            model = modelcontext(model)
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to instantiate distributions. "
                "Add variable inside a 'with model:' block, or use the '.dist' syntax for a standalone distribution."
            )

        if not isinstance(name, str):
            raise TypeError(f"Name needs to be a string but got: {name}")

        dims = convert_dims_with_ellipsis(dims)
        if dims is None:
            dim_lengths = {}
        else:
            try:
                dim_lengths = {dim: model.dim_lengths[dim] for dim in dims if dim is not Ellipsis}
            except KeyError:
                raise ValueError(
                    f"Not all dims {dims} are part of the model coords. "
                    f"Add them at initialization time or use `model.add_coord` before defining the distribution."
                )

        if observed is not None:
            observed = cls._as_xtensor(observed)

            # Propagate observed dims to dim_lengths
            for observed_dim in observed.type.dims:
                if observed_dim not in dim_lengths:
                    dim_lengths[observed_dim] = model.dim_lengths[observed_dim]

        rv = cls.dist(*dist_params, dim_lengths=dim_lengths, **kwargs)

        # User provided dims must specify all dims or use ellipsis
        if dims is not None:
            if (... not in dims) and (set(dims) != set(rv.type.dims)):
                raise ValueError(
                    f"Provided dims {dims} do not match the distribution's output dims {rv.type.dims}. "
                    "Use ellipsis to specify all other dimensions."
                )
            # Use provided dims to transpose the output to the desired order
            rv = rv.transpose(*dims)

        rv_dims = rv.type.dims
        if observed is None:
            if default_transform is UNSET:
                default_transform = cls.default_transform
        else:
            # Align observed dims with those of the RV
            # TODO: If this fails give a more informative error message
            observed = observed.transpose(*rv_dims)

        # Check user didn't pass regular transforms
        if transform not in (UNSET, None):
            if not isinstance(transform, DimTransform):
                raise TypeError(
                    f"Transform must be a DimTransform, form pymc.dims.transforms, but got {type(transform)}."
                )
        if default_transform not in (UNSET, None):
            if not isinstance(default_transform, DimTransform):
                raise TypeError(
                    f"default_transform must be a DimTransform, from pymc.dims.transforms, but got {type(default_transform)}."
                )

        rv = model.register_rv(
            rv,
            name=name,
            observed=observed,
            total_size=total_size,
            dims=rv_dims,
            transform=transform,
            default_transform=default_transform,
            initval=initval,
        )

        return as_xtensor(rv, dims=rv_dims)

    @classmethod
    def dist(
        cls,
        dist_params,
        *,
        dim_lengths: dict[str, Variable | int] | None = None,
        core_dims: str | Sequence[str] | None = None,
        **kwargs,
    ) -> XTensorVariable:
        for invalid_kwarg in ("size", "shape", "dims"):
            if invalid_kwarg in kwargs:
                raise TypeError(f"DimDistribution does not accept {invalid_kwarg} argument.")

        # XRV requires only extra_dims, not dims
        dist_params = [cls._as_xtensor(param) for param in dist_params]

        if dim_lengths is None:
            extra_dims = None
        else:
            # Exclude dims that are implied by the parameters or core_dims
            implied_dims = set(chain.from_iterable(param.type.dims for param in dist_params))
            if core_dims is not None:
                if isinstance(core_dims, str):
                    implied_dims.add(core_dims)
                else:
                    implied_dims.update(core_dims)

            extra_dims = {
                dim: length for dim, length in dim_lengths.items() if dim not in implied_dims
            }
        if kwargs.get("rng") is None:
            kwargs["rng"] = pt.random.shared_rng(seed=None)
        _, rv = cls.xrv_op(
            *dist_params,
            extra_dims=extra_dims,
            core_dims=core_dims,
            return_next_rng=True,
            **kwargs,
        )
        return rv


class VectorDimDistribution(DimDistribution):
    @classmethod
    def dist(self, *args, core_dims: str | Sequence[str] | None = None, **kwargs):
        # Add a helpful error message if core_dims is not provided
        if core_dims is None:
            raise ValueError(
                f"{self.__name__} requires core_dims to be specified, as it involves non-scalar inputs or outputs."
                "Check the documentation of the distribution for details."
            )
        return super().dist(*args, core_dims=core_dims, **kwargs)


class PositiveDimDistribution(DimDistribution):
    """Base class for positive continuous distributions."""

    default_transform = log_transform


class UnitDimDistribution(DimDistribution):
    """Base class for unit-valued distributions."""

    default_transform = log_odds_transform


def expand_dist_dims(dist: XTensorVariable, extra_dims: dict[str, Any]) -> XTensorVariable:
    if overlap := (set(extra_dims) & set(dist.dims)):
        raise ValueError(f"extra_dims already present in distribution: {sorted(overlap)}")

    op = None if dist.owner is None else dist.owner.op
    match op:
        case XRV():
            # Recreate dist with new extra dims
            dist_props = dist.owner.op._props_dict()
            dist_props["extra_dims"] = (*(extra_dims.keys()), *dist_props["extra_dims"])
            new_dist_op = type(dist.owner.op)(**dist_props)
            _old_rng, *params_and_dim_lengths = dist.owner.inputs
            # We don't propagate the old RNG, because we don't want the new and old dists to be correlated
            new_rng = pt.random.shared_rng(seed=None)
            return new_dist_op(new_rng, *extra_dims.values(), *params_and_dim_lengths)
        case Transpose():
            return expand_dist_dims(dist.owner.inputs[0], extra_dims=extra_dims).transpose(
                ..., *dist.dims
            )
        case _:
            raise NotImplementedError(f"expand_dist_dims not implemented for {dist} with op {op}")


@infer_support_axes.register(XTensorFromTensor)
def measure_xtensor_from_tensor(op, var):
    return support_axes(var.owner.inputs[0])


@rewrite_logprob_query.register(XTensorFromTensor)
def rewrite_xtensor_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    base = rv.owner.inputs[0]
    axes = support_axes(base)
    term = logprob_query(bind_value(fgraph, base, _to_tensor(op, value)))
    return [_to_xtensor(op, value, term, axes)]
