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
from typing import cast

import numpy as np

from pytensor.graph import node_rewriter
from pytensor.graph.basic import Variable
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.op import RandomVariable
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.basic import XTensorFromTensor, xtensor_from_tensor
from pytensor.xtensor.type import XTensorVariable

from pymc import SymbolicRandomVariable, modelcontext
from pymc.dims.distributions.transforms import DimTransform, log_odds_transform, log_transform
from pymc.distributions.distribution import _support_point, support_point
from pymc.distributions.shape_utils import DimsWithEllipsis, convert_dims_with_ellipsis
from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.tensor import MeasurableDimShuffle
from pymc.logprob.utils import filter_measurable_variables
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


class MeasurableXTensorFromTensor(MeasurableOp, XTensorFromTensor):
    __props__ = ("dims", "core_dims")  # type: ignore[assignment]

    def __init__(self, dims, core_dims):
        super().__init__(dims=dims)
        self.core_dims = tuple(core_dims) if core_dims is not None else None


@node_rewriter([XTensorFromTensor])
def find_measurable_xtensor_from_tensor(fgraph, node) -> list[XTensorVariable] | None:
    if isinstance(node.op, MeasurableXTensorFromTensor):
        return None

    xs = filter_measurable_variables(node.inputs)

    if not xs:
        # Check if we have a transposition instead
        # The rewrite that introduces measurable tranpsoses refuses to apply to multivariate RVs
        # So we have a chance of inferring the core dims!
        [ds] = node.inputs
        ds_node = ds.owner
        if not (
            ds_node is not None
            and isinstance(ds_node.op, DimShuffle)
            and ds_node.op.is_transpose
            and filter_measurable_variables(ds_node.inputs)
        ):
            return None
        [x] = ds_node.inputs
        if not (
            x.owner is not None and isinstance(x.owner.op, RandomVariable | SymbolicRandomVariable)
        ):
            return None

        measurable_x = MeasurableDimShuffle(**ds_node.op._props_dict())(x)  # type: ignore[attr-defined]

        ndim_supp = x.owner.op.ndim_supp
        if ndim_supp:
            inverse_transpose = np.argsort(ds_node.op.shuffle)
            dims = node.op.dims
            dims_before_transpose = tuple(dims[i] for i in inverse_transpose)
            core_dims = dims_before_transpose[-ndim_supp:]
        else:
            core_dims = ()

        new_out = MeasurableXTensorFromTensor(dims=node.op.dims, core_dims=core_dims)(measurable_x)
    else:
        # If this happens we know there's no measurable transpose in between and we can
        # safely infer the core_dims positionally when the inner logp is returned
        new_out = MeasurableXTensorFromTensor(dims=node.op.dims, core_dims=None)(*node.inputs)
    return [cast(XTensorVariable, new_out)]


@_logprob.register(MeasurableXTensorFromTensor)
def measurable_xtensor_from_tensor(op, values, rv, **kwargs):
    rv_logp = _logprob(rv.owner.op, tuple(v.values for v in values), *rv.owner.inputs, **kwargs)
    if op.core_dims is None:
        # The core_dims of the inner rv are on the right
        dims = op.dims[: rv_logp.ndim]
    else:
        # We inferred where the core_dims are!
        dims = [d for d in op.dims if d not in op.core_dims]
    return xtensor_from_tensor(rv_logp, dims=dims)


measurable_ir_rewrites_db.register(
    "measurable_xtensor_from_tensor", find_measurable_xtensor_from_tensor, "basic", "xtensor"
)


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
        return cls.xrv_op(*dist_params, extra_dims=extra_dims, core_dims=core_dims, **kwargs)


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
