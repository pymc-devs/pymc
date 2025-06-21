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

from pytensor.graph import node_rewriter
from pytensor.tensor.elemwise import DimShuffle
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.basic import XTensorFromTensor, xtensor_from_tensor
from pytensor.xtensor.type import XTensorVariable

from pymc import modelcontext
from pymc.dims.model import with_dims
from pymc.dims.transforms import log_odds_transform, log_transform
from pymc.distributions.distribution import _support_point, support_point
from pymc.distributions.shape_utils import DimsWithEllipsis, convert_dims
from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.rewriting import measurable_ir_rewrites_db
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
    pass


@node_rewriter([XTensorFromTensor])
def find_measurable_xtensor_from_tensor(fgraph, node) -> list[XTensorVariable] | None:
    if isinstance(node.op, MeasurableXTensorFromTensor):
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    return [MeasurableXTensorFromTensor(dims=node.op.dims)(*node.inputs)]


@_logprob.register(MeasurableXTensorFromTensor)
def measurable_xtensor_from_tensor(op, values, rv, **kwargs):
    rv_logp = _logprob(rv.owner.op, tuple(v.values for v in values), *rv.owner.inputs, **kwargs)
    return xtensor_from_tensor(rv_logp, dims=op.dims)


measurable_ir_rewrites_db.register(
    "measurable_xtensor_from_tensor", find_measurable_xtensor_from_tensor, "basic", "xtensor"
)


class DimDistribution:
    """Base class for PyMC distribution that wrap pytensor.xtensor.random operations, and follow xarray-like semantics."""

    xrv_op: Callable
    default_transform: Callable | None = None

    @staticmethod
    def _as_xtensor(x):
        try:
            return as_xtensor(x)
        except TypeError:
            try:
                return with_dims(x)
            except ValueError:
                raise ValueError(
                    f"Variable {x} must have dims associated with it.\n"
                    "To avoid subtle bugs, PyMC does not make any assumptions about the dims of the parameters.\n"
                    "Convert parameters to an xarray.DataArray, pymc.dims.Data or pytensor.xtensor.as_xtensor with explicit dims."
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
    ) -> XTensorVariable:
        try:
            model = modelcontext(model)
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to instantiate distributions. "
                "Add variable inside a 'with model:' block, or use the '.dist' syntax for a standalone distribution."
            )

        if not isinstance(name, str):
            raise TypeError(f"Name needs to be a string but got: {name}")

        if dims is None:
            dims_dict = {}
        else:
            dims = convert_dims(dims)
            try:
                dims_dict = {dim: model.dim_lengths[dim] for dim in dims if dim is not Ellipsis}
            except KeyError:
                raise ValueError(
                    f"Not all dims {dims} are part of the model coords. "
                    f"Add them at initialization time or use `model.add_coord` before defining the distribution."
                )

        if observed is not None:
            observed = cls._as_xtensor(observed)

            # Propagate observed dims to dims_dict
            for observed_dim in observed.type.dims:
                if observed_dim not in dims_dict:
                    dims_dict[observed_dim] = model.dim_lengths[observed_dim]

        rv = cls.dist(*dist_params, dims_dict=dims_dict, **kwargs)

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
        dims_dict: dict[str, int] | None = None,
        core_dims: str | Sequence[str] | None = None,
        **kwargs,
    ) -> XTensorVariable:
        for invalid_kwarg in ("size", "shape", "dims"):
            if invalid_kwarg in kwargs:
                raise TypeError(f"DimDistribution does not accept {invalid_kwarg} argument.")

        # XRV requires only extra_dims, not dims
        dist_params = [cls._as_xtensor(param) for param in dist_params]

        if dims_dict is None:
            extra_dims = None
        else:
            parameter_implied_dims = set(
                chain.from_iterable(param.type.dims for param in dist_params)
            )
            extra_dims = {
                dim: length
                for dim, length in dims_dict.items()
                if dim not in parameter_implied_dims
            }
        return cls.xrv_op(*dist_params, extra_dims=extra_dims, core_dims=core_dims, **kwargs)


class MultivariateDimDistribution(DimDistribution):
    @classmethod
    def dist(self, *args, core_dims: str | Sequence[str] | None = None, **kwargs):
        # Add a helpful error message if core_dims is not provided
        if core_dims is None:
            raise ValueError(
                f"{self.__name__} requires core_dims to be specified, as it is a multivariate distribution."
                "Check the documentation of the distribution for details."
            )
        return super().dist(*args, core_dims=core_dims, **kwargs)


class PositiveDimDistribution(DimDistribution):
    """Base class for positive continuous distributions."""

    default_transform = log_transform


class UnitDimDistribution(DimDistribution):
    """Base class for unit-valued distributions."""

    default_transform = log_odds_transform
