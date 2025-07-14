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

from pytensor.graph.basic import Variable
from pytensor.tensor.elemwise import DimShuffle
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc import modelcontext
from pymc.dims.model import with_dims
from pymc.distributions import transforms
from pymc.distributions.distribution import _support_point, support_point
from pymc.distributions.shape_utils import DimsWithEllipsis, convert_dims_with_ellipsis
from pymc.logprob.transforms import Transform
from pymc.util import UNSET


@_support_point.register(DimShuffle)
def dimshuffle_support_point(ds_op, _, rv):
    # We implement support point for DimShuffle because
    # DimDistribution can register a transposed version of a variable.

    return ds_op(support_point(rv))


class DimDistribution:
    """Base class for PyMC distribution that wrap pytensor.xtensor.random operations, and follow xarray-like semantics."""

    xrv_op: Callable
    default_transform: Transform | None = None

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
                    "To avoid subtle bugs, PyMC does not make any assumptions about the dims of parameters.\n"
                    "Use `as_xtensor` with the `dims` keyword argument to specify the dims explicitly."
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
            observed = observed.transpose(*rv_dims).values

        rv = model.register_rv(
            rv.values,
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

    default_transform = transforms.log


class UnitDimDistribution(DimDistribution):
    """Base class for unit-valued distributions."""

    default_transform = transforms.logodds
