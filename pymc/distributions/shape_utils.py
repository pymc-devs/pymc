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

"""Common shape operations to broadcast samples from probability distributions for stochastic nodes in PyMC."""

import warnings

from collections.abc import Sequence
from functools import singledispatch
from typing import Any, TypeAlias, cast

import numpy as np

from pytensor import config
from pytensor import tensor as pt
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op, compute_test_value
from pytensor.raise_op import Assert
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorVariable

from pymc.model import modelcontext
from pymc.pytensorf import convert_observed_data

__all__ = [
    "to_tuple",
    "rv_size_is_none",
    "change_dist_size",
]

from pymc.exceptions import ShapeError
from pymc.pytensorf import PotentialShapeType
from pymc.util import _add_future_warning_tag


def to_tuple(shape):
    """Convert ints, arrays, and Nones to tuples.

    Parameters
    ----------
    shape: None, int or array-like
        Represents the shape to convert to tuple.

    Returns
    -------
    shape : tuple
        If `shape` is None, returns an empty tuple. If it's an int, (shape,) is
        returned. If it is array-like, tuple(shape) is returned.
    """
    if shape is None:
        return ()
    temp = np.atleast_1d(shape)
    if temp.size == 0:
        return ()
    else:
        return tuple(temp)


def _check_shape_type(shape):
    out = []
    try:
        shape = np.atleast_1d(shape)
        for s in shape:
            if isinstance(s, np.ndarray) and s.ndim > 0:
                raise TypeError(f"Value {s} is not a valid integer")
            o = int(s)
            if o != s:
                raise TypeError(f"Value {s} is not a valid integer")
            out.append(o)
    except Exception:
        raise TypeError(f"Supplied value {shape} does not represent a valid shape")
    return tuple(out)


# User-provided can be lazily specified as scalars
Shape: TypeAlias = int | TensorVariable | Sequence[int | Variable]
Dims: TypeAlias = str | Sequence[str | None]
Size: TypeAlias = int | TensorVariable | Sequence[int | Variable]

# After conversion to vectors
StrongShape: TypeAlias = TensorVariable | tuple[int | Variable, ...]
StrongDims: TypeAlias = Sequence[str | None]
StrongSize: TypeAlias = TensorVariable | tuple[int | Variable, ...]


def convert_dims(dims: Dims | None) -> StrongDims | None:
    """Process a user-provided dims variable into None or a valid dims tuple."""
    if dims is None:
        return None

    if isinstance(dims, str):
        dims = (dims,)
    elif isinstance(dims, list | tuple):
        dims = tuple(dims)
    else:
        raise ValueError(f"The `dims` parameter must be a tuple, str or list. Actual: {type(dims)}")

    return dims


def convert_shape(shape: Shape) -> StrongShape | None:
    """Process a user-provided shape variable into None or a valid shape object."""
    if shape is None or (isinstance(shape, Variable) and isinstance(shape.type, NoneTypeT)):
        return None
    elif isinstance(shape, int) or (isinstance(shape, TensorVariable) and shape.ndim == 0):
        shape = (shape,)
    elif isinstance(shape, TensorVariable) and shape.ndim == 1:
        shape = tuple(shape)
    elif isinstance(shape, list | tuple):
        shape = tuple(shape)
    else:
        raise ValueError(
            f"The `shape` parameter must be a tuple, TensorVariable, int or list. Actual: {type(shape)}"
        )

    return shape


def convert_size(size: Size) -> StrongSize | None:
    """Process a user-provided size variable into None or a valid size object."""
    if size is None or (isinstance(size, Variable) and isinstance(size.type, NoneTypeT)):
        return None
    elif isinstance(size, int) or (isinstance(size, TensorVariable) and size.ndim == 0):
        return (size,)
    elif isinstance(size, TensorVariable) and size.ndim == 1:
        return tuple(size)
    elif isinstance(size, list | tuple):
        return tuple(size)
    else:
        raise ValueError(
            f"The `size` parameter must be a tuple, TensorVariable, int or list. Actual: {type(size)}"
        )


def shape_from_dims(dims: StrongDims, model) -> StrongShape:
    """Determine shape from a `dims` tuple.

    Parameters
    ----------
    dims : array-like
        A vector of dimension names or None.
    model : pm.Model
        The current model on stack.

    Returns
    -------
    dims : tuple of (str or None)
        Names or None for all RV dimensions.
    """
    # Dims must be known already
    unknowndim_dims = set(dims) - set(model.dim_lengths)
    if unknowndim_dims:
        raise KeyError(
            f"Dimensions {unknowndim_dims} are unknown to the model and cannot be used to specify a `shape`."
        )

    return tuple(model.dim_lengths[dname] for dname in dims)


def find_size(
    shape: StrongShape | None,
    size: StrongSize | None,
    ndim_supp: int,
) -> StrongSize | None:
    """Determine the size keyword argument for creating a Distribution.

    Parameters
    ----------
    shape
        A tuple specifying the final shape of a distribution
    size
        A tuple specifying the size of a distribution
    ndim_supp : int
        The support dimension of the distribution.
        0 if a univariate distribution, 1 or higher for multivariate distributions.

    Returns
    -------
    size : tuble of int or TensorVariable, optional
        The size argument for creating the Distribution
    """
    if size is not None:
        return size

    if shape is not None:
        ndim_expected = len(tuple(shape))
        ndim_batch = ndim_expected - ndim_supp
        return tuple(shape)[:ndim_batch]

    return None


def rv_size_is_none(size: TensorVariable | Constant | None) -> bool:
    """Check whether an rv size is None (i.e., NoneConst)."""
    if size is None:
        return True
    return isinstance(size.type, NoneTypeT)


@singledispatch
def _change_dist_size(op: Op, dist: TensorVariable, new_size, expand):
    raise NotImplementedError(
        f"Variable {dist} of type {op} has no _change_dist_size implementation."
    )


def change_dist_size(
    dist: TensorVariable,
    new_size: PotentialShapeType,
    expand: bool = False,
) -> TensorVariable:
    """Change or expand the size of a Distribution.

    Parameters
    ----------
    dist:
        The old distribution to be resized.
    new_size:
        The new size of the distribution.
    expand: bool, optional
        If True, `new_size` is prepended to the existing distribution `size`, so that
        the final size is equal to (*new_size, *dist.size). Defaults to false.

    Returns
    -------
    dist : TensorVariable
        A new distribution variable that is equivalent to the original distribution with
        the new size. The new distribution will not reuse the old RandomState/Generator
        input, so it will be independent from the original distribution.

    Examples
    --------
    .. code-block:: python

        x = Normal.dist(shape=(2, 3))
        new_x = change_dist_size(x, new_size=(5, 3), expand=False)
        assert new_x.eval().shape == (5, 3)

        new_x = change_dist_size(x, new_size=(5, 3), expand=True)
        assert new_x.eval().shape == (5, 3, 2, 3)

    """
    # Check the dimensionality of the `new_size` kwarg
    new_size_ndim = np.ndim(new_size)  # type: ignore[arg-type]
    if new_size_ndim > 1:
        raise ShapeError("The `new_size` must be ≤1-dimensional.", actual=new_size_ndim)
    elif new_size_ndim == 0:
        new_size = (new_size,)  # type: ignore[assignment]
    else:
        new_size = tuple(new_size)  # type: ignore[arg-type]

    op = dist.owner.op
    new_dist = _change_dist_size(op, dist, new_size=new_size, expand=expand)
    _add_future_warning_tag(new_dist)

    new_dist.name = dist.name
    for k, v in dist.tag.__dict__.items():
        new_dist.tag.__dict__.setdefault(k, v)

    if config.compute_test_value != "off":
        compute_test_value(new_dist)

    return new_dist


@_change_dist_size.register(RandomVariable)
def change_rv_size(op, rv, new_size, expand) -> TensorVariable:
    # Extract the RV node that is to be resized
    rv_node = rv.owner
    old_rng, old_size, *dist_params = rv_node.inputs

    if expand:
        shape = tuple(rv_node.op._infer_shape(old_size, dist_params))
        old_size = shape[: len(shape) - rv_node.op.ndim_supp]
        new_size = tuple(new_size) + tuple(old_size)

    # Make sure the new size is a tensor. This dtype-aware conversion helps
    # to not unnecessarily pick up a `Cast` in some cases (see #4652).
    new_size = pt.as_tensor(new_size, ndim=1, dtype="int64")

    new_rv = rv_node.op(*dist_params, size=new_size, dtype=rv.type.dtype)

    # Replicate "traditional" rng default_update, if that was set for old_rng
    default_update = getattr(old_rng, "default_update", None)
    if default_update is not None:
        if default_update is rv_node.outputs[0]:
            new_rv.owner.inputs[0].default_update = new_rv.owner.outputs[0]
        else:
            warnings.warn(
                f"Update expression of {rv} RNG could not be replicated in resized variable",
                UserWarning,
            )

    return new_rv


@_change_dist_size.register(SpecifyShape)
def change_specify_shape_size(op, ss, new_size, expand) -> TensorVariable:
    inner_var, *shapes = ss.owner.inputs
    new_var = _change_dist_size(inner_var.owner.op, inner_var, new_size=new_size, expand=expand)

    new_shapes = [None] * new_var.ndim
    # Old specify_shape is still valid
    if expand:
        if len(shapes) > 0:
            new_shapes[-len(shapes) :] = shapes
    # Old specify_shape is still valid for support dimensions. We do not reintroduce
    # checks for resized dimensions, although we could...
    else:
        ndim_supp = new_var.owner.op.ndim_supp
        if ndim_supp > 0:
            new_shapes[-ndim_supp:] = shapes[-ndim_supp:]

    # specify_shape has a wrong signature https://github.com/aesara-devs/aesara/issues/1164
    return pt.specify_shape(new_var, new_shapes)  # type: ignore[arg-type]


def get_support_shape(
    support_shape: Sequence[int | np.ndarray | TensorVariable] | None,
    *,
    shape: Shape | None = None,
    dims: Dims | None = None,
    observed: Any | None = None,
    support_shape_offset: Sequence[int] | None = None,
    ndim_supp: int = 1,
) -> TensorVariable | None:
    """Extract the support shapes from shape / dims / observed information.

    Parameters
    ----------
    support_shape:
        User-specified support shape for multivariate distribution
    shape:
        User-specified shape for multivariate distribution
    dims:
        User-specified dims for multivariate distribution
    observed:
        User-specified observed data from multivariate distribution
    support_shape_offset:
        Difference between last shape dimensions and the length of
        explicit support shapes in multivariate distribution, defaults to 0.
        For timeseries, this is shape[-1] = support_shape[-1] + 1
    ndim_supp:
        Number of support dimensions of the given multivariate distribution, defaults to 1

    Returns
    -------
    support_shape
        Support shape, if specified directly by user, or inferred from the last dimensions of
        shape / dims / observed. When two sources of support shape information are provided,
        a symbolic Assert is added to ensure they are consistent.
    """
    if ndim_supp < 1:
        raise NotImplementedError("ndim_supp must be bigger than 0")
    if support_shape_offset is None:
        support_shape_offset = [0] * ndim_supp
    elif isinstance(support_shape_offset, int):
        support_shape_offset = [support_shape_offset] * ndim_supp
    inferred_support_shape: Sequence[int | np.ndarray | Variable] | None = None

    if shape is not None:
        shape = to_tuple(shape)
        assert isinstance(shape, tuple)
        if len(shape) < ndim_supp:
            raise ValueError(
                f"Number of shape dimensions is too small for ndim_supp of {ndim_supp}"
            )
        inferred_support_shape = [
            shape[i] - support_shape_offset[i] for i in np.arange(-ndim_supp, 0)
        ]

    if inferred_support_shape is None and dims is not None:
        dims = convert_dims(dims)
        assert isinstance(dims, tuple)
        if len(dims) < ndim_supp:
            raise ValueError(f"Number of dims is too small for ndim_supp of {ndim_supp}")
        model = modelcontext(None)
        inferred_support_shape = [
            model.dim_lengths[dims[i]] - support_shape_offset[i] for i in np.arange(-ndim_supp, 0)
        ]

    if inferred_support_shape is None and observed is not None:
        observed = convert_observed_data(observed)
        if observed.ndim < ndim_supp:
            raise ValueError(
                f"Number of observed dimensions is too small for ndim_supp of {ndim_supp}"
            )
        inferred_support_shape = [
            observed.shape[i] - support_shape_offset[i] for i in np.arange(-ndim_supp, 0)
        ]

    if inferred_support_shape is None:
        if support_shape is not None:
            # Only source of information was the originally provided support_shape
            inferred_support_shape = support_shape
        else:
            # We did not learn anything
            return None
    elif support_shape is not None:
        # There were two sources of support_shape, make sure they are consistent
        inferred_support_shape = [
            cast(
                Variable,
                Assert(msg="support_shape does not match respective shape dimension")(
                    inferred, pt.eq(inferred, explicit)
                ),
            )
            for inferred, explicit in zip(inferred_support_shape, support_shape)
        ]

    return pt.stack(inferred_support_shape)


def get_support_shape_1d(
    support_shape: int | np.ndarray | TensorVariable | None,
    *,
    shape: Shape | None = None,
    dims: Dims | None = None,
    observed: Any | None = None,
    support_shape_offset: int = 0,
) -> TensorVariable | None:
    """
    Extract the support shapes from shape / dims / observed information.

    Helper function for cases when you just care about one dimension.
    """
    support_shape_tuple = get_support_shape(
        support_shape=(support_shape,) if support_shape is not None else None,
        shape=shape,
        dims=dims,
        observed=observed,
        support_shape_offset=(support_shape_offset,),
    )

    if support_shape_tuple is not None:
        (support_shape_,) = support_shape_tuple
        return support_shape_
    else:
        return None


def implicit_size_from_params(
    *params: TensorVariable,
    ndims_params: Sequence[int],
) -> TensorVariable:
    """Infer the size of a distribution from the batch dimenesions of its parameters."""
    batch_shapes = []
    for param, ndim in zip(params, ndims_params):
        batch_shape = list(param.shape[:-ndim] if ndim > 0 else param.shape)
        # Overwrite broadcastable dims
        for i, broadcastable in enumerate(param.type.broadcastable[: len(batch_shape)]):
            if broadcastable:
                batch_shape[i] = 1
        batch_shapes.append(batch_shape)

    return pt.as_tensor(
        pt.broadcast_shape(
            *batch_shapes,
            arrays_are_shapes=True,
        )
    )
