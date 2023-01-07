#   Copyright 2021 The PyMC Developers
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

# -*- coding: utf-8 -*-
"""
A collection of common shape operations needed for broadcasting
samples from probability distributions for stochastic nodes in PyMC.
"""
import warnings

from functools import singledispatch
from typing import Any, Optional, Sequence, Tuple, Union, cast

import numpy as np

from pytensor import config
from pytensor import tensor as at
from pytensor.graph.basic import Variable
from pytensor.graph.op import Op, compute_test_value
from pytensor.raise_op import Assert
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.var import TensorVariable
from typing_extensions import TypeAlias

from pymc.model import modelcontext
from pymc.pytensorf import convert_observed_data

__all__ = [
    "to_tuple",
    "shapes_broadcasting",
    "broadcast_dist_samples_shape",
    "get_broadcastable_dist_samples",
    "broadcast_distribution_samples",
    "broadcast_dist_samples_to",
    "rv_size_is_none",
    "change_dist_size",
]

from pymc.exceptions import ShapeError
from pymc.pytensorf import PotentialShapeType
from pymc.util import _add_future_warning_tag


def to_tuple(shape):
    """Convert ints, arrays, and Nones to tuples

    Parameters
    ----------
    shape: None, int or array-like
        Represents the shape to convert to tuple.

    Returns
    -------
    If `shape` is None, returns an empty tuple. If it's an int, (shape,) is
    returned. If it is array-like, tuple(shape) is returned.
    """
    if shape is None:
        return tuple()
    temp = np.atleast_1d(shape)
    if temp.size == 0:
        return tuple()
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


def shapes_broadcasting(*args, raise_exception=False):
    """Return the shape resulting from broadcasting multiple shapes.
    Represents numpy's broadcasting rules.

    Parameters
    ----------
    *args: array-like of int
        Tuples or arrays or lists representing the shapes of arrays to be
        broadcast.
    raise_exception: bool (optional)
        Controls whether to raise an exception or simply return `None` if
        the broadcasting fails.

    Returns
    -------
    Resulting shape. If broadcasting is not possible and `raise_exception` is
    False, then `None` is returned. If `raise_exception` is `True`, a
    `ValueError` is raised.
    """
    x = list(_check_shape_type(args[0])) if args else ()
    for arg in args[1:]:
        y = list(_check_shape_type(arg))
        if len(x) < len(y):
            x, y = y, x
        if len(y) > 0:
            x[-len(y) :] = [
                j if i == 1 else i if j == 1 else i if i == j else 0
                for i, j in zip(x[-len(y) :], y)
            ]
        if not all(x):
            if raise_exception:
                raise ValueError(
                    "Supplied shapes {} do not broadcast together".format(
                        ", ".join([f"{a}" for a in args])
                    )
                )
            else:
                return None
    return tuple(x)


def broadcast_dist_samples_shape(shapes, size=None):
    """Apply shape broadcasting to shape tuples but assuming that the shapes
    correspond to draws from random variables, with the `size` tuple possibly
    prepended to it. The `size` prepend is ignored to consider if the supplied
    `shapes` can broadcast or not. It is prepended to the resulting broadcasted
    `shapes`, if any of the shape tuples had the `size` prepend.

    Parameters
    ----------
    shapes: Iterable of tuples holding the distribution samples shapes
    size: None, int or tuple (optional)
        size of the sample set requested.

    Returns
    -------
    tuple of the resulting shape

    Examples
    --------
    .. code-block:: python

        size = 100
        shape0 = (size,)
        shape1 = (size, 5)
        shape2 = (size, 4, 5)
        out = broadcast_dist_samples_shape([shape0, shape1, shape2],
                                           size=size)
        assert out == (size, 4, 5)

    .. code-block:: python

        size = 100
        shape0 = (size,)
        shape1 = (5,)
        shape2 = (4, 5)
        out = broadcast_dist_samples_shape([shape0, shape1, shape2],
                                           size=size)
        assert out == (size, 4, 5)

    .. code-block:: python

        size = 100
        shape0 = (1,)
        shape1 = (5,)
        shape2 = (4, 5)
        out = broadcast_dist_samples_shape([shape0, shape1, shape2],
                                           size=size)
        assert out == (4, 5)
    """
    if size is None:
        broadcasted_shape = shapes_broadcasting(*shapes)
        if broadcasted_shape is None:
            raise ValueError(
                "Cannot broadcast provided shapes {} given size: {}".format(
                    ", ".join([f"{s}" for s in shapes]), size
                )
            )
        return broadcasted_shape
    shapes = [_check_shape_type(s) for s in shapes]
    _size = to_tuple(size)
    # samples shapes without the size prepend
    sp_shapes = [s[len(_size) :] if _size == s[: min([len(_size), len(s)])] else s for s in shapes]
    try:
        broadcast_shape = shapes_broadcasting(*sp_shapes, raise_exception=True)
    except ValueError:
        raise ValueError(
            "Cannot broadcast provided shapes {} given size: {}".format(
                ", ".join([f"{s}" for s in shapes]), size
            )
        )
    broadcastable_shapes = []
    for shape, sp_shape in zip(shapes, sp_shapes):
        if _size == shape[: len(_size)]:
            # If size prepends the shape, then we have to add broadcasting axis
            # in the middle
            p_shape = (
                shape[: len(_size)]
                + (1,) * (len(broadcast_shape) - len(sp_shape))
                + shape[len(_size) :]
            )
        else:
            p_shape = shape
        broadcastable_shapes.append(p_shape)
    return shapes_broadcasting(*broadcastable_shapes, raise_exception=True)


def get_broadcastable_dist_samples(
    samples, size=None, must_bcast_with=None, return_out_shape=False
):
    """Get a view of the samples drawn from distributions which adds new axes
    in between the `size` prepend and the distribution's `shape`. These views
    should be able to broadcast the samples from the distrubtions taking into
    account the `size` (i.e. the number of samples) of the draw, which is
    prepended to the sample's `shape`. Optionally, one can supply an extra
    `must_bcast_with` to try to force samples to be able to broadcast with a
    given shape. A `ValueError` is raised if it is not possible to broadcast
    the provided samples.

    Parameters
    ----------
    samples: Iterable of ndarrays holding the sampled values
    size: None, int or tuple (optional)
        size of the sample set requested.
    must_bcast_with: None, int or tuple (optional)
        Tuple shape to which the samples must be able to broadcast
    return_out_shape: bool (optional)
        If `True`, this function also returns the output's shape and not only
        samples views.

    Returns
    -------
    broadcastable_samples: List of the broadcasted sample arrays
    broadcast_shape: If `return_out_shape` is `True`, the resulting broadcast
        shape is returned.

    Examples
    --------
    .. code-block:: python

        must_bcast_with = (3, 1, 5)
        size = 100
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(size, 5)
        sample2 = np.random.randn(size, 4, 5)
        out = broadcast_dist_samples_to(
            [sample0, sample1, sample2],
            size=size,
            must_bcast_with=must_bcast_with,
        )
        assert out[0].shape == (size, 1, 1, 1)
        assert out[1].shape == (size, 1, 1, 5)
        assert out[2].shape == (size, 1, 4, 5)
        assert np.all(sample0[:, None, None, None] == out[0])
        assert np.all(sample1[:, None, None] == out[1])
        assert np.all(sample2[:, None] == out[2])

    .. code-block:: python

        size = 100
        must_bcast_with = (3, 1, 5)
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(5)
        sample2 = np.random.randn(4, 5)
        out = broadcast_dist_samples_to(
            [sample0, sample1, sample2],
            size=size,
            must_bcast_with=must_bcast_with,
        )
        assert out[0].shape == (size, 1, 1, 1)
        assert out[1].shape == (5,)
        assert out[2].shape == (4, 5)
        assert np.all(sample0[:, None, None, None] == out[0])
        assert np.all(sample1 == out[1])
        assert np.all(sample2 == out[2])
    """
    samples = [np.asarray(p) for p in samples]
    _size = to_tuple(size)
    must_bcast_with = to_tuple(must_bcast_with)
    # Raw samples shapes
    p_shapes = [p.shape for p in samples] + [_check_shape_type(must_bcast_with)]
    out_shape = broadcast_dist_samples_shape(p_shapes, size=size)
    # samples shapes without the size prepend
    sp_shapes = [
        s[len(_size) :] if _size == s[: min([len(_size), len(s)])] else s for s in p_shapes
    ]
    broadcast_shape = shapes_broadcasting(*sp_shapes, raise_exception=True)
    broadcastable_samples = []
    for param, p_shape, sp_shape in zip(samples, p_shapes, sp_shapes):
        if _size == p_shape[: min([len(_size), len(p_shape)])]:
            # If size prepends the shape, then we have to add broadcasting axis
            # in the middle
            slicer_head = [slice(None)] * len(_size)
            slicer_tail = [np.newaxis] * (len(broadcast_shape) - len(sp_shape)) + [
                slice(None)
            ] * len(sp_shape)
        else:
            # If size does not prepend the shape, then we have leave the
            # parameter as is
            slicer_head = []
            slicer_tail = [slice(None)] * len(sp_shape)
        broadcastable_samples.append(param[tuple(slicer_head + slicer_tail)])
    if return_out_shape:
        return broadcastable_samples, out_shape
    else:
        return broadcastable_samples


def broadcast_distribution_samples(samples, size=None):
    """Broadcast samples drawn from distributions taking into account the
    size (i.e. the number of samples) of the draw, which is prepended to
    the sample's shape.

    Parameters
    ----------
    samples: Iterable of ndarrays holding the sampled values
    size: None, int or tuple (optional)
        size of the sample set requested.

    Returns
    -------
    List of broadcasted sample arrays

    Examples
    --------
    .. code-block:: python

        size = 100
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(size, 5)
        sample2 = np.random.randn(size, 4, 5)
        out = broadcast_distribution_samples([sample0, sample1, sample2],
                                             size=size)
        assert all((o.shape == (size, 4, 5) for o in out))
        assert np.all(sample0[:, None, None] == out[0])
        assert np.all(sample1[:, None, :] == out[1])
        assert np.all(sample2 == out[2])

    .. code-block:: python

        size = 100
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(5)
        sample2 = np.random.randn(4, 5)
        out = broadcast_distribution_samples([sample0, sample1, sample2],
                                             size=size)
        assert all((o.shape == (size, 4, 5) for o in out))
        assert np.all(sample0[:, None, None] == out[0])
        assert np.all(sample1 == out[1])
        assert np.all(sample2 == out[2])
    """
    return np.broadcast_arrays(*get_broadcastable_dist_samples(samples, size=size))


def broadcast_dist_samples_to(to_shape, samples, size=None):
    """Broadcast samples drawn from distributions to a given shape, taking into
    account the size (i.e. the number of samples) of the draw, which is
    prepended to the sample's shape.

    Parameters
    ----------
    to_shape: Tuple shape onto which the samples must be able to broadcast
    samples: Iterable of ndarrays holding the sampled values
    size: None, int or tuple (optional)
        size of the sample set requested.

    Returns
    -------
    List of the broadcasted sample arrays

    Examples
    --------
    .. code-block:: python

        to_shape = (3, 1, 5)
        size = 100
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(size, 5)
        sample2 = np.random.randn(size, 4, 5)
        out = broadcast_dist_samples_to(
            to_shape,
            [sample0, sample1, sample2],
            size=size
        )
        assert np.all((o.shape == (size, 3, 4, 5) for o in out))
        assert np.all(sample0[:, None, None, None] == out[0])
        assert np.all(sample1[:, None, None] == out[1])
        assert np.all(sample2[:, None] == out[2])

    .. code-block:: python

        size = 100
        to_shape = (3, 1, 5)
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(5)
        sample2 = np.random.randn(4, 5)
        out = broadcast_dist_samples_to(
            to_shape,
            [sample0, sample1, sample2],
            size=size
        )
        assert np.all((o.shape == (size, 3, 4, 5) for o in out))
        assert np.all(sample0[:, None, None, None] == out[0])
        assert np.all(sample1 == out[1])
        assert np.all(sample2 == out[2])
    """
    samples, to_shape = get_broadcastable_dist_samples(
        samples, size=size, must_bcast_with=to_shape, return_out_shape=True
    )
    return [np.broadcast_to(o, to_shape) for o in samples]


# User-provided can be lazily specified as scalars
Shape: TypeAlias = Union[int, TensorVariable, Sequence[Union[int, Variable]]]
Dims: TypeAlias = Union[str, Sequence[Optional[str]]]
Size: TypeAlias = Union[int, TensorVariable, Sequence[Union[int, Variable]]]

# After conversion to vectors
StrongShape: TypeAlias = Union[TensorVariable, Tuple[Union[int, Variable], ...]]
StrongDims: TypeAlias = Sequence[Optional[str]]
StrongSize: TypeAlias = Union[TensorVariable, Tuple[Union[int, Variable], ...]]


def convert_dims(dims: Optional[Dims]) -> Optional[StrongDims]:
    """Process a user-provided dims variable into None or a valid dims tuple."""
    if dims is None:
        return None

    if isinstance(dims, str):
        dims = (dims,)
    elif isinstance(dims, (list, tuple)):
        dims = tuple(dims)
    else:
        raise ValueError(f"The `dims` parameter must be a tuple, str or list. Actual: {type(dims)}")

    return dims


def convert_shape(shape: Shape) -> Optional[StrongShape]:
    """Process a user-provided shape variable into None or a valid shape object."""
    if shape is None:
        return None
    elif isinstance(shape, int) or (isinstance(shape, TensorVariable) and shape.ndim == 0):
        shape = (shape,)
    elif isinstance(shape, TensorVariable) and shape.ndim == 1:
        shape = tuple(shape)
    elif isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    else:
        raise ValueError(
            f"The `shape` parameter must be a tuple, TensorVariable, int or list. Actual: {type(shape)}"
        )

    return shape


def convert_size(size: Size) -> Optional[StrongSize]:
    """Process a user-provided size variable into None or a valid size object."""
    if size is None:
        return None
    elif isinstance(size, int) or (isinstance(size, TensorVariable) and size.ndim == 0):
        size = (size,)
    elif isinstance(size, TensorVariable) and size.ndim == 1:
        size = tuple(size)
    elif isinstance(size, (list, tuple)):
        size = tuple(size)
    else:
        raise ValueError(
            f"The `size` parameter must be a tuple, TensorVariable, int or list. Actual: {type(size)}"
        )

    return size


def shape_from_dims(dims: StrongDims, model) -> StrongShape:
    """Determines shape from a `dims` tuple.

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
    shape: Optional[StrongShape],
    size: Optional[StrongSize],
    ndim_supp: int,
) -> Optional[StrongSize]:
    """Determines the size keyword argument for creating a Distribution.

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


def rv_size_is_none(size: Variable) -> bool:
    """Check whether an rv size is None (ie., at.Constant([]))"""
    return size.type.shape == (0,)  # type: ignore [attr-defined]


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
    new_size_ndim = np.ndim(new_size)  # type: ignore
    if new_size_ndim > 1:
        raise ShapeError("The `new_size` must be ≤1-dimensional.", actual=new_size_ndim)
    elif new_size_ndim == 0:
        new_size = (new_size,)  # type: ignore
    else:
        new_size = tuple(new_size)  # type: ignore

    new_dist = _change_dist_size(dist.owner.op, dist, new_size=new_size, expand=expand)
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
    old_rng, old_size, dtype, *dist_params = rv_node.inputs

    if expand:
        shape = tuple(rv_node.op._infer_shape(old_size, dist_params))
        old_size = shape[: len(shape) - rv_node.op.ndim_supp]
        new_size = tuple(new_size) + tuple(old_size)

    # Make sure the new size is a tensor. This dtype-aware conversion helps
    # to not unnecessarily pick up a `Cast` in some cases (see #4652).
    new_size = at.as_tensor(new_size, ndim=1, dtype="int64")

    new_rv = rv_node.op(*dist_params, size=new_size, dtype=dtype)

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

    # specify_shape has a wrong signature https://github.com/pytensor-devs/pytensor/issues/1164
    return at.specify_shape(new_var, new_shapes)  # type: ignore


def get_support_shape(
    support_shape: Optional[Sequence[Union[int, np.ndarray, TensorVariable]]],
    *,
    shape: Optional[Shape] = None,
    dims: Optional[Dims] = None,
    observed: Optional[Any] = None,
    support_shape_offset: Sequence[int] = None,
    ndim_supp: int = 1,
) -> Optional[TensorVariable]:
    """Extract the support shapes from shape / dims / observed information

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
    inferred_support_shape: Optional[Sequence[Union[int, np.ndarray, Variable]]] = None

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
            model.dim_lengths[dims[i]] - support_shape_offset[i]  # type: ignore
            for i in np.arange(-ndim_supp, 0)
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

    # We did not learn anything
    if inferred_support_shape is None and support_shape is None:
        return None
    # Only source of information was the originally provided support_shape
    elif inferred_support_shape is None:
        inferred_support_shape = support_shape
    # There were two sources of support_shape, make sure they are consistent
    elif support_shape is not None:
        inferred_support_shape = [
            cast(
                Variable,
                Assert(msg="support_shape does not match respective shape dimension")(
                    inferred, at.eq(inferred, explicit)
                ),
            )
            for inferred, explicit in zip(inferred_support_shape, support_shape)
        ]

    return at.stack(inferred_support_shape)


def get_support_shape_1d(
    support_shape: Optional[Union[int, np.ndarray, TensorVariable]],
    *,
    shape: Optional[Shape] = None,
    dims: Optional[Dims] = None,
    observed: Optional[Any] = None,
    support_shape_offset: int = 0,
) -> Optional[TensorVariable]:
    """Helper function for cases when you just care about one dimension."""
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
