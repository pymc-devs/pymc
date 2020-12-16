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

# -*- coding: utf-8 -*-
"""
A collection of common numpy array shape operations needed for broadcasting
samples from probability distributions for stochastic nodes in PyMC.
"""
import numpy as np

__all__ = [
    "to_tuple",
    "shapes_broadcasting",
    "broadcast_dist_samples_shape",
    "get_broadcastable_dist_samples",
    "broadcast_distribution_samples",
    "broadcast_dist_samples_to",
]


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
    """Get a view of the samples drawn from distributions which adds new axises
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
