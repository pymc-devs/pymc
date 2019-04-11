#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of common numpy array shape operations needed for broadcasting
samples from probability distributions for stochastic nodes in PyMC.
"""
import numpy as np
from .dist_math import to_tuple


__all__ = [
    "broadcast_shapes",
    "broadcast_distribution_samples",
    "get_broadcastable_distribution_samples",
    "broadcast_distribution_samples_shape",
]


def check_shape_type(shape):
    try:
        for s in shape:
            int(s)
    except Exception:
        raise TypeError(
            "Supplied value {} does not represent a valid shape".
            format(shape)
        )


def broadcast_shapes(*args, raise_exception=False):
    """Return the shape resulting from broadcasting multiple shapes.
    Represents numpy's broadcasting rules.
    Parameters
    ----------
    *args : array-like of int
        Tuples or arrays or lists representing the shapes of arrays to be
        broadcast.
    Returns
    -------
    Resulting shape or None if broadcasting is not possible.
    """
    x = list(np.atleast_1d(args[0])) if args else ()
    check_shape_type(x)
    for arg in args[1:]:
        y = list(np.atleast_1d(arg))
        check_shape_type(y)
        if len(x) < len(y):
            x, y = y, x
        x[-len(y):] = [j if i == 1 else i if j == 1 else i if i == j else 0
                       for i, j in zip(x[-len(y):], y)]
        if not all(x):
            if raise_exception:
                raise ValueError(
                    "Supplied shapes {} do not broadcast together".
                    format(", ".join(["{}".format(a) for a in args]))
                )
            else:
                return None
    return tuple(x)


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
    return np.broadcast_arrays(
        *get_broadcastable_distribution_samples(samples, size=size)
    )


def get_broadcastable_distribution_samples(samples, size=None):
    """Get a view of the samples drawn from distributions which adds new axises
    in between the size prepend and the distribution shape. These views should
    be able to broadcast the samples from the distrubtions, taking into account
    the size (i.e. the number of samples) of the draw, which is prepended to
    the sample's shape.

    Parameters
    ----------
    samples: Iterable of ndarrays holding the sampled values
    size: None, int or tuple (optional)
        size of the sample set requested.

    Returns
    -------
    List of broadcastable views of the sample arrays

    Examples
    --------
    .. code-block:: python
        size = 100
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(size, 5)
        sample2 = np.random.randn(size, 4, 5)
        out = get_broadcastable_distribution_samples(
            [sample0, sample1, sample2],
            size=size
        )
        assert np.all(sample0[:, None, None] == out[0])
        assert np.all(sample1[:, None, :] == out[1])
        assert np.all(sample2 == out[2])

    .. code-block:: python
        size = 100
        sample0 = np.random.randn(size)
        sample1 = np.random.randn(5)
        sample2 = np.random.randn(4, 5)
        out = get_broadcastable_distribution_samples(
            [sample0, sample1, sample2],
            size=size
        )
        assert np.all(sample0[:, None, None] == out[0])
        assert np.all(sample1 == out[1])
        assert np.all(sample2 == out[2])
    """
    samples = [np.atleast_1d(p) for p in samples]
    if size is None:
        return samples
    _size = to_tuple(size)
    # Raw samples shapes
    p_shapes = [p.shape for p in samples]
    # samples shapes without the size prepend
    sp_shapes = [s[len(_size) :] if _size == s[: len(_size)] else s for s in p_shapes]
    broadcast_shape = broadcast_shapes(*sp_shapes, raise_exception=True)
    broadcastable_samples = []
    for param, p_shape, sp_shape in zip(samples, p_shapes, sp_shapes):
        if _size == p_shape[: len(_size)]:
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
    return broadcastable_samples


def broadcast_distribution_samples_shape(shapes, size=None):
    """Get the resulting broadcasted shape for samples drawn from distributions,
    taking into account the size (i.e. the number of samples) of the draw,
    which is prepended to the sample's shape.

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
        out = broadcast_distribution_samples_shape([shape0, shape1, shape2],
                                                   size=size)
        assert out == (size, 4, 5)

    .. code-block:: python
        size = 100
        shape0 = (size,)
        shape1 = (5,)
        shape2 = (4, 5)
        out = broadcast_distribution_samples_shape([shape0, shape1, shape2],
                                                   size=size)
        assert out == (size, 4, 5)
    """
    if size is None:
        broadcasted_shape = broadcast_shapes(*shapes)
        if broadcasted_shape is None:
            raise ValueError(
                "Cannot broadcast provided shapes {} given size: {}".format(
                    ", ".join(["{}".format(s) for s in shapes]), size
                )
            )
        return broadcasted_shape
    _size = to_tuple(size)
    # samples shapes without the size prepend
    sp_shapes = [s[len(_size) :] if _size == s[: len(_size)] else s for s in shapes]
    try:
        broadcast_shape = broadcast_shapes(*sp_shapes, raise_exception=True)
    except ValueError:
        raise ValueError(
            "Cannot broadcast provided shapes {} given size: {}".format(
                ", ".join(["{}".format(s) for s in shapes]), size
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
    broadcasted_shape = broadcast_shapes(*broadcastable_shapes)
    if broadcasted_shape is None:
        raise ValueError(
            "Cannot broadcast provided shapes {} given size: {}".format(
                ", ".join(["{}".format(s) for s in shapes]), size
            )
        )
    return broadcasted_shape
