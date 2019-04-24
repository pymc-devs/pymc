import pytest
import itertools
import numpy as np
import pymc3 as pm
from pymc3.distributions.shape_utils import (
    shapes_broadcasting,
    broadcast_dist_samples_shape,
    get_broadcastable_dist_samples,
    broadcast_distribution_samples,
    broadcast_dist_samples_to,
)
from pymc3.distributions.dist_math import to_tuple

test_shapes = [
    (tuple(), (1,), (4,), (5, 4)),
    (tuple(), (1,), (7,), (5, 4)),
    (tuple(), (1,), (1, 4), (5, 4)),
    (tuple(), (1,), (10, 4), (5, 4)),
    (tuple(), (1,), (10,), (5, 4)),
    (tuple(), (1,), (1, 4), (5, 4)),
    (tuple(), (1,), (3, 4), (5, 4)),
    (tuple(), (1,), (1, 1, 4), (5, 4)),
    (tuple(), (1,), (10, 1, 4), (5, 4)),
    (tuple(), (1,), (5, 1), (5, 4)),
    (tuple(), (1,), (5, 3), (5, 4)),
    (tuple(), (1,), (10, 5, 1), (5, 4)),
    (tuple(), (1,), (10, 5, 3), (5, 4)),
    (tuple(), (1,), (10, 5, 4), (5, 4)),
    (tuple(), (1,), (5, 4), (5, 4)),
]
test_sizes = [
    None,
    tuple(),
    1,
    (1,),
    10,
    (10,),
    (1, 1),
    (10, 1),
    (7,),
    (1, 10),
    (5,),
    (5, 4),
]


class TestShapesBroadcasting:
    @pytest.mark.parametrize(
        "bad_input",
        [None, [None], "asd", 3.6, {1: 2}, {3}, [8, [8]], "3", ["3"], np.array([[2]])],
        ids=str,
    )
    def test_type_check_raises(self, bad_input):
        with pytest.raises(TypeError):
            shapes_broadcasting(bad_input, tuple(), raise_exception=True)
            shapes_broadcasting(bad_input, tuple(), raise_exception=False)

    @pytest.mark.parametrize("raise_exception", [False, True], ids=str)
    def test_type_check_success(self, raise_exception):
        inputs = [3, 3.0, tuple(), [3], (3,), np.array(3), np.array([3])]
        out = shapes_broadcasting(*inputs, raise_exception=raise_exception)
        assert out == (3,)

    @pytest.mark.parametrize(
        ["shapes", "raise_exception"],
        itertools.product(test_shapes, [False, True]),
        ids=str,
    )
    def test_broadcasting(self, shapes, raise_exception):
        try:
            expected_out = np.broadcast(*[np.empty(s) for s in shapes]).shape
        except ValueError:
            expected_out = None
        if expected_out is None:
            if raise_exception:
                with pytest.raises(ValueError):
                    shapes_broadcasting(*shapes, raise_exception=raise_exception)
            else:
                out = shapes_broadcasting(*shapes, raise_exception=raise_exception)
                assert out is None
        else:
            out = shapes_broadcasting(*shapes, raise_exception=raise_exception)
            assert out == expected_out

    @pytest.mark.parametrize(
        ["size", "shapes"], itertools.product(test_sizes, test_shapes), ids=str
    )
    def test_broadcast_dist_samples_shape(self, size, shapes):
        size_ = to_tuple(size)
        shapes_ = [s if s[:len(size_)] != size_ else s[len(size_):] for s in shapes]
        try:
            expected_out = np.broadcast(*[np.empty(s) for s in shapes_]).shape
        except ValueError:
            expected_out = None
        if expected_out is not None and any((s[: len(size_)] == size_ for s in shapes)):
            expected_out = size_ + expected_out
        if expected_out is None:
            with pytest.raises(ValueError):
                broadcast_dist_samples_shape(shapes, size=size)
        else:
            out = broadcast_dist_samples_shape(shapes, size=size)
            assert out == expected_out
