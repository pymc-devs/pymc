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

import numpy as np
import pytest

from theano import tensor as tt

import pymc3 as pm

from pymc3.distributions.shape_utils import (
    broadcast_dist_samples_shape,
    broadcast_dist_samples_to,
    broadcast_distribution_samples,
    get_broadcastable_dist_samples,
    shapes_broadcasting,
    to_tuple,
)

test_shapes = [
    (tuple(), (1,), (4,), (5, 4)),
    (tuple(), (1,), (7,), (5, 4)),
    (tuple(), (1,), (1, 4), (5, 4)),
    (tuple(), (1,), (5, 1), (5, 4)),
    (tuple(), (1,), (3, 4), (5, 4)),
    (tuple(), (1,), (5, 3), (5, 4)),
    (tuple(), (1,), (10, 4), (5, 4)),
    (tuple(), (1,), (10,), (5, 4)),
    (tuple(), (1,), (1, 1, 4), (5, 4)),
    (tuple(), (1,), (10, 1, 4), (5, 4)),
    (tuple(), (1,), (10, 5, 4), (5, 4)),
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
    (1, 10),
    (5,),
    (5, 4),
    (1, 1, 1, 1),
]
test_to_shapes = [None, tuple(), (10, 5, 4), (10, 1, 1, 5, 1)]


@pytest.fixture(params=test_sizes, ids=str)
def fixture_sizes(request):
    return request.param


@pytest.fixture(params=test_shapes, ids=str)
def fixture_shapes(request):
    return request.param


@pytest.fixture(params=[False, True], ids=str)
def fixture_exception_handling(request):
    return request.param


@pytest.fixture()
def samples_to_broadcast(fixture_sizes, fixture_shapes):
    samples = [np.empty(s) for s in fixture_shapes]
    try:
        broadcast_shape = broadcast_dist_samples_shape(fixture_shapes, size=fixture_sizes)
    except ValueError:
        broadcast_shape = None
    return fixture_sizes, samples, broadcast_shape


@pytest.fixture(params=test_to_shapes, ids=str)
def samples_to_broadcast_to(request, samples_to_broadcast):
    to_shape = request.param
    size, samples, broadcast_shape = samples_to_broadcast
    if broadcast_shape is not None:
        try:
            broadcast_shape = broadcast_dist_samples_shape(
                [broadcast_shape, to_tuple(to_shape)], size=size
            )
        except ValueError:
            broadcast_shape = None
    return to_shape, size, samples, broadcast_shape


@pytest.fixture
def fixture_model():
    with pm.Model() as model:
        n = 5
        dim = 4
        with pm.Model():
            cov = pm.InverseGamma("cov", alpha=1, beta=1)
            x = pm.Normal("x", mu=np.ones((dim,)), sigma=pm.math.sqrt(cov), shape=(n, dim))
            eps = pm.HalfNormal("eps", np.ones((n, 1)), shape=(n, dim))
            mu = pm.Deterministic("mu", tt.sum(x + eps, axis=-1))
            y = pm.Normal("y", mu=mu, sigma=1, shape=(n,))
    return model, [cov, x, eps, y]


class TestShapesBroadcasting:
    @pytest.mark.parametrize(
        "bad_input",
        [None, [None], "asd", 3.6, {1: 2}, {3}, [8, [8]], "3", ["3"], np.array([[2]])],
        ids=str,
    )
    def test_type_check_raises(self, bad_input):
        with pytest.raises(TypeError):
            shapes_broadcasting(bad_input, tuple(), raise_exception=True)
        with pytest.raises(TypeError):
            shapes_broadcasting(bad_input, tuple(), raise_exception=False)

    def test_type_check_success(self):
        inputs = [3, 3.0, tuple(), [3], (3,), np.array(3), np.array([3])]
        out = shapes_broadcasting(*inputs)
        assert out == (3,)

    def test_broadcasting(self, fixture_shapes, fixture_exception_handling):
        shapes = fixture_shapes
        raise_exception = fixture_exception_handling
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

    def test_broadcast_dist_samples_shape(self, fixture_sizes, fixture_shapes):
        size = fixture_sizes
        shapes = fixture_shapes
        size_ = to_tuple(size)
        shapes_ = [
            s if s[: min([len(size_), len(s)])] != size_ else s[len(size_) :] for s in shapes
        ]
        try:
            expected_out = np.broadcast(*[np.empty(s) for s in shapes_]).shape
        except ValueError:
            expected_out = None
        if expected_out is not None and any(
            s[: min([len(size_), len(s)])] == size_ for s in shapes
        ):
            expected_out = size_ + expected_out
        if expected_out is None:
            with pytest.raises(ValueError):
                broadcast_dist_samples_shape(shapes, size=size)
        else:
            out = broadcast_dist_samples_shape(shapes, size=size)
            assert out == expected_out


class TestSamplesBroadcasting:
    def test_broadcast_distribution_samples(self, samples_to_broadcast):
        size, samples, broadcast_shape = samples_to_broadcast
        if broadcast_shape is not None:
            outs = broadcast_distribution_samples(samples, size=size)
            assert all(o.shape == broadcast_shape for o in outs)
        else:
            with pytest.raises(ValueError):
                broadcast_distribution_samples(samples, size=size)

    def test_get_broadcastable_dist_samples(self, samples_to_broadcast):
        size, samples, broadcast_shape = samples_to_broadcast
        if broadcast_shape is not None:
            size_ = to_tuple(size)
            outs, out_shape = get_broadcastable_dist_samples(
                samples, size=size, return_out_shape=True
            )
            assert out_shape == broadcast_shape
            for i, o in zip(samples, outs):
                ishape = i.shape
                if ishape[: min([len(size_), len(ishape)])] == size_:
                    expected_shape = (
                        size_ + (1,) * (len(broadcast_shape) - len(ishape)) + ishape[len(size_) :]
                    )
                else:
                    expected_shape = ishape
                assert o.shape == expected_shape
            assert shapes_broadcasting(*[o.shape for o in outs]) == broadcast_shape
        else:
            with pytest.raises(ValueError):
                get_broadcastable_dist_samples(samples, size=size)

    def test_broadcast_dist_samples_to(self, samples_to_broadcast_to):
        to_shape, size, samples, broadcast_shape = samples_to_broadcast_to
        if broadcast_shape is not None:
            outs = broadcast_dist_samples_to(to_shape, samples, size=size)
            assert all(o.shape == broadcast_shape for o in outs)
        else:
            with pytest.raises(ValueError):
                broadcast_dist_samples_to(to_shape, samples, size=size)


def test_sample_generate_values(fixture_model, fixture_sizes):
    model, RVs = fixture_model
    size = to_tuple(fixture_sizes)
    with model:
        prior = pm.sample_prior_predictive(samples=fixture_sizes)
        for rv in RVs:
            assert prior[rv.name].shape == size + tuple(rv.distribution.shape)
