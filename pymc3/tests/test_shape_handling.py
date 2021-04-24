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
import sys

import aesara
import numpy as np
import pytest

from aesara import tensor as at
from aesara.tensor.shape import SpecifyShape

import pymc3 as pm

from pymc3.distributions.distribution import _validate_shape_dims_size
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
            mu = pm.Deterministic("mu", at.sum(x + eps, axis=-1))
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


@pytest.mark.xfail(reason="InverseGamma was not yet refactored")
def test_sample_generate_values(fixture_model, fixture_sizes):
    model, RVs = fixture_model
    size = to_tuple(fixture_sizes)
    with model:
        prior = pm.sample_prior_predictive(samples=fixture_sizes)
        for rv in RVs:
            assert prior[rv.name].shape == size + tuple(rv.distribution.shape)


class TestShapeDimsSize:
    @pytest.mark.parametrize("support_shape", [(), (9,)])
    @pytest.mark.parametrize("input_shape", [(), (2,), (3, 5)])
    @pytest.mark.parametrize("batch_shape", [(), (6,), (7, 8)])
    def test_parametrization_combos(self, support_shape, input_shape, batch_shape):
        ndim_batch = len(batch_shape)
        ndim_inputs = len(input_shape)
        ndim_support = len(support_shape)
        ndim = ndim_batch + ndim_inputs + ndim_support

        if ndim_support == 0:
            dist = pm.Normal
            inputs = dict(mu=np.ones(input_shape))
            expected = batch_shape + input_shape
        elif ndim_support == 1:
            dist = pm.MvNormal
            mu_shape = input_shape + support_shape
            inputs = dict(mu=np.ones(mu_shape), cov=np.eye(support_shape[0]))
            expected = batch_shape + input_shape + support_shape
        else:
            raise NotImplementedError(
                f"No tests implemented for {ndim_support}-dimensional RV support."
            )

        # Without dimensionality kwargs, the RV shape depends only on its inputs and support
        assert dist.dist(**inputs).eval().shape == input_shape + support_shape

        # The `shape` includes the support dims (0 in the case of univariates).
        assert (
            dist.dist(**inputs, shape=(*batch_shape, *input_shape, *support_shape)).eval().shape
            == expected
        )

        # In contrast, `size` is without support dims
        assert dist.dist(**inputs, size=(*batch_shape, *input_shape)).eval().shape == expected

        # With Ellipsis in the last position, `shape` is independent of all parameter and support dims.
        assert dist.dist(**inputs, shape=(*batch_shape, ...)).eval().shape == expected

        # This test uses fixed-length dimensions that are specified through model coords.
        # Here those coords are created depending on the test parametrization.
        coords = {}
        support_dims = []
        input_dims = []
        batch_dims = []

        for d in support_shape:
            dname = f"support_dim_{d}"
            coords[dname] = [f"c_{i}" for i in range(d)]
            support_dims.append(dname)
        assert len(support_dims) == ndim_support

        for d in input_shape:
            dname = f"input_dim_{d}"
            coords[dname] = [f"c_{i}" for i in range(d)]
            input_dims.append(dname)
        assert len(input_dims) == ndim_inputs

        for d in batch_shape:
            dname = f"batch_dim_{d}"
            coords[dname] = [f"c_{i}" for i in range(d)]
            batch_dims.append(dname)
        assert len(batch_dims) == ndim_batch

        # The `dims` are only available with a model.
        with pm.Model(coords=coords) as pmodel:
            rv_dims_full = dist(
                "rv_dims_full", **inputs, dims=batch_dims + input_dims + support_dims
            )
            assert rv_dims_full.eval().shape == expected
            assert len(pmodel.RV_dims["rv_dims_full"]) == ndim
            assert rv_dims_full.eval().shape == expected

            rv_dims_ellipsis = dist("rv_dims_ellipsis", **inputs, dims=(*batch_dims, ...))
            assert len(pmodel.RV_dims["rv_dims_ellipsis"]) == ndim
            assert rv_dims_ellipsis.eval().shape == expected

    def test_define_dims_on_the_fly(self):
        with pm.Model() as pmodel:
            agedata = aesara.shared(np.array([10, 20, 30]))

            # Associate the "patient" dim with an implied dimension
            age = pm.Normal("age", agedata, dims=("patient",))
            assert "patient" in pmodel.dim_lengths
            assert pmodel.dim_lengths["patient"].eval() == 3

            # Use the dim to replicate a new RV
            effect = pm.Normal("effect", 0, dims=("patient",))
            assert effect.ndim == 1
            assert effect.eval().shape == (3,)

            # Now change the length of the implied dimension
            agedata.set_value([1, 2, 3, 4])
            # The change should propagate all the way through
            assert effect.eval().shape == (4,)

    @pytest.mark.xfail(reason="Simultaneous use of size and dims is not implemented")
    def test_data_defined_size_dimension_can_register_dimname(self):
        with pm.Model() as pmodel:
            x = pm.Data("x", [[1, 2, 3, 4]], dims=("first", "second"))
            assert "first" in pmodel.dim_lengths
            assert "second" in pmodel.dim_lengths
            # two dimensions are implied; a "third" dimension is created
            y = pm.Normal("y", mu=x, shape=(2, ...), dims=("third", "first", "second"))
            assert "third" in pmodel.dim_lengths
            assert y.eval().shape() == (2, 1, 4)

    def test_can_resize_data_defined_size(self):
        with pm.Model() as pmodel:
            x = pm.Data("x", [[1, 2, 3, 4]], dims=("first", "second"))
            y = pm.Normal("y", mu=0, dims=("first", "second"))
            z = pm.Normal("z", mu=y, observed=np.ones((1, 4)))
            assert x.eval().shape == (1, 4)
            assert y.eval().shape == (1, 4)
            assert z.eval().shape == (1, 4)
            assert "first" in pmodel.dim_lengths
            assert "second" in pmodel.dim_lengths
            pmodel.set_data("x", [[1, 2], [3, 4], [5, 6]])
            assert x.eval().shape == (3, 2)
            assert y.eval().shape == (3, 2)
            assert z.eval().shape == (3, 2)

    @pytest.mark.xfail(
        condition=sys.platform == "win32",
        reason="See https://github.com/pymc-devs/pymc3/issues/4652.",
    )
    def test_observed_with_column_vector(self):
        with pm.Model() as model:
            pm.Normal("x1", mu=0, sd=1, observed=np.random.normal(size=(3, 4)))
            model.logp()
            pm.Normal("x2", mu=0, sd=1, observed=np.random.normal(size=(3, 1)))
            model.logp()

    def test_dist_api_basics(self):
        mu = aesara.shared(np.array([1, 2, 3]))
        with pytest.raises(NotImplementedError, match="API is not yet supported"):
            pm.Normal.dist(mu=mu, dims=("town",))
        assert pm.Normal.dist(mu=mu, shape=(3,)).eval().shape == (3,)
        assert pm.Normal.dist(mu=mu, shape=(5, 3)).eval().shape == (5, 3)
        assert pm.Normal.dist(mu=mu, shape=(7, ...)).eval().shape == (7, 3)
        assert pm.Normal.dist(mu=mu, size=(4, 3)).eval().shape == (4, 3)
        assert pm.Normal.dist(mu=mu, size=(8, ...)).eval().shape == (8, 3)

    def test_tensor_shape(self):
        s1 = at.scalar(dtype="int32")
        rv = pm.Uniform.dist(1, [1, 2], shape=[s1, 2])
        f = aesara.function([s1], rv, mode=aesara.Mode("py"))
        assert f(3).shape == (3, 2)
        assert f(7).shape == (7, 2)

        # As long as its length is fixed, it can also be a vector Variable
        rv_a = pm.Normal.dist(size=(7, 3))
        assert rv_a.shape.ndim == 1
        rv_b = pm.MvNormal.dist(mu=np.ones(3), cov=np.eye(3), shape=rv_a.shape)
        assert rv_b.eval().shape == (7, 3)

    def test_tensor_size(self):
        s1 = at.scalar(dtype="int32")
        s2 = at.scalar(dtype="int32")
        rv = pm.Uniform.dist(1, [1, 2], size=[s1, s2, ...])
        f = aesara.function([s1, s2], rv, mode=aesara.Mode("py"))
        assert f(3, 5).shape == (3, 5, 2)
        assert f(7, 3).shape == (7, 3, 2)

        # As long as its length is fixed, it can also be a vector Variable
        rv_a = pm.Normal.dist(size=(7, 4))
        assert rv_a.shape.ndim == 1
        rv_b = pm.MvNormal.dist(mu=np.ones(3), cov=np.eye(3), size=rv_a.shape)
        assert rv_b.eval().shape == (7, 4, 3)

    def test_auto_assert_shape(self):
        with pytest.raises(AssertionError, match="will never match"):
            pm.Normal.dist(mu=[1, 2], shape=[])

        mu = at.vector()
        rv = pm.Normal.dist(mu=mu, shape=[3, 4])
        f = aesara.function([mu], rv, mode=aesara.Mode("py"))
        assert f([1, 2, 3, 4]).shape == (3, 4)

        with pytest.raises(AssertionError, match=r"Got shape \(3, 2\), expected \(3, 4\)."):
            f([1, 2])

        # The `shape` can be symbolic!
        # This example has a batch dimension too, so under the hood it
        # becomes a symbolic input to `specify_shape` AND a symbolic `batch_shape`.
        s1 = at.scalar(dtype="int32")
        s2 = at.scalar(dtype="int32")
        rv = pm.Uniform.dist(2, [4, 5], shape=[s1, s2])
        assert isinstance(rv.owner.op, SpecifyShape)
        f = aesara.function([s1, s2], rv, mode=aesara.Mode("py"))
        assert f(3, 2).shape == (3, 2)
        assert f(7, 2).shape == (7, 2)
        with pytest.raises(
            AssertionError,
            match=r"Got shape \(3, 2\), expected \(3, 4\).",
        ):
            f(3, 4)
        pass

    def test_lazy_flavors(self):

        _validate_shape_dims_size(shape=5)
        _validate_shape_dims_size(dims="town")
        _validate_shape_dims_size(size=7)

        assert pm.Uniform.dist(2, [4, 5], size=[3, 4, 2]).eval().shape == (3, 4, 2)
        assert pm.Uniform.dist(2, [4, 5], shape=[3, 2]).eval().shape == (3, 2)
        with pm.Model(coords=dict(town=["Greifswald", "Madrid"])):
            assert pm.Normal("n2", mu=[1, 2], dims=("town",)).eval().shape == (2,)

    def test_invalid_flavors(self):
        # redundant parametrizations
        with pytest.raises(ValueError, match="Passing both"):
            _validate_shape_dims_size(shape=(2,), dims=("town",))
        with pytest.raises(ValueError, match="Passing both"):
            _validate_shape_dims_size(dims=("town",), size=(2,))
        with pytest.raises(ValueError, match="Passing both"):
            _validate_shape_dims_size(shape=(3,), size=(3,))

        # invalid, but not necessarly rare
        with pytest.raises(ValueError, match="must be an int, list or tuple"):
            _validate_shape_dims_size(size="notasize")

        # invalid ellipsis positions
        with pytest.raises(ValueError, match="may only appear in the last position"):
            _validate_shape_dims_size(shape=(3, ..., 2))
        with pytest.raises(ValueError, match="may only appear in the last position"):
            _validate_shape_dims_size(dims=(..., "town"))
        with pytest.raises(ValueError, match="may only appear in the last position"):
            _validate_shape_dims_size(size=(3, ..., ...))
