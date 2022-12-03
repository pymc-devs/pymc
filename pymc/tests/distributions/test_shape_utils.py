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
import re
import warnings

import pytensor
import numpy as np
import pytest

from pytensor import tensor as at
from pytensor.compile.mode import Mode
from pytensor.graph import Constant, ancestors
from pytensor.tensor import TensorVariable
from pytensor.tensor.random import normal
from pytensor.tensor.shape import SpecifyShape

import pymc as pm

from pymc import ShapeError
from pymc.distributions.shape_utils import (
    broadcast_dist_samples_shape,
    broadcast_dist_samples_to,
    broadcast_distribution_samples,
    change_dist_size,
    convert_dims,
    convert_shape,
    convert_size,
    get_broadcastable_dist_samples,
    get_support_shape,
    get_support_shape_1d,
    rv_size_is_none,
    shapes_broadcasting,
    to_tuple,
)
from pymc.model import Model

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


class TestShapesBroadcasting:
    @pytest.mark.parametrize(
        "bad_input",
        [None, [None], "asd", 3.6, {1: 2}, {3}, [8, [8]], "3", ["3"], np.array([[2]])],
        ids=str,
    )
    def test_type_check_raises(self, bad_input):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", ".*ragged nested sequences.*", np.VisibleDeprecationWarning
            )
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
            expected_out = np.broadcast(*(np.empty(s) for s in shapes)).shape
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
            expected_out = np.broadcast(*(np.empty(s) for s in shapes_)).shape
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
            assert shapes_broadcasting(*(o.shape for o in outs)) == broadcast_shape
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


class TestSizeShapeDimsObserved:
    @pytest.mark.parametrize("param_shape", [(), (2,)])
    @pytest.mark.parametrize("batch_shape", [(), (3,)])
    @pytest.mark.parametrize(
        "parametrization",
        [
            "implicit",
            "shape",
            "dims",
            "size",
        ],
    )
    def test_param_and_batch_shape_combos(
        self, param_shape: tuple, batch_shape: tuple, parametrization: str
    ):
        coords = {}
        param_dims = []
        batch_dims = []

        # Create coordinates corresponding to the parameter shape
        for d in param_shape:
            dname = f"param_dim_{d}"
            coords[dname] = [f"c_{i}" for i in range(d)]
            param_dims.append(dname)
        assert len(param_dims) == len(param_shape)
        # Create coordinates corresponding to the batch shape
        for d in batch_shape:
            dname = f"batch_dim_{d}"
            coords[dname] = [f"c_{i}" for i in range(d)]
            batch_dims.append(dname)
        assert len(batch_dims) == len(batch_shape)

        with pm.Model(coords=coords) as pmodel:
            mu = pytensor.shared(np.random.normal(size=param_shape))

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                if parametrization == "implicit":
                    rv = pm.Normal("rv", mu=mu).shape == param_shape
                else:
                    expected_shape = batch_shape + param_shape
                    if parametrization == "shape":
                        rv = pm.Normal("rv", mu=mu, shape=batch_shape + param_shape)
                        assert rv.eval().shape == expected_shape
                    elif parametrization == "dims":
                        rv = pm.Normal("rv", mu=mu, dims=batch_dims + param_dims)
                        assert rv.eval().shape == expected_shape
                    elif parametrization == "size":
                        rv = pm.Normal("rv", mu=mu, size=batch_shape + param_shape)
                        assert rv.eval().shape == expected_shape
                    else:
                        raise NotImplementedError("Invalid test case parametrization.")

    def test_broadcast_by_dims(self):
        with pm.Model(coords={"broadcast_dim": range(3)}) as m:
            x = pm.Normal("x", mu=np.zeros((1,)), dims=("broadcast_dim",))
            assert x.eval().shape == (3,)

    def test_broadcast_by_observed(self):
        with pm.Model() as m:
            x = pm.Normal("x", mu=np.zeros((1,)), observed=np.zeros((3,)))
            assert x.eval().shape == (3,)

    def test_simultaneous_shape_and_dims(self):
        with pm.Model() as pmodel:
            x = pm.ConstantData("x", [1, 2, 3], dims="ddata")

            # The shape and dims tuples correspond to each other.
            # Note: No checks are performed that implied shape (x), shape and dims actually match.
            y = pm.Normal("y", mu=x, shape=(2, 3), dims=("dshape", "ddata"))
            assert pmodel.named_vars_to_dims["y"] == ("dshape", "ddata")

            assert "dshape" in pmodel.dim_lengths
            assert y.eval().shape == (2, 3)

    def test_simultaneous_size_and_dims(self):
        with pm.Model() as pmodel:
            x = pm.ConstantData("x", [1, 2, 3], dims="ddata")
            assert "ddata" in pmodel.dim_lengths

            # Size does not include support dims, so this test must use a dist with support dims.
            kwargs = dict(name="y", size=(2, 3), mu=at.ones((3, 4)), cov=at.eye(4))
            y = pm.MvNormal(**kwargs, dims=("dsize", "ddata", "dsupport"))
            assert pmodel.named_vars_to_dims["y"] == ("dsize", "ddata", "dsupport")

            assert "dsize" in pmodel.dim_lengths
            assert y.eval().shape == (2, 3, 4)

    def test_simultaneous_dims_and_observed(self):
        with pm.Model() as pmodel:
            x = pm.ConstantData("x", [1, 2, 3], dims="ddata")
            assert "ddata" in pmodel.dim_lengths

            # Note: No checks are performed that observed and dims actually match.
            y = pm.Normal("y", observed=[0, 0, 0], dims="ddata")
            assert pmodel.named_vars_to_dims["y"] == ("ddata",)
            assert y.eval().shape == (3,)

    def test_define_dims_on_the_fly_raises(self):
        # Check that trying to use dims that are not pre-specified fails, even if their
        # length could be inferred from the shape of the variables
        msg = "Dimensions {'patient'} are unknown to the model"
        with pm.Model() as pmodel:
            with pytest.raises(KeyError, match=msg):
                pm.Normal("x", [0, 1, 2], dims=("patient",))

            with pytest.raises(KeyError, match=msg):
                pm.Normal("x", observed=[0, 1, 2], dims=("patient",))

    def test_can_resize_data_defined_size(self):
        with pm.Model() as pmodel:
            x = pm.MutableData("x", [[1, 2, 3, 4]], dims=("first", "second"))
            y = pm.Normal("y", mu=0, dims=("first", "second"))
            z = pm.Normal("z", mu=y, observed=np.ones((1, 4)), size=y.shape)
            assert x.eval().shape == (1, 4)
            assert y.eval().shape == (1, 4)
            assert z.eval().shape == (1, 4)
            assert "first" in pmodel.dim_lengths
            assert "second" in pmodel.dim_lengths
            pmodel.set_data("x", [[1, 2], [3, 4], [5, 6]])
            assert x.eval().shape == (3, 2)
            assert y.eval().shape == (3, 2)
            assert z.eval().shape == (3, 2)

    def test_size32_doesnt_break_broadcasting(self):
        size32 = at.constant([1, 10], dtype="int32")
        rv = pm.Normal.dist(0, 1, size=size32)
        assert rv.broadcastable == (True, False)

    def test_observed_with_column_vector(self):
        """This test is related to https://github.com/pymc-devs/pytensor/issues/390 which breaks
        broadcastability of column-vector RVs. This unexpected change in type can lead to
        incompatibilities during graph rewriting for model.logp evaluation.
        """
        with pm.Model() as model:
            # The `observed` is a broadcastable column vector
            obs = [
                at.as_tensor_variable(np.ones((3, 1), dtype=pytensor.config.floatX)) for _ in range(4)
            ]
            assert all(obs_.broadcastable == (False, True) for obs_ in obs)

            # Both shapes describe broadcastable volumn vectors
            size64 = at.constant([3, 1], dtype="int64")
            # But the second shape is upcasted from an int32 vector
            cast64 = at.cast(at.constant([3, 1], dtype="int32"), dtype="int64")

            pm.Normal("size64", mu=0, sigma=1, size=size64, observed=obs[0])
            pm.Normal("shape64", mu=0, sigma=1, shape=size64, observed=obs[1])
            assert model.compile_logp()({})

            pm.Normal("size_cast64", mu=0, sigma=1, size=cast64, observed=obs[2])
            pm.Normal("shape_cast64", mu=0, sigma=1, shape=cast64, observed=obs[3])
            assert model.compile_logp()({})

    def test_dist_api_works(self):
        mu = pytensor.shared(np.array([1, 2, 3]))
        with pytest.raises(NotImplementedError, match="API is not supported"):
            pm.Normal.dist(mu=mu, dims=("town",))
        assert pm.Normal.dist(mu=mu, shape=(3,)).eval().shape == (3,)
        assert pm.Normal.dist(mu=mu, shape=(5, 3)).eval().shape == (5, 3)
        assert pm.Normal.dist(mu=mu, size=(3,)).eval().shape == (3,)
        assert pm.Normal.dist(mu=mu, size=(4, 3)).eval().shape == (4, 3)

    def test_mvnormal_shape_size_difference(self):
        # Parameters add one batch dimension (4), shape is what you'd expect.
        # Under the hood the shape(4, 3) becomes size=(4,) and the RV is initially
        # created as (4, 4, 3). The internal ndim-check then recreates it with size=None.
        rv = pm.MvNormal.dist(mu=np.ones((4, 3)), cov=np.eye(3), shape=(4, 3))
        assert rv.ndim == 2
        assert tuple(rv.shape.eval()) == (4, 3)

        # shape adds two dimensions (5, 4)
        # Under the hood the shape=(5, 4, 3) becomes size=(5, 4).
        # The RV is created as (5, 4, 3) right away.
        rv = pm.MvNormal.dist(mu=[1, 2, 3], cov=np.eye(3), shape=(5, 4, 3))
        assert rv.ndim == 3
        assert tuple(rv.shape.eval()) == (5, 4, 3)

        # parameters add 1 batch dimension (4), shape adds another (5)
        # Under the hood the shape=(5, 4, 3) becomes size=(5, 4)
        # The RV is initially created as (5, 4, 3, 4, 3) and then recreated and resized.
        rv = pm.MvNormal.dist(mu=np.ones((4, 3)), cov=np.eye(3), shape=(5, 4, 3))
        assert rv.ndim == 3
        assert tuple(rv.shape.eval()) == (5, 4, 3)

        rv = pm.MvNormal.dist(mu=[1, 2, 3], cov=np.eye(3), size=(5, 4))
        assert tuple(rv.shape.eval()) == (5, 4, 3)

        rv = pm.MvNormal.dist(mu=np.ones((5, 4, 3)), cov=np.eye(3), size=(5, 4))
        assert tuple(rv.shape.eval()) == (5, 4, 3)

    def test_convert_dims(self):
        assert convert_dims(dims="town") == ("town",)
        with pytest.raises(ValueError, match="must be a tuple, str or list"):
            convert_dims(3)

    def test_convert_shape(self):
        assert convert_shape(5) == (5,)
        with pytest.raises(ValueError, match="tuple, TensorVariable, int or list"):
            convert_shape(shape="notashape")

    def test_convert_size(self):
        assert convert_size(7) == (7,)
        with pytest.raises(ValueError, match="tuple, TensorVariable, int or list"):
            convert_size(size="notasize")

    def test_lazy_flavors(self):
        assert pm.Uniform.dist(2, [4, 5], size=[3, 2]).eval().shape == (3, 2)
        assert pm.Uniform.dist(2, [4, 5], shape=[3, 2]).eval().shape == (3, 2)
        with pm.Model(coords=dict(town=["Greifswald", "Madrid"])):
            assert pm.Normal("n1", mu=[1, 2], dims="town").eval().shape == (2,)
            assert pm.Normal("n2", mu=[1, 2], dims=["town"]).eval().shape == (2,)

    def test_invalid_flavors(self):
        with pytest.raises(ValueError, match="Passing both"):
            pm.Normal.dist(0, 1, shape=(3,), size=(3,))

    def test_size_from_dims_rng_update(self):
        """Test that when setting size from dims we update the rng properly
        See https://github.com/pymc-devs/pymc/issues/5653
        """
        with pm.Model(coords=dict(x_dim=range(2))):
            x = pm.Normal("x", dims=("x_dim",))

        fn = pm.pytensorf.compile_pymc([], x)
        # Check that both function outputs (rng and draws) come from the same Apply node
        assert fn.maker.fgraph.outputs[0].owner is fn.maker.fgraph.outputs[1].owner

        # Confirm that the rng is properly offset, otherwise the second value of the first
        # draw, would match the first value of the second draw
        assert fn()[1] != fn()[0]

    def test_size_from_observed_rng_update(self):
        """Test that when setting size from observed we update the rng properly
        See https://github.com/pymc-devs/pymc/issues/5653
        """
        with pm.Model():
            x = pm.Normal("x", observed=[0, 1])

        fn = pm.pytensorf.compile_pymc([], x)
        # Check that both function outputs (rng and draws) come from the same Apply node
        assert fn.maker.fgraph.outputs[0].owner is fn.maker.fgraph.outputs[1].owner

        # Confirm that the rng is properly offset, otherwise the second value of the first
        # draw, would match the first value of the second draw
        assert fn()[1] != fn()[0]

    def test_explicit_size_shape_none(self):
        with pm.Model() as m:
            x = pm.Normal("x", shape=None, observed=[1, 2, 3])
            y = pm.Normal("y", size=None, observed=[1, 2, 3, 4])
        assert x.shape.eval().item() == 3
        assert y.shape.eval().item() == 4


def test_rv_size_is_none():
    rv = pm.Normal.dist(0, 1, size=None)
    assert rv_size_is_none(rv.owner.inputs[1])

    rv = pm.Normal.dist(0, 1, size=())
    assert rv_size_is_none(rv.owner.inputs[1])

    rv = pm.Normal.dist(0, 1, size=1)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = pm.Bernoulli.dist(0.5)
    rv = pm.Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = pm.Normal.dist(0, 1).size
    rv = pm.Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])


def test_change_rv_size():
    loc = at.as_tensor_variable([1, 2])
    rng = pytensor.shared(np.random.default_rng())
    rv = normal(loc=loc, rng=rng)
    assert rv.ndim == 1
    assert tuple(rv.shape.eval()) == (2,)

    with pytest.raises(ShapeError, match="must be ≤1-dimensional"):
        change_dist_size(rv, new_size=[[2, 3]])
    with pytest.raises(ShapeError, match="must be ≤1-dimensional"):
        change_dist_size(rv, new_size=at.as_tensor_variable([[2, 3], [4, 5]]))

    rv_new = change_dist_size(rv, new_size=(3,), expand=True)
    assert rv_new.ndim == 2
    assert tuple(rv_new.shape.eval()) == (3, 2)

    # Make sure that the shape used to determine the expanded size doesn't
    # depend on the old `RandomVariable`.
    rv_new_ancestors = set(ancestors((rv_new,)))
    assert loc in rv_new_ancestors
    assert rv not in rv_new_ancestors

    # Check that the old rng is not reused
    assert rv_new.owner.inputs[0] is not rng

    rv_newer = change_dist_size(rv_new, new_size=(4,), expand=True)
    assert rv_newer.ndim == 3
    assert tuple(rv_newer.shape.eval()) == (4, 3, 2)

    # Make sure we avoid introducing a `Cast` by converting the new size before
    # constructing the new `RandomVariable`
    rv = normal(0, 1)
    new_size = np.array([4, 3], dtype="int32")
    rv_newer = change_dist_size(rv, new_size=new_size, expand=False)
    assert rv_newer.ndim == 2
    assert isinstance(rv_newer.owner.inputs[1], Constant)
    assert tuple(rv_newer.shape.eval()) == (4, 3)

    rv = normal(0, 1)
    new_size = at.as_tensor(np.array([4, 3], dtype="int32"))
    rv_newer = change_dist_size(rv, new_size=new_size, expand=True)
    assert rv_newer.ndim == 2
    assert tuple(rv_newer.shape.eval()) == (4, 3)

    rv = normal(0, 1)
    new_size = at.as_tensor(2, dtype="int32")
    rv_newer = change_dist_size(rv, new_size=new_size, expand=True)
    assert rv_newer.ndim == 1
    assert tuple(rv_newer.shape.eval()) == (2,)


def test_change_rv_size_default_update():
    rng = pytensor.shared(np.random.default_rng(0))
    x = normal(rng=rng)

    # Test that "traditional" default_update is translated to the new rng
    rng.default_update = x.owner.outputs[0]
    new_x = change_dist_size(x, new_size=(2,))
    new_rng = new_x.owner.inputs[0]
    assert rng.default_update is x.owner.outputs[0]
    assert new_rng.default_update is new_x.owner.outputs[0]

    # Test that "non-traditional" default_update raises UserWarning
    next_rng = pytensor.shared(np.random.default_rng(1))
    rng.default_update = next_rng
    with pytest.warns(UserWarning, match="could not be replicated in resized variable"):
        new_x = change_dist_size(x, new_size=(2,))
    new_rng = new_x.owner.inputs[0]
    assert rng.default_update is next_rng
    assert not hasattr(new_rng, "default_update")

    # Test that default_update is not set if there was none before
    del rng.default_update
    new_x = change_dist_size(x, new_size=(2,))
    new_rng = new_x.owner.inputs[0]
    assert not hasattr(new_rng, "default_update")


def test_change_specify_shape_size_univariate():
    with pytensor.config.change_flags(mode=Mode("py")):
        s1, s2 = at.iscalars("s1", "s2")
        x = at.random.normal(size=(s1, s2))
        x = at.specify_shape(x, (5, 3))
        x.eval({s1: 5, s2: 3}).shape == (5, 3)

        new_x = change_dist_size(x, (10, 5))
        # SpecifyShape is no longer present
        assert not isinstance(new_x.owner.op, SpecifyShape)
        assert new_x.eval().shape == (10, 5)

        new_x = change_dist_size(x, (10, 5), expand=True)
        # SpecifyShape is still present
        assert isinstance(new_x.owner.op, SpecifyShape)
        new_x.eval({s1: 5, s2: 3}).shape == (10, 5, 5, 3)
        with pytest.raises(AssertionError, match=re.escape("expected (None, None, 5, 3)")):
            new_x.eval({s1: 6, s2: 3})


def test_change_specify_shape_size_multivariate():
    with pytensor.config.change_flags(mode=Mode("py")):
        batch, supp = at.iscalars("batch", "supp")
        x = at.random.multivariate_normal(at.zeros(supp), at.eye(supp), size=(batch,))
        x = at.specify_shape(x, (5, 3))
        x.eval({batch: 5, supp: 3}).shape == (5, 3)

        new_x = change_dist_size(x, (10, 5))
        # SpecifyShape is still present in the support dimension
        assert isinstance(new_x.owner.op, SpecifyShape)
        assert new_x.eval({supp: 3}).shape == (10, 5, 3)
        with pytest.raises(AssertionError, match=re.escape("expected (None, None, 3)")):
            new_x.eval({supp: 4})

        new_x = change_dist_size(x, (10, 5), expand=True)
        # SpecifyShape is still present in both support and batch dimension
        assert isinstance(new_x.owner.op, SpecifyShape)
        new_x.eval({batch: 5, supp: 3}).shape == (10, 5, 5, 3)
        with pytest.raises(AssertionError, match=re.escape("expected (None, None, 5, 3)")):
            new_x.eval({batch: 6, supp: 3}).shape == (10, 5, 5, 3)


@pytest.mark.parametrize(
    "support_shape, shape, support_shape_offset, expected_support_shape, consistent",
    [
        (10, None, 0, 10, True),
        (10, None, 1, 10, True),
        (None, (10,), 0, 10, True),
        (None, (10,), 1, 9, True),
        (None, (10, 5), 0, 5, True),
        (None, None, 0, None, True),
        (10, (10,), 0, 10, True),
        (10, (11,), 1, 10, True),
        (10, (5, 5), 0, 5, False),
        (10, (5, 10), 1, 9, False),
    ],
)
@pytest.mark.parametrize("info_source", ("shape", "dims", "observed"))
def test_get_support_shape_1d(
    info_source, support_shape, shape, support_shape_offset, expected_support_shape, consistent
):
    if info_source == "shape":
        inferred_support_shape = get_support_shape_1d(
            support_shape=support_shape, shape=shape, support_shape_offset=support_shape_offset
        )

    elif info_source == "dims":
        if shape is None:
            dims = None
            coords = {}
        else:
            dims = tuple(str(i) for i, _ in enumerate(shape))
            coords = {str(i): range(shape) for i, shape in enumerate(shape)}
        with Model(coords=coords):
            inferred_support_shape = get_support_shape_1d(
                support_shape=support_shape, dims=dims, support_shape_offset=support_shape_offset
            )

    elif info_source == "observed":
        if shape is None:
            observed = None
        else:
            observed = np.zeros(shape)
        inferred_support_shape = get_support_shape_1d(
            support_shape=support_shape,
            observed=observed,
            support_shape_offset=support_shape_offset,
        )

    if not isinstance(inferred_support_shape, TensorVariable):
        assert inferred_support_shape == expected_support_shape
    else:
        if consistent:
            assert inferred_support_shape.eval() == expected_support_shape
        else:
            # check that inferred steps is still correct by ignoring the assert
            f = pytensor.function(
                [], inferred_support_shape, mode=Mode().including("local_remove_all_assert")
            )
            assert f() == expected_support_shape
            with pytest.raises(AssertionError, match="support_shape does not match"):
                inferred_support_shape.eval()


@pytest.mark.parametrize(
    "support_shape, shape, support_shape_offset, expected_support_shape, ndim_supp, consistent",
    [
        ((10, 5), None, (0,), (10, 5), 1, True),
        ((10, 5), None, (1, 1), (10, 5), 1, True),
        (None, (10, 5), (0,), 5, 1, True),
        (None, (10, 5), (1,), 4, 1, True),
        (None, (10, 5, 2), (0,), 2, 1, True),
        (None, None, None, None, 1, True),
        ((10, 5), (10, 5), None, (10, 5), 2, True),
        ((10, 5), (11, 10, 5), None, (10, 5), 2, True),
        (None, (11, 10, 5), (0, 1, 0), (11, 9, 5), 3, True),
        ((10, 5), (10, 5, 5), (0,), (5,), 1, False),
        ((10, 5), (10, 5), (1, 1), (9, 4), 2, False),
    ],
)
@pytest.mark.parametrize("info_source", ("shape", "dims", "observed"))
def test_get_support_shape(
    info_source,
    support_shape,
    shape,
    support_shape_offset,
    expected_support_shape,
    ndim_supp,
    consistent,
):
    if info_source == "shape":
        inferred_support_shape = get_support_shape(
            support_shape=support_shape,
            shape=shape,
            support_shape_offset=support_shape_offset,
            ndim_supp=ndim_supp,
        )

    elif info_source == "dims":
        if shape is None:
            dims = None
            coords = {}
        else:
            dims = tuple(str(i) for i, _ in enumerate(shape))
            coords = {str(i): range(shape) for i, shape in enumerate(shape)}
        with Model(coords=coords):
            inferred_support_shape = get_support_shape(
                support_shape=support_shape,
                dims=dims,
                support_shape_offset=support_shape_offset,
                ndim_supp=ndim_supp,
            )

    elif info_source == "observed":
        if shape is None:
            observed = None
        else:
            observed = np.zeros(shape)
        inferred_support_shape = get_support_shape(
            support_shape=support_shape,
            observed=observed,
            support_shape_offset=support_shape_offset,
            ndim_supp=ndim_supp,
        )

    if not isinstance(inferred_support_shape, TensorVariable):
        assert inferred_support_shape == expected_support_shape
    else:
        if consistent:
            assert (inferred_support_shape.eval() == expected_support_shape).all()
        else:
            # check that inferred support shape is still correct by ignoring the assert
            f = pytensor.function(
                [], inferred_support_shape, mode=Mode().including("local_remove_all_assert")
            )
            assert (f() == expected_support_shape).all()
            with pytest.raises(AssertionError, match="support_shape does not match"):
                inferred_support_shape.eval()
