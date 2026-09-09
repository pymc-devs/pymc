#   Copyright 2024 - present The PyMC Developers
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
import pytensor
import pytest

from pytensor.compile import SharedVariable
from pytensor.graph import Constant

import pymc as pm

from pymc import Deterministic, do
from pymc.data import Data
from pymc.distributions import HalfNormal, Normal
from pymc.exceptions import NotConstantValueError
from pymc.model import Model
from pymc.model.transform.optimization import (
    freeze_dims_and_data,
    model_to_float32,
    model_to_float64,
)
from pymc.pytensorf import constant_fold


def test_freeze_dims_and_data():
    with Model(coords={"test_dim": range(5)}) as m:
        std = Data("test_data", [1])
        x = HalfNormal("x", std, dims=("test_dim",))
        y = Normal("y", shape=x.shape[0] + 1)

    x_logp, y_logp = m.logp(sum=False)

    assert not isinstance(std, Constant)
    assert x.type.shape == (None,)
    assert y.type.shape == (None,)
    assert x_logp.type.shape == (None,)
    assert y_logp.type.shape == (None,)

    frozen_m = freeze_dims_and_data(m)
    data, x, y = frozen_m["test_data"], frozen_m["x"], frozen_m["y"]
    x_logp, y_logp = frozen_m.logp(sum=False)
    assert isinstance(data, Constant)
    assert x.type.shape == (5,)
    assert y.type.shape == (6,)
    assert x_logp.type.shape == (5,)
    assert y_logp.type.shape == (6,)

    # Test trying to update a frozen data or dim raises an informative error
    with frozen_m:
        with pytest.raises(TypeError, match="The variable `test_data` must be a `SharedVariable`"):
            frozen_m.set_data("test_data", values=[2])
        with pytest.raises(
            TypeError, match="The dim_length of `test_dim` must be a `SharedVariable`"
        ):
            frozen_m.set_dim("test_dim", new_length=6, coord_values=range(6))

    # Test we can still update original model
    with m:
        m.set_data("test_data", values=[2])
        m.set_dim("test_dim", new_length=6, coord_values=range(6))
    assert m["test_data"].get_value() == [2]
    assert m.dim_lengths["test_dim"].get_value() == 6


def test_freeze_dims_nothing_to_change():
    with Model(coords={"test_dim": range(5)}) as m:
        x = HalfNormal("x", shape=(5,))
        y = Normal("y", shape=x.shape[0] + 1)

    assert m.point_logps() == freeze_dims_and_data(m).point_logps()


def test_freeze_dims_and_data_preserves_initvals():
    # fgraph_from_model drops (and rejects) initial values; freeze_dims_and_data preserves
    # constant and strategy-string ones on the rebuilt model without touching the original.
    with Model() as m:
        Normal("u", initval=7.0)  # concrete
        Normal("p", shape=3, initval="prior")  # strategy string

    frozen_m = freeze_dims_and_data(m)
    assert m.rvs_to_initial_values[m["u"]] == 7.0  # original untouched
    np.testing.assert_allclose(frozen_m.initial_point(0)["u"], m.initial_point(0)["u"])
    np.testing.assert_allclose(frozen_m.initial_point(1)["p"], m.initial_point(1)["p"])

    # Symbolic initial values reference the original graph and cannot be transplanted.
    with Model() as m2:
        d = Data("d", [1.0, 2.0])
        Normal("s", shape=2, initval=d * 2)
    with pytest.raises(NotImplementedError, match="symbolic initial value"):
        freeze_dims_and_data(m2)


def test_freeze_dims_and_data_subset():
    with Model(coords={"dim1": range(3), "dim2": range(5)}) as m:
        data1 = Data("data1", [1, 2, 3], dims="dim1")
        data2 = Data("data2", [1, 2, 3, 4, 5], dims="dim2")
        var1 = Normal("var1", dims="dim1")
        var2 = Normal("var2", dims="dim2")
        x = data1 * var1
        y = data2 * var2
        det = Deterministic("det", x[:, None] + y[None, :])

    assert det.type.shape == (None, None)

    new_m = freeze_dims_and_data(m, dims=["dim1"], data=[])
    assert new_m["det"].type.shape == (3, None)
    assert isinstance(new_m.dim_lengths["dim1"], Constant) and new_m.dim_lengths["dim1"].data == 3
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=["dim2"], data=[])
    assert new_m["det"].type.shape == (None, 5)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], Constant) and new_m.dim_lengths["dim2"].data == 5
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=["dim1", "dim2"], data=[])
    assert new_m["det"].type.shape == (3, 5)
    assert isinstance(new_m.dim_lengths["dim1"], Constant) and new_m.dim_lengths["dim1"].data == 3
    assert isinstance(new_m.dim_lengths["dim2"], Constant) and new_m.dim_lengths["dim2"].data == 5
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=[], data=["data1"])
    assert new_m["det"].type.shape == (3, None)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], Constant) and np.all(new_m["data1"].data == [1, 2, 3])
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=[], data=["data2"])
    assert new_m["det"].type.shape == (None, 5)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], Constant) and np.all(new_m["data2"].data == [1, 2, 3, 4, 5])

    new_m = freeze_dims_and_data(m, dims=[], data=["data1", "data2"])
    assert new_m["det"].type.shape == (3, 5)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], Constant) and np.all(new_m["data1"].data == [1, 2, 3])
    assert isinstance(new_m["data2"], Constant) and np.all(new_m["data2"].data == [1, 2, 3, 4, 5])

    new_m = freeze_dims_and_data(m, dims=["dim1"], data=["data2"])
    assert new_m["det"].type.shape == (3, 5)
    assert isinstance(new_m.dim_lengths["dim1"], Constant) and new_m.dim_lengths["dim1"].data == 3
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], Constant) and np.all(new_m["data2"].data == [1, 2, 3, 4, 5])


def test_freeze_dim_after_do_intervention():
    with Model(coords={"test_dim": range(5)}) as m:
        mu = Data("mu", [0, 1, 2, 3, 4], dims="test_dim")
        x = Normal("x", mu=mu, dims="test_dim")

    do_m = do(m, {mu: mu * 100})
    assert do_m["x"].type.shape == (None,)

    frozen_do_m = freeze_dims_and_data(do_m)
    assert frozen_do_m["x"].type.shape == (5,)


def test_freeze_dims_and_data_partially_observed_rv():
    # Regression test for #7387

    with Model(coords={"a": [0, 1, 2]}) as model:
        y = Normal("y", 0, observed=[0, 0, np.nan], dims="a")

    with pytest.raises(NotConstantValueError):
        constant_fold([y.shape])

    frozen_y = freeze_dims_and_data(model)["y"]
    assert constant_fold([frozen_y.shape]) == (3,)


class TestModelToFloat32:
    @staticmethod
    def _mixed_model():
        rng = np.random.default_rng(4)
        x_data = rng.normal(size=10)
        with Model(coords={"g": range(3)}) as m:
            x = Data("x", x_data)
            idx = Data("idx", np.arange(10))
            beta = Normal("beta")
            sigma = HalfNormal("sigma")
            z = pm.ZeroSumNormal("z", dims="g")
            det = Deterministic("det", beta * x + z.mean())
            Normal("y", mu=det[idx], sigma=sigma, observed=x_data * 2)
        return m

    def test_dtypes_converted(self):
        m = self._mixed_model()
        m32 = model_to_float32(m)

        for name in ("x", "beta", "sigma", "z", "det", "y"):
            assert m32[name].type.dtype == "float32", name
        assert m32["idx"].type.dtype == m["idx"].type.dtype
        for rv in m32.free_RVs + m32.observed_RVs:
            assert m32.rvs_to_values[rv].type.dtype == "float32"
        # Static shapes and transforms are preserved
        for name in ("x", "beta", "sigma", "z", "det", "y"):
            assert m32[name].type.shape == m[name].type.shape, name
        assert type(m32.rvs_to_transforms[m32["z"]]) is type(m.rvs_to_transforms[m["z"]])
        with Model() as m_static:
            pm.ZeroSumNormal("z", shape=(3,))
        assert model_to_float32(m_static)["z"].type.shape == (3,)

    def test_logp_and_draws(self):
        m = self._mixed_model()
        m32 = model_to_float32(m)

        ip64 = m.initial_point()
        logp64 = m.compile_logp()(ip64)
        with pytensor.config.change_flags(floatX="float32"):
            ip32 = m32.initial_point()
            assert all(v.dtype == "float32" for v in ip32.values())
            logp32 = m32.compile_logp()(ip32)
            dlogp32 = m32.compile_dlogp()(ip32)
        np.testing.assert_allclose(logp32, logp64, rtol=1e-5)
        assert np.asarray(dlogp32).dtype == "float32"
        assert pm.draw(m32["z"], random_seed=1).dtype == "float32"

    def test_round_trip(self):
        m = self._mixed_model()
        m64 = model_to_float64(model_to_float32(m))
        for name in ("x", "beta", "sigma", "z", "det", "y"):
            assert m64[name].type.dtype == "float64", name
        np.testing.assert_allclose(
            m64.compile_logp()(m64.initial_point()),
            m.compile_logp()(m.initial_point()),
            rtol=1e-5,
        )

    def test_explicit_cast_redirected(self):
        with Model() as m:
            x = Data("x", np.arange(5))  # int64
            Normal("y", mu=x.astype("float64"), observed=np.zeros(5))
        m32 = model_to_float32(m)
        assert m32["y"].type.dtype == "float32"
        assert m32["y"].owner.inputs[3].type.dtype == "float32"

    def test_preserves_initvals(self):
        with Model() as m:
            sigma = HalfNormal("sigma", initval=np.array(5.0))
            beta = Normal("beta", initval="prior")
        m32 = model_to_float32(m)
        initval = m32.rvs_to_initial_values[m32["sigma"]]
        assert initval.dtype == "float32" and initval == 5.0
        assert m32.rvs_to_initial_values[m32["beta"]] == "prior"

    def test_sample_smoke(self):
        m32 = model_to_float32(self._mixed_model())
        with pytensor.config.change_flags(floatX="float32"):
            with m32:
                idata = pm.sample(
                    draws=10,
                    tune=10,
                    chains=1,
                    progressbar=False,
                    random_seed=1,
                    compute_convergence_checks=False,
                )
        # The sampler may store draws as float64; just check sampling worked
        assert np.isfinite(idata.posterior["beta"]).all()

    def test_transform_with_foreign_dtype_constants(self):
        # Transforms travel with the model as objects and may bake float64 constants
        # into value-space graphs; model_to_float32 must keep those graphs float32.
        import pytensor.tensor as pt

        from pymc.logprob.transforms import Transform

        class ScaledTransform(Transform):
            name = "scaled"
            scale = np.array(2.0, dtype="float64")  # baked float64 constant

            def forward(self, value, *inputs):
                return value * self.scale

            def backward(self, value, *inputs):
                return value / self.scale

            def log_jac_det(self, value, *inputs):
                return -pt.log(self.scale) * pt.ones_like(value)

        with Model() as m:
            Normal("x", 0, 1, default_transform=ScaledTransform())

        m32 = model_to_float32(m)
        with pytensor.config.change_flags(floatX="float32"):
            ip32 = m32.initial_point()
            assert all(v.dtype == "float32" for v in ip32.values())
            logp32 = m32.compile_logp()(ip32)
        np.testing.assert_allclose(logp32, m.compile_logp()(m.initial_point()), rtol=1e-5)
