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
import re

import numpy as np
import pytest
import scipy

from pytensor.xtensor import as_xtensor
from xarray import DataArray

import pymc as pm

from pymc import dims as pmx
from pymc import icdf, logccdf, logcdf, logp
from pymc.dims import Normal

pytestmark = pytest.mark.filterwarnings(
    "error",
    "ignore::numba.core.errors.NumbaPerformanceWarning",
)


def test_distribution_dims():
    coords = {
        "a": range(2),
        "b": range(3),
        "c": range(5),
        "d": range(7),
    }
    with pm.Model(coords=coords) as model:
        x = pmx.Data("x", np.random.randn(2, 3, 5), dims=("a", "b", "c"))
        y1 = pmx.Normal("y1", mu=x)
        assert y1.type.dims == ("a", "b", "c")
        assert y1.eval().shape == (2, 3, 5)

        y2 = pmx.Normal("y2", mu=x, dims=("a", "b", "c"))  # redundant
        assert y2.type.dims == ("a", "b", "c")
        assert y2.eval().shape == (2, 3, 5)

        y3 = pmx.Normal("y3", mu=x, dims=("b", "a", "c"))  # Implies a transpose
        assert y3.type.dims == ("b", "a", "c")
        assert y3.eval().shape == (3, 2, 5)

        y4 = pmx.Normal("y4", mu=x, dims=("a", ...))
        assert y4.type.dims == ("a", "b", "c")
        assert y4.eval().shape == (2, 3, 5)

        y5 = pmx.Normal("y5", mu=x, dims=("b", ...))  # Implies a transpose
        assert y5.type.dims == ("b", "a", "c")
        assert y5.eval().shape == (3, 2, 5)

        y6 = pmx.Normal("y6", mu=x, dims=("b", ..., "a"))  # Implies a transpose
        assert y6.type.dims == ("b", "c", "a")
        assert y6.eval().shape == (3, 5, 2)

        y7 = pmx.Normal("y7", mu=x, dims=(..., "b"))  # Implies a transpose
        assert y7.type.dims == ("a", "c", "b")
        assert y7.eval().shape == (2, 5, 3)

        y8 = pmx.Normal("y8", mu=x, dims=("d", "a", "b", "c"))  # Adds an extra dimension
        assert y8.type.dims == ("d", "a", "b", "c")
        assert y8.eval().shape == (7, 2, 3, 5)

        y9 = pmx.Normal("y9", mu=x, dims=("d", ...))  # Adds an extra dimension
        assert y9.type.dims == ("d", "a", "b", "c")
        assert y9.eval().shape == (7, 2, 3, 5)

        y10 = pmx.Normal(
            "y10", mu=x, dims=("b", "a", "c", "d")
        )  # Adds an extra dimension and implies a transpose
        assert y10.type.dims == ("b", "a", "c", "d")
        assert y10.eval().shape == (3, 2, 5, 7)

        y11 = pmx.Normal(
            "y11", mu=x, dims=("c", ..., "d")
        )  # Adds an extra dimension and implies a transpose
        assert y11.type.dims == ("c", "a", "b", "d")
        assert y11.eval().shape == (5, 2, 3, 7)

        # Invalid cases
        err_msg = "Provided dims ('a', 'b') do not match the distribution's output dims ('a', 'b', 'c'). Use ellipsis to specify all other dimensions."
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            # Missing a dimension
            pmx.Normal("y_bad", mu=x, dims=("a", "b"))

        err_msg = "Provided dims ('d',) do not match the distribution's output dims ('d', 'a', 'b', 'c'). Use ellipsis to specify all other dimensions."
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            # Only specifies the extra dimension
            pmx.Normal("y_bad", mu=x, dims=("d",))

        err_msg = "Not all dims ('a', 'b', 'c', 'e') are part of the model coords. Add them at initialization time or use `model.add_coord` before defining the distribution"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            # Extra dimension not in coords
            pmx.Normal("y_bad", mu=x, dims=("a", "b", "c", "e"))


def test_multivariate_distribution_dims():
    coords = {
        "batch": range(2),
        "core1": range(3),
        "core2": range(3),
    }
    with pm.Model(coords=coords) as m:
        mu = pmx.Normal("mu", dims=("batch", "core1"))
        chol, _, _ = pm.LKJCholeskyCov(
            "chol",
            eta=1,
            n=3,
            sd_dist=pm.Exponential.dist(1),
        )
        chol_xr = pmx.math.as_xtensor(chol, dims=("core1", "core2"))

        x1 = pmx.MvNormal(
            "x1",
            mu,
            chol=chol_xr,
            core_dims=("core1", "core2"),
        )
        assert x1.type.dims == ("batch", "core1")
        assert x1.eval().shape == (2, 3)

        x2 = pmx.MvNormal(
            "x2",
            mu,
            chol=chol_xr,
            core_dims=("core1", "core2"),
            dims=("batch", "core1"),
        )
        assert x2.type.dims == ("batch", "core1")
        assert x2.eval().shape == (2, 3)

        x3 = pmx.MvNormal(
            "x3",
            mu,
            chol=chol_xr,
            core_dims=("core2", "core1"),
            dims=("batch", ...),
        )
        assert x3.type.dims == ("batch", "core1")
        assert x3.eval().shape == (2, 3)

        x4 = pmx.MvNormal(
            "x4",
            mu,
            chol=chol_xr,
            core_dims=("core1", "core2"),
            # Implies transposition
            dims=("core1", ...),
        )
        assert x4.type.dims == ("core1", "batch")
        assert x4.eval().shape == (3, 2)

        # Errors
        err_msg = "MvNormal requires 2 core_dims"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            # Missing core_dims
            pmx.MvNormal(
                "x_bad",
                mu,
                chol=chol_xr,
            )

        with pytest.raises(ValueError, match="Dimension batch not found in either input"):
            pmx.MvNormal(
                "x_bad",
                mu,
                chol=chol_xr,
                # Invalid because batch is not on chol_xr
                core_dims=("core1", "batch"),
            )

        with pytest.raises(ValueError, match="Parameter mu_renamed has invalid core dimensions"):
            mu_renamed = mu.rename({"batch": "core2"})
            mu_renamed.name = "mu_renamed"
            pmx.MvNormal(
                "x_bad",
                mu_renamed,
                chol=chol_xr,
                # Invalid because mu has both core dimensions (after renaming)
                core_dims=("core1", "core2"),
            )

        # Invalid because core2 is not a core output dimension
        err_msg = "Dimensions {'core2'} do not exist. Expected one or more of: ('batch', 'core1')"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            pmx.MvNormal(
                "x_bad", mu, chol=chol_xr, core_dims=("core1", "core2"), dims=("core2", ...)
            )


@pytest.mark.parametrize("rv_unique_dims", [False, True])
@pytest.mark.parametrize("value_unique_dims", [False, True])
@pytest.mark.parametrize("value_aligned", [True, False])
def test_density_helpers(rv_unique_dims, value_unique_dims, value_aligned):
    mean = np.array([1, 2, 3])
    sigma = 1
    x = Normal.dist(
        mu=as_xtensor(mean, dims=("city",)),
        sigma=sigma,
        dim_lengths={"time": 2},
    )

    # Manipulate x to generate the desired type for x_value
    x_value_ref = x
    if rv_unique_dims:
        x_value_ref = x_value_ref.isel(time=0)
    if value_unique_dims:
        x_value_ref = x_value_ref.expand_dims(value_batch=7, axis=0)
    x_value = x_value_ref.type()

    # Get a dense random test value with the shape of x_value
    shape = pm.math.as_tensor(x_value_ref.shape).eval(mode="FAST_COMPILE")
    x_test_value = np.random.default_rng(211).normal(size=shape)

    # This is the test value we'll use with the reference scipy functions
    x_aligned_test_value = x_test_value

    if not value_aligned:
        x_value = x_value.T.type()
        x_test_value = x_test_value.T

    def assert_allclose_broadcastable(res, exp, **kwargs):
        if rv_unique_dims:
            exp = exp[..., None, :]
            assert res.shape[-2] == 2
        exp_bcast = np.broadcast_to(exp, res.shape)
        np.testing.assert_allclose(res, exp_bcast, **kwargs)

    x_logp = logp(x, x_value)
    if value_unique_dims:
        assert x_logp.dims == ("value_batch", *x.dims)
    else:
        assert x.type.is_super(
            x_logp.type
        )  # x_logp type is compatible (but possible more precise) than x
    assert_allclose_broadcastable(
        x_logp.eval({x_value: DataArray(x_test_value, dims=x_value.dims)}),
        scipy.stats.norm.logpdf(x_aligned_test_value, mean, sigma),
    )

    x_logcdf = logcdf(x, x_value)
    if value_unique_dims:
        assert x_logcdf.dims == ("value_batch", *x.dims)
    else:
        assert x.type.is_super(x_logcdf.type)
    assert_allclose_broadcastable(
        x_logcdf.eval({x_value: DataArray(x_test_value, dims=x_value.dims)}),
        scipy.stats.norm.logcdf(x_aligned_test_value, mean, sigma),
        rtol=1e-6,
    )

    x_logccdf = logccdf(x, x_value)
    if value_unique_dims:
        assert x_logccdf.dims == ("value_batch", *x.dims)
    else:
        assert x.type.is_super(x_logccdf.type)
    assert_allclose_broadcastable(
        x_logccdf.eval({x_value: DataArray(x_test_value, dims=x_value.dims)}),
        scipy.stats.norm.logsf(x_aligned_test_value, mean, sigma),
        rtol=1e-6,
    )

    icdf_test_value = scipy.special.expit(x_test_value)
    icdf_aligned_test_value = scipy.special.expit(x_aligned_test_value)
    x_icdf = icdf(x, x_value)
    if value_unique_dims:
        assert x_icdf.dims == ("value_batch", *x.dims)
    else:
        assert x.type.is_super(x_icdf.type)
    assert_allclose_broadcastable(
        x_icdf.eval({x_value: DataArray(icdf_test_value, dims=x_value.dims)}),
        scipy.stats.norm.ppf(icdf_aligned_test_value, mean, sigma),
    )
