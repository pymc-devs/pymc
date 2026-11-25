#   Copyright 2026 - present The PyMC Developers
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
import pytensor.xtensor.math as ptxm
import pytest

from pytensor import scan
from pytensor.tensor.random.op import RandomVariable
from pytensor.xtensor import as_xtensor, full_like
from pytensor.xtensor.basic import xtensor_from_tensor
from pytensor.xtensor.random import as_xrv, shared_rng
from scipy import stats

import pymc.distributions as regular_distributions

from pymc import draw as pm_draw
from pymc.dims import Censored, CustomDist, HalfNormal, Normal, Uniform
from pymc.model.core import Model
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph

pytestmark = pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
)


def normal_dist(mu, sigma, extra_dims):
    return Normal.dist(mu, sigma, dim_lengths=extra_dims)


def normal_logp(value, mu, sigma):
    return -0.5 * ((value - mu) / sigma) ** 2 - ptxm.log(sigma * np.sqrt(2 * np.pi))


def normal_logcdf(value, mu, sigma):
    return ptxm.log(1 + ptxm.erf((value - mu) / (sigma * np.sqrt(2)))) - np.log(2)


def test_custom_dist():
    """Infer logp while propagating parameter dims, extra dims and a support point override."""
    coords = {"city": range(3), "obs": range(2)}
    mu = as_xtensor([1.0, 2.0, 3.0], dims=("city",))
    with Model(coords=coords) as model:
        x = CustomDist(
            "x",
            mu,
            2,
            dist=normal_dist,
            dims=("obs", "city"),
            support_point=lambda rv, mu, sigma: full_like(rv, 2),
        )

    with Model(coords=coords) as reference_model:
        regular_distributions.Normal(
            "x",
            [1.0, 2.0, 3.0],
            2,
            shape=(reference_model.dim_lengths["obs"], 3),
            dims=("obs", "city"),
        )

    assert x.dims == ("obs", "city")
    assert_equivalent_random_graph(model, reference_model)
    assert_equivalent_logp_graph(model, reference_model)
    np.testing.assert_array_equal(model.initial_point()["x"], np.full((2, 3), 2))


@pytest.mark.parametrize("dist", [None, normal_dist])
def test_logp_override(dist):
    """The callback uses named broadcasting and overrides any inferred logp."""
    data = np.arange(6, dtype="float64").reshape(2, 3)
    sigma = as_xtensor([1.0, 2.0], dims=("obs",))
    with Model(coords={"city": range(3), "obs": range(2)}) as model:
        mu = Normal("mu", dims="city")
        x = CustomDist(
            "x",
            mu,
            sigma,
            dist=dist,
            logp=lambda value, mu, sigma: 2 * normal_logp(value, mu, sigma),
            observed=as_xtensor(data, dims=("obs", "city")),
            dims=("city", "obs"),
        )

    np.testing.assert_allclose(
        model.compile_logp(vars=[x])(model.initial_point()),
        2 * stats.norm.logpdf(data, scale=np.array([1.0, 2.0])[:, None]).sum(),
    )


def test_logp_only_initial_point():
    mu = as_xtensor([1, 2, 3], dims=("city",))
    with Model(coords={"city": range(3), "obs": range(2)}) as model:
        x = CustomDist("x", mu, 1, logp=normal_logp, dtype="float32", dims=("obs", "city"))

    initial_point = model.initial_point()["x"]
    np.testing.assert_array_equal(initial_point, np.zeros((2, 3)))
    assert initial_point.dtype == "float32"
    with pytest.raises(NotImplementedError, match="Attempted to run dist"):
        pm_draw(x)


def test_default_support_point_reduced_param_dim():
    def averaged_normal_dist(mu, extra_dims):
        return Normal.dist(mu.mean("city"), dim_lengths=extra_dims)

    mu = as_xtensor(np.arange(3, dtype="float64"), dims=("city",))
    with Model(coords={"obs": range(2)}) as model:
        CustomDist("x", mu, dist=averaged_normal_dist, dims="obs")

    np.testing.assert_array_equal(model.initial_point()["x"], np.ones(2))


def lognormal_dist(mu, sigma, extra_dims):
    return ptxm.exp(normal_dist(mu, sigma, extra_dims))


def shifted_halfnormal_dist(mu, sigma, extra_dims):
    return mu + 2 * HalfNormal.dist(sigma, dim_lengths=extra_dims)


def shifted_uniform_dist(mu, sigma, extra_dims):
    return Uniform.dist(mu + 10, mu + 11, dim_lengths=extra_dims)


@pytest.mark.parametrize(
    "dist, expected",
    [
        (lognormal_dist, np.exp(0.5)),
        (shifted_halfnormal_dist, 0.5 + 2 * 3.0),
        (shifted_uniform_dist, 11.0),
    ],
)
def test_default_support_point_has_finite_logp(dist, expected):
    """The default support point is not on the boundary or outside the support (as zeros was)."""
    with Model(coords={"city": range(3)}) as model:
        CustomDist("x", 0.5, 3.0, dist=dist, dims="city")

    initial_point = model.initial_point()
    np.testing.assert_allclose(initial_point["x"], np.full(3, expected))
    assert np.isfinite(model.compile_logp()(initial_point))


def test_default_support_point_nested_random_variables():
    """Support points of dependent random variables, including a nested CustomDist, are chained."""

    def nested_dist(mu, extra_dims):
        first = CustomDist.dist(mu, 1, dist=lognormal_dist, dim_lengths=extra_dims)
        return HalfNormal.dist(first) + first

    with Model(coords={"city": range(3)}) as model:
        CustomDist("x", 0.5, dist=nested_dist, dims="city")

    # HalfNormal support point is its sigma
    np.testing.assert_allclose(model.initial_point()["x"], np.full(3, 2 * np.exp(0.5)))


def test_default_support_point_scan():
    def scan_dist(mu, extra_dims):
        def step(xtm1, rng):
            next_rng, x = regular_distributions.Normal.dist(xtm1 + 1, rng=rng).owner.outputs
            return x, next_rng

        rng = pytensor.shared(np.random.default_rng())
        xs, _next_rng = scan(
            step, outputs_info=[mu.values, rng], n_steps=3, name="xs", return_updates=False
        )
        return xtensor_from_tensor(xs, dims=("step", *mu.dims))

    mu = as_xtensor(np.zeros(2), dims=("city",))
    with Model(coords={"city": range(2), "step": range(3)}) as model:
        CustomDist("x", mu, dist=scan_dist, dims=("step", "city"))

    np.testing.assert_array_equal(
        model.initial_point()["x"], np.array([[1, 1], [2, 2], [3, 3]], dtype="float64")
    )


def test_default_support_point_without_inner_support_point():
    """Fall back to zeros if some random variable in the inner graph has no support point."""

    class NoSupportPointRV(RandomVariable):
        name = "no_support_point"
        signature = "()->()"
        dtype = "floatX"

        @classmethod
        def rng_fn(cls, rng, loc, size):
            return rng.normal(loc, 1, size)

    def no_support_point_dist(mu, extra_dims):
        _, rv = as_xrv(NoSupportPointRV())(
            mu, extra_dims=extra_dims, rng=shared_rng(seed=None), return_next_rng=True
        )
        return rv

    mu = as_xtensor(np.arange(3, dtype="float64"), dims=("city",))
    with Model(coords={"obs": range(2), "city": range(3)}) as model:
        CustomDist("x", mu, dist=no_support_point_dist, dims=("obs", "city"))

    np.testing.assert_array_equal(model.initial_point()["x"], np.zeros((2, 3)))


def test_censored_compound_custom_dist():
    """A factory can expand and infer logp for a compound CustomDist."""

    def logitnormal_dist(mu, sigma, extra_dims):
        return ptxm.sigmoid(normal_dist(mu, sigma, extra_dims))

    with Model(coords={"city": range(3)}) as model:
        base = CustomDist.dist(0, 1, dist=logitnormal_dist)
        y = Censored("y", base, lower=0.2, upper=0.8, dims="city")

    draws = pm_draw(y, draws=3, random_seed=1)
    assert draws.shape == (3, 3)
    assert ((draws >= 0.2) & (draws <= 0.8)).all()
    np.testing.assert_allclose(model.initial_point()["y"], np.full(3, 0.5))
    np.testing.assert_allclose(
        model.compile_logp()({"y": np.full(3, 0.5)}),
        3 * (stats.norm.logpdf(0) - np.log(0.25)),
    )


def test_censored_custom_dist():
    """Censoring dispatches the xtensor logp/logcdf overrides after expanding dims."""
    with Model(coords={"city": range(5)}) as model:
        base = CustomDist.dist(0, 1, dist=normal_dist, logp=normal_logp, logcdf=normal_logcdf)
        y = Censored("y", base, lower=-1, upper=1, dims="city")

    draws = pm_draw(y, draws=3, random_seed=1)
    assert draws.shape == (3, 5)
    assert ((draws >= -1) & (draws <= 1)).all()
    values = np.array([-1, -0.5, 0, 0.5, 1])
    expected = stats.norm.logcdf(-1) + stats.norm.logpdf(values[1:-1]).sum() + stats.norm.logsf(1)
    np.testing.assert_allclose(model.compile_logp()({"y": values}), expected)


def test_multiple_rngs():
    def compound_dist(mu, sigma, extra_dims):
        first = normal_dist(mu, sigma, extra_dims)
        return Normal.dist(first, sigma)

    rv = CustomDist.dist(0, 1, dist=compound_dist, dim_lengths={"city": 3})
    draws = pm_draw(rv, draws=3, random_seed=1)
    np.testing.assert_array_equal(draws, pm_draw(rv, draws=3, random_seed=1))
    assert not np.array_equal(draws[0], draws[1])


def test_dist_must_use_extra_dims():
    with Model(coords={"city": range(3)}):
        with pytest.raises(ValueError, match="output is missing dims"):
            CustomDist(
                "x", 0, 1, dist=lambda mu, sigma, extra_dims: Normal.dist(mu, sigma), dims="city"
            )
