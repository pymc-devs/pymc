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
import pytensor.xtensor.math as ptxm
import pytest

from pytensor.xtensor import as_xtensor, full_like
from scipy import stats

import pymc.distributions as regular_distributions

from pymc import draw as pm_draw
from pymc.dims import Censored, CustomDist, Normal
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

    np.testing.assert_array_equal(model.initial_point()["x"], np.zeros(2))


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
