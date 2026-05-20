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
import pytensor.tensor as pt
import pytest

from pytensor.xtensor import as_xtensor

import pymc.distributions as regular_distributions

from pymc.dims import CustomDist, Normal
from pymc.model.core import Model
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph

pytestmark = pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
)


class TestCustomDistSymbolic:
    """Tests for the symbolic (dist=) path of pmd.CustomDist."""

    def test_basic(self):
        """Symbolic path: dist function wrapping Normal.dist, compared against regular Normal."""

        def normal_dist(mu, sigma):
            return Normal.dist(mu, sigma)

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist("x", 0, 1, dist=normal_dist, dims="city")

        with Model(coords=coords) as reference_model:
            regular_distributions.Normal("x", 0, 1, dims="city")

        assert_equivalent_random_graph(model, reference_model)
        assert_equivalent_logp_graph(model, reference_model)

    def test_param_dims_propagate(self):
        """Params with dims propagate to the output."""

        def normal_dist(mu, sigma):
            return Normal.dist(mu, sigma)

        coords = {"city": range(5)}
        mu = as_xtensor(np.array([0, 1, 2, 3, 4]), dims=("city",))
        sigma = as_xtensor(np.array([1, 2, 3, 4, 5]), dims=("city",))

        with Model(coords=coords) as model:
            x = CustomDist("x", mu, sigma, dist=normal_dist)

        assert set(x.dims) == {"city"}
        assert x.type.shape == (5,)


class TestCustomDistBlackbox:
    """Tests for the black-box (logp=/random=) path of pmd.CustomDist."""

    def test_logp_basic(self):
        """Black-box path with logp function and dims on output."""

        def normal_logp(value, mu, sigma):
            v = value.values
            return pt.sum(
                -0.5 * ((v - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * pt.constant(np.pi)))
            )

        coords = {"city": range(5)}
        rng = np.random.default_rng(42)
        observed = as_xtensor(rng.normal(0, 1, size=5), dims=("city",))

        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                logp=normal_logp,
                observed=observed,
                dims="city",
            )

        # Test that logp evaluates without error and returns finite values
        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

    def test_random_logp(self):
        """Black-box path with both random and logp."""

        def normal_logp(value, mu, sigma):
            v = value.values
            return pt.sum(
                -0.5 * ((v - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * pt.constant(np.pi)))
            )

        def normal_random(mu, sigma, rng=None, size=None):
            return rng.normal(loc=mu, scale=sigma, size=size)

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                logp=normal_logp,
                random=normal_random,
                dims="city",
            )

        # Verify shape via draw
        from pymc import draw as pm_draw

        draws = pm_draw(model["x"], draws=3)
        assert draws.shape == (3, 5)

        # Verify logp
        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

    def test_logcdf(self):
        """Black-box path with logcdf function."""

        def normal_logp(value, mu, sigma):
            v = value.values
            return pt.sum(
                -0.5 * ((v - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * pt.constant(np.pi)))
            )

        def normal_logcdf(value, mu, sigma):
            v = value.values
            return pt.sum(
                pt.log(pt.erf((v - mu) / (sigma * pt.sqrt(2.0))) + 1.0) - pt.log(pt.constant(2.0))
            )

        coords = {"city": range(5)}
        rng = np.random.default_rng(42)
        observed = as_xtensor(rng.normal(0, 1, size=5), dims=("city",))

        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                logp=normal_logp,
                logcdf=normal_logcdf,
                observed=observed,
                dims="city",
            )

        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

    def test_mu_as_model_var(self):
        """Black-box path with mu as a model variable (no dims on mu)."""

        def normal_logp(value, mu, sigma):
            v = value.values
            return pt.sum(
                -0.5 * ((v - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * pt.constant(np.pi)))
            )

        coords = {"city": range(5)}
        rng = np.random.default_rng(42)
        observed = as_xtensor(rng.normal(0, 1, size=5), dims=("city",))

        with Model(coords=coords) as model:
            mu = Normal("mu", 0, 1)
            CustomDist(
                "x",
                mu,
                1,
                logp=normal_logp,
                observed=observed,
                dims="city",
            )

        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

    def test_support_point(self):
        """Black-box path with custom support_point."""

        def normal_logp(value, mu, sigma):
            v = value.values
            return pt.sum(
                -0.5 * ((v - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * pt.constant(np.pi)))
            )

        def custom_support_point(rv, size, mu, sigma):
            return pt.full_like(rv, mu)

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                logp=normal_logp,
                support_point=custom_support_point,
                dims="city",
            )

        from pymc.distributions.distribution import support_point

        sp = support_point(model["x"])
        np.testing.assert_allclose(sp.eval(), np.zeros(5))
