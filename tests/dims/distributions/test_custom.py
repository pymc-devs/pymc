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
import pytensor.xtensor.math as ptxm
import pytest

from pytensor.xtensor import as_xtensor, full_like

import pymc.distributions as regular_distributions

from pymc import draw as pm_draw
from pymc.dims import CustomDist, Normal, Poisson
from pymc.model.core import Model
from tests.dims.utils import assert_equivalent_logp_graph, assert_equivalent_random_graph

pytestmark = pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
)


class TestCustomDistSymbolic:
    """Tests for the symbolic (dist=) path of pmd.CustomDist."""

    def test_compound_non_xrv_output(self):
        """Compound dist with non-XRV output gets extra dims via the extra_dims argument."""

        def logitnormal_dist(mu, sigma, extra_dims):
            return ptxm.sigmoid(Normal.dist(mu=mu, sigma=sigma, dim_lengths=extra_dims))

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            x = CustomDist("x", 0, 1, dist=logitnormal_dist, dims="city")

        assert set(x.dims) == {"city"}

        draws = pm_draw(model["x"], draws=5)
        assert draws.shape == (5, 5)
        assert (draws > 0).all() and (draws < 1).all()

    def test_basic(self):
        """Symbolic path: dist function wrapping Normal.dist, compared against regular Normal."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist("x", 0, 1, dist=normal_dist, dims="city")

        with Model(coords=coords) as reference_model:
            regular_distributions.Normal("x", 0, 1, dims="city")

        assert_equivalent_random_graph(model, reference_model)
        assert_equivalent_logp_graph(model, reference_model)

    def test_param_dims_propagate(self):
        """Params with dims propagate to the output."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        coords = {"city": range(5)}
        mu = as_xtensor(np.array([0, 1, 2, 3, 4]), dims=("city",))
        sigma = as_xtensor(np.array([1, 2, 3, 4, 5]), dims=("city",))

        with Model(coords=coords) as model:
            x = CustomDist("x", mu, sigma, dist=normal_dist)

        assert set(x.dims) == {"city"}
        assert x.type.shape == (5,)

    def test_dist_must_use_extra_dims(self):
        """A dist function that ignores extra_dims fails when extra dims are requested."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma)

        coords = {"city": range(5)}
        with Model(coords=coords):
            with pytest.raises(ValueError, match="output is missing dims"):
                CustomDist("x", 0, 1, dist=normal_dist, dims="city")


class TestCustomDistArbitrary:
    """Tests for the arbitrarily-defined (logp=) path of pmd.CustomDist."""

    def test_logp_only_cannot_sample(self):
        """Forward sampling a CustomDist defined only through logp raises."""

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist("x", 0, 1, logp=normal_logp, dims="city")

        with pytest.raises(NotImplementedError, match="CustomDist that was constructed without"):
            pm_draw(model["x"])

    def test_logp_basic(self):
        """Arbitrary path with logp function and dims on output."""

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

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

    def test_hybrid_dist_logp(self):
        """Hybrid path: dist for sampling + logp override."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                dist=normal_dist,
                logp=normal_logp,
                dims="city",
            )

        # Verify sampling works (via draw)
        draws = pm_draw(model["x"], draws=3)
        assert draws.shape == (3, 5)

        # Verify logp evaluates
        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

    def test_hybrid_derived_params(self):
        """Hybrid path: dist derives params."""

        def poisson_dist(a, b, c, extra_dims):
            lam = a + b + c
            return Poisson.dist(mu=lam, dim_lengths=extra_dims)

        def poisson_logp(value, a, b, c):
            value = value.values
            lam = (a + b + c).values
            return pt.sum(value * pt.log(lam) - lam - pt.gammaln(value + 1))

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0.5,
                0.3,
                0.2,
                dist=poisson_dist,
                logp=poisson_logp,
                dims="city",
            )

        # pm.draw — evaluates the full dist graph (compound inference)
        draws = pm_draw(model["x"], draws=3)
        assert draws.shape == (3, 5)
        assert draws.dtype.kind == "i"

        # logp evaluates
        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

    def test_hybrid_logp_override(self):
        """Hybrid path: verify user logp overrides auto-derived logp."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        def scaled_logp(value, mu, sigma):
            """Custom logp that multiplies normal logp by 2."""
            value, mu, sigma = value.values, mu.values, sigma.values
            normal_logp = pt.sum(
                -0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi))
            )
            return 2.0 * normal_logp

        coords = {"city": range(5)}
        with Model(coords=coords) as model_hybrid:
            CustomDist(
                "x",
                0,
                1,
                dist=normal_dist,
                logp=scaled_logp,
                dims="city",
            )

        # Auto-derived logp (no logp override)
        with Model(coords=coords) as model_auto:
            CustomDist(
                "x",
                0,
                1,
                dist=normal_dist,
                dims="city",
            )

        ip = model_hybrid.initial_point()
        hybrid_logp = model_hybrid.compile_logp()(ip)
        auto_logp = model_auto.compile_logp()(ip)
        # Hybrid logp should be 2x auto logp
        np.testing.assert_allclose(hybrid_logp, 2.0 * auto_logp)

    def test_hybrid_basic_dims(self):
        """Hybrid path with dims on params."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

        coords = {"city": range(5)}
        mu = as_xtensor(np.array([0.0, 0.5, 1.0, 1.5, 2.0]), dims=("city",))
        sigma = as_xtensor(np.array([1.0, 1.1, 1.2, 1.3, 1.4]), dims=("city",))

        with Model(coords=coords) as model:
            x = CustomDist("x", mu, sigma, dist=normal_dist, logp=normal_logp)

        assert set(x.dims) == {"city"}
        ip = model.initial_point()
        logp_val = model.compile_logp()(ip)
        assert np.isfinite(logp_val)

        draws = pm_draw(model["x"], draws=3)
        assert draws.shape == (3, 5)

    def test_logcdf(self):
        """Arbitrary path with logcdf function."""

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

        def normal_logcdf(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(
                pt.log(pt.erf((value - mu) / (sigma * pt.sqrt(2.0))) + 1.0)
                - pt.log(pt.constant(2.0))
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
        """Arbitrary path with mu as a model variable (no dims on mu)."""

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

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

    def test_logp_inferred_with_other_overrides(self):
        """The logp is still derived from the dist graph when only other methods are overridden."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        def custom_support_point(rv, mu, sigma):
            return full_like(rv, mu)

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                1.0,
                2.0,
                dist=normal_dist,
                support_point=custom_support_point,
                dims="city",
            )

        with Model(coords=coords) as reference_model:
            regular_distributions.Normal("x", 1.0, 2.0, dims="city")

        ip = model.initial_point()
        np.testing.assert_allclose(ip["x"], np.ones(5))
        np.testing.assert_allclose(
            model.compile_logp()(ip),
            reference_model.compile_logp()(ip),
        )

    def test_default_support_point(self):
        """Default support point of a CustomDist without a dist function."""

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                logp=normal_logp,
                dims="city",
            )

        # The initial point does not require evaluating the random graph,
        # which would raise for a CustomDist without a dist function
        ip = model.initial_point()
        np.testing.assert_allclose(ip["x"], np.zeros(5))

    def test_default_support_point_reduced_param_dim(self):
        """Default support point drops param dims that the dist graph reduced away."""

        def averaged_normal_dist(mu, extra_dims):
            return Normal.dist(mu.mean("city"), dim_lengths=extra_dims)

        def normal_logp(value, mu):
            value, mu_mean = value.values, mu.mean("city").values
            return pt.sum(-0.5 * (value - mu_mean) ** 2 - pt.log(pt.sqrt(2 * np.pi)))

        coords = {"city": range(5), "obs": range(3)}
        mu = as_xtensor(np.arange(5, dtype="float64"), dims=("city",))

        with Model(coords=coords) as model:
            CustomDist("x", mu, dist=averaged_normal_dist, logp=normal_logp, dims="obs")

        ip = model.initial_point()
        np.testing.assert_allclose(ip["x"], np.zeros(3))

    def test_hybrid_support_point(self):
        """Hybrid path with custom support_point."""

        def normal_dist(mu, sigma, extra_dims):
            return Normal.dist(mu, sigma, dim_lengths=extra_dims)

        def normal_logp(value, mu, sigma):
            value, mu, sigma = value.values, mu.values, sigma.values
            return pt.sum(-0.5 * ((value - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))

        def custom_support_point(rv, mu, sigma):
            return full_like(rv, mu)

        coords = {"city": range(5)}
        with Model(coords=coords) as model:
            CustomDist(
                "x",
                0,
                1,
                dist=normal_dist,
                logp=normal_logp,
                support_point=custom_support_point,
                dims="city",
            )

        # Like all dims distributions, support_point is reachable after lowering
        ip = model.initial_point()
        np.testing.assert_allclose(ip["x"], np.zeros(5))
