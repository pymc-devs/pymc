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
import xarray as xr

import pymc as pm

from pymc.model.transform.transforms import constrain_values, unconstrain_values


def test_unconstrain_roundtrip_with_rv_dependent_transform():
    """Regression: transforms that reference other model RVs (e.g. TruncatedNormal bounds)."""
    rng = np.random.default_rng(42)
    with pm.Model() as model:
        lower = pm.Normal("lower", mu=0, sigma=1)
        upper = pm.Normal("upper", mu=5, sigma=1)
        pm.TruncatedNormal("x", mu=2, sigma=1, lower=lower, upper=upper)

    posterior = pm.sample_prior_predictive(
        draws=10,
        random_seed=rng,
        model=model,
    ).prior.to_dataset()

    with model:
        unconstrained = unconstrain_values(posterior, compile_kwargs={"mode": "FAST_COMPILE"})
        roundtrip = constrain_values(unconstrained, compile_kwargs={"mode": "FAST_COMPILE"})

    assert set(roundtrip.coords) == {"chain", "draw"}
    xr.testing.assert_equal(roundtrip["chain"], posterior["chain"])
    xr.testing.assert_equal(roundtrip["draw"], posterior["draw"])
    for rv in model.free_RVs:
        assert roundtrip[rv.name].dims == posterior[rv.name].dims
        np.testing.assert_allclose(
            roundtrip[rv.name].values,
            posterior[rv.name].values,
            atol=1e-6,
            err_msg=f"Roundtrip failed for {rv.name}",
        )


def test_unconstrain_roundtrip_linreg():
    """Roundtrip through unconstrain/constrain for a linear regression model."""
    rng = np.random.default_rng(42)
    with pm.Model() as model:
        x = pm.Data("x", np.array([1.0, 2.0, 3.0]))
        y = pm.Data("y", np.array([2.0, 4.0, 6.0]))
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("obs", alpha + beta * x, sigma, observed=y)

    posterior = pm.sample_prior_predictive(
        draws=10,
        random_seed=rng,
        model=model,
    ).prior.to_dataset()

    with model:
        unconstrained = unconstrain_values(posterior, compile_kwargs={"mode": "FAST_COMPILE"})
        roundtrip = constrain_values(unconstrained, compile_kwargs={"mode": "FAST_COMPILE"})

    assert set(roundtrip.coords) == {"chain", "draw"}
    for rv in model.free_RVs:
        assert roundtrip[rv.name].dims == posterior[rv.name].dims
        np.testing.assert_allclose(
            roundtrip[rv.name].values,
            posterior[rv.name].values,
            atol=1e-6,
            err_msg=f"Roundtrip failed for {rv.name}",
        )


def test_unconstrain_roundtrip_zerosum():
    """Roundtrip through unconstrain/constrain for a ZeroSumNormal (shape-changing transform)."""
    rng = np.random.default_rng(42)
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", 1)
        pm.ZeroSumNormal("offset", sigma=sigma, shape=3)

    posterior = pm.sample_prior_predictive(
        draws=10,
        random_seed=rng,
        model=model,
    ).prior.to_dataset()

    with model:
        unconstrained = unconstrain_values(posterior, compile_kwargs={"mode": "FAST_COMPILE"})
        roundtrip = constrain_values(unconstrained, compile_kwargs={"mode": "FAST_COMPILE"})

    assert set(roundtrip.coords) == {"chain", "draw"}
    xr.testing.assert_equal(roundtrip["chain"], posterior["chain"])
    xr.testing.assert_equal(roundtrip["draw"], posterior["draw"])
    assert roundtrip["sigma"].dims == ("chain", "draw")
    # ZeroSum reduces dimensionality: shape 3 -> unconstrained shape 2;
    # roundtrip dim gets double prefix since forward pass already created a synthetic name
    assert roundtrip["offset"].dims == ("chain", "draw", "dim_dim_offset_dim_0_0_0")
    for rv in model.free_RVs:
        np.testing.assert_allclose(
            roundtrip[rv.name].values,
            posterior[rv.name].values,
            atol=1e-6,
            err_msg=f"Roundtrip failed for {rv.name}",
        )


def test_identity_transform_returns_views():
    """When no transform is applied, outputs should share memory with inputs."""
    rng = np.random.default_rng(42)
    with pm.Model() as model:
        pm.Normal("a", mu=0, sigma=1)
        pm.Normal("b", mu=0, sigma=1, shape=3)

    posterior = xr.Dataset(
        {
            "a": xr.DataArray(
                rng.normal(size=(2, 10)),
                dims=("chain", "draw"),
                coords={"chain": [0, 1], "draw": np.arange(10)},
            ),
            "b": xr.DataArray(
                rng.normal(size=(2, 10, 3)),
                dims=("chain", "draw", "b_dim_0"),
                coords={"chain": [0, 1], "draw": np.arange(10)},
            ),
        }
    )

    with model:
        result = unconstrain_values(posterior, compile_kwargs={"mode": "FAST_COMPILE"})

    assert set(result.coords) == {"chain", "draw"}
    xr.testing.assert_equal(result["chain"], posterior["chain"])
    xr.testing.assert_equal(result["draw"], posterior["draw"])
    for rv in model.free_RVs:
        assert result[rv.name].dims == posterior[rv.name].dims
        assert np.shares_memory(result[rv.name].values, posterior[rv.name].values)
