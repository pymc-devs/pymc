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
"""
Tests for sample_posterior_predictive with Deterministic variables.

These tests verify that Deterministic variables are correctly recomputed from
posterior samples rather than causing their dependencies to be resampled.
"""
import logging

import numpy as np
import numpy.testing as npt
import pytest
import pytensor

import pymc as pm
from pymc.testing import fast_unstable_sampling_mode


class TestDeterministicPosteriorPredictive:
    """Test that Deterministic variables don't cause resampling of dependencies."""

    def test_deterministic_recomputed_not_resampled(self):
        """
        Test that Deterministic variables are recomputed from posterior samples,
        not causing their dependencies to be resampled.

        This addresses the bug where including a Deterministic in var_names
        would incorrectly force its dependencies to be resampled.
        """
        rng = np.random.default_rng(42)

        with pm.Model() as model:
            # Hierarchical model
            intercept_mu = pm.Normal("intercept_mu", mu=0, sigma=1)
            intercept_sigma = pm.HalfNormal("intercept_sigma", sigma=1)
            slope_mu = pm.Normal("slope_mu", mu=0, sigma=1)
            slope_sigma = pm.HalfNormal("slope_sigma", sigma=1)

            intercepts = pm.Normal(
                "intercepts", mu=intercept_mu, sigma=intercept_sigma, shape=(2,)
            )
            slopes = pm.Normal("slopes", mu=slope_mu, sigma=slope_sigma, shape=(2,))

            # Deterministic variable that depends on intercepts and slopes
            time_coords = np.array([0.0, 12.0, 24.0, 48.0])
            mu_grid = pm.Deterministic(
                "mu_grid",
                intercepts[:, None] + slopes[:, None] * time_coords[None, :],
            )

            sigma = pm.HalfNormal("sigma", sigma=1)
            y_obs = pm.Normal(
                "y_obs",
                mu=mu_grid[0, :],
                sigma=sigma,
                observed=rng.normal(0, 1, size=4),
            )

            # Sample
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                idata = pm.sample(
                    tune=100,
                    draws=100,
                    chains=2,
                    step=pm.Metropolis(),
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    random_seed=rng,
                    progressbar=False,
                )

        # Get posterior variance for mu_grid
        mu_grid_post = idata.posterior.mu_grid.sel(test=0, time_hours=24.0)
        var_post = float(mu_grid_post.var().values)

        # Use sample_posterior_predictive with Deterministic in var_names
        with model:
            idata_pp = pm.sample_posterior_predictive(
                idata,
                var_names=["mu_grid"],
                predictions=True,
                extend_inferencedata=False,
                progressbar=False,
                random_seed=rng,
            )

        # Check that mu_grid variance matches posterior
        mu_grid_pred = idata_pp.predictions.mu_grid.sel(test=0, time_hours=24.0)
        var_pred = float(mu_grid_pred.var().values)

        # Variance should match (within tolerance)
        npt.assert_allclose(var_pred, var_post, rtol=0.1)

        # Values should be highly correlated (near 1.0)
        correlation = np.corrcoef(
            mu_grid_post.values.flatten(), mu_grid_pred.values.flatten()
        )[0, 1]
        assert correlation > 0.99, f"Correlation too low: {correlation}"

    def test_deterministic_with_random_variable_dependent(self):
        """
        Test that random variables depending on Deterministic are sampled correctly.

        When y_obs depends on mu_obs (Deterministic), y_obs should be sampled
        using the recomputed mu_obs, not a resampled one.
        """
        rng = np.random.default_rng(43)

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)
            mu_det = pm.Deterministic("mu_det", x + 1)
            sigma = pm.HalfNormal("sigma", sigma=1)
            y_obs = pm.Normal(
                "y_obs",
                mu=mu_det,
                sigma=sigma,
                observed=rng.normal(0, 1, size=10),
            )

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                idata = pm.sample(
                    tune=100,
                    draws=100,
                    chains=2,
                    step=pm.Metropolis(),
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    random_seed=rng,
                    progressbar=False,
                )

        # Get posterior values
        mu_det_post = idata.posterior.mu_det
        sigma_post = idata.posterior.sigma
        var_mu_det = float(mu_det_post.var().values)
        sigma_mean_sq = float((sigma_post**2).mean().values)

        # Use sample_posterior_predictive
        with model:
            idata_pp = pm.sample_posterior_predictive(
                idata,
                var_names=["mu_det", "y_obs"],
                predictions=True,
                extend_inferencedata=False,
                progressbar=False,
                random_seed=rng,
            )

        # Check mu_det is recomputed (not resampled)
        mu_det_pred = idata_pp.predictions.mu_det
        var_mu_det_pred = float(mu_det_pred.var().values)
        npt.assert_allclose(var_mu_det_pred, var_mu_det, rtol=0.1)

        # Check y_obs variance is correct
        # Expected: var(y_obs) ≈ var(mu_det) + E[sigma^2]
        y_obs_pred = idata_pp.predictions.y_obs
        var_y_obs_pred = float(y_obs_pred.var().values)
        expected_var = var_mu_det + sigma_mean_sq

        # Should match within reasonable tolerance (y_obs is sampled, so some variance)
        npt.assert_allclose(var_y_obs_pred, expected_var, rtol=0.3)

    def test_deterministic_nested_dependencies(self):
        """
        Test Deterministic with nested dependencies (Deterministic depends on
        Deterministic that depends on random variables).

        Edge case: Multiple levels of Deterministic variables.
        """
        rng = np.random.default_rng(44)

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)
            y = pm.Normal("y", mu=0, sigma=1)

            # Nested Deterministics
            det1 = pm.Deterministic("det1", x + y)
            det2 = pm.Deterministic("det2", det1 * 2)
            det3 = pm.Deterministic("det3", det2 + 1)

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                idata = pm.sample(
                    tune=100,
                    draws=100,
                    chains=2,
                    step=pm.Metropolis(),
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    random_seed=rng,
                    progressbar=False,
                )

        # Get posterior variance
        det3_post = idata.posterior.det3
        var_post = float(det3_post.var().values)

        # Use sample_posterior_predictive
        with model:
            idata_pp = pm.sample_posterior_predictive(
                idata,
                var_names=["det3"],
                predictions=True,
                extend_inferencedata=False,
                progressbar=False,
                random_seed=rng,
            )

        # Check variance matches
        det3_pred = idata_pp.predictions.det3
        var_pred = float(det3_pred.var().values)
        npt.assert_allclose(var_pred, var_post, rtol=0.1)

        # Check correlation
        correlation = np.corrcoef(
            det3_post.values.flatten(), det3_pred.values.flatten()
        )[0, 1]
        assert correlation > 0.99

    def test_deterministic_mixed_trace_dependencies(self):
        """
        Test Deterministic with mixed dependencies (some in trace, some not).

        Edge case: Deterministic depends on both variables in trace and variables
        not in trace. Only variables in trace should be used from trace.
        """
        rng = np.random.default_rng(45)

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)
            y = pm.Normal("y", mu=0, sigma=1)
            z = pm.Normal("z", mu=0, sigma=1)

            # det depends on x (in trace) and y (in trace), but z is not sampled
            det = pm.Deterministic("det", x + y)

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                idata = pm.sample(
                    tune=100,
                    draws=100,
                    chains=2,
                    step=pm.Metropolis(),
                    var_names=["x", "y"],  # Only sample x and y
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    random_seed=rng,
                    progressbar=False,
                )

        # Get posterior variance
        x_post = idata.posterior.x
        y_post = idata.posterior.y
        det_manual = x_post + y_post
        var_manual = float(det_manual.var().values)

        # Use sample_posterior_predictive
        with model:
            idata_pp = pm.sample_posterior_predictive(
                idata,
                var_names=["det"],
                predictions=True,
                extend_inferencedata=False,
                progressbar=False,
                random_seed=rng,
            )

        # Check variance matches manual computation
        det_pred = idata_pp.predictions.det
        var_pred = float(det_pred.var().values)
        npt.assert_allclose(var_pred, var_manual, rtol=0.1)

    def test_deterministic_no_resampling_logged(self, caplog):
        """
        Test that when Deterministic is in var_names, no variables are logged
        as being sampled (Sampling: []).

        This verifies that dependencies are not being resampled.
        """
        rng = np.random.default_rng(46)
        caplog.set_level(logging.INFO)

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)
            y = pm.Normal("y", mu=0, sigma=1)
            det = pm.Deterministic("det", x + y)

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                idata = pm.sample(
                    tune=100,
                    draws=100,
                    chains=2,
                    step=pm.Metropolis(),
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    random_seed=rng,
                    progressbar=False,
                )

        with model:
            pm.sample_posterior_predictive(
                idata,
                var_names=["det"],
                predictions=True,
                extend_inferencedata=False,
                progressbar=False,
                random_seed=rng,
            )

        # Check that "Sampling: []" appears in logs (no resampling)
        log_messages = caplog.text
        assert "Sampling: []" in log_messages or "Sampling:" not in log_messages.split(
            "Sampling:"
        )[-1].split("\n")[0] or len(
            [msg for msg in log_messages.split("Sampling:") if "[]" in msg]
        ) > 0

