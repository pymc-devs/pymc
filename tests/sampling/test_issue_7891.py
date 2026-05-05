"""
Test for issue #7891: Pre-sampling validation of dims/coords consistency
https://github.com/pymc-devs/pymc/issues/7891
"""

import pytest

import pymc as pm


class TestPreSamplingValidation:
    """Test that shape mismatches are caught before sampling begins."""

    def test_dims_coords_shape_mismatch_caught_early(self):
        """Test that shape mismatch is caught before sampling.

        This is the main test case from issue #7891. A deterministic
        variable declares dims=['other'] with 4 elements, but its
        actual shape comes from dim_1 with 3 elements.
        """
        coords = {'dim_1': ['a', 'b', 'c'], 'other': [1, 2, 3, 4]}
        with pm.Model(coords=coords):
            x = pm.Normal('x', mu=0, sigma=1, dims=['dim_1'])
            # Shape mismatch: mu has shape (3,) from x but declares
            # dims=['other'] with length 4
            pm.Deterministic('mu', 1 + x, dims=['other'])
            pm.HalfNormal('sigma', 1)
            pm.Normal('y_obs', mu=x, sigma=1, dims=['dim_1'])

            # This should raise a ValueError during pre-sampling
            # validation
            match_text = "Pre-sampling validation failed"
            with pytest.raises(ValueError, match=match_text):
                pm.sample(draws=10, tune=10, chains=1, random_seed=42)

    def test_valid_model_samples_successfully(self):
        """Test that a model with correct dims/coords samples."""
        with pm.Model(coords={'dim_1': ['a', 'b', 'c']}):
            x = pm.Normal('x', mu=0, sigma=1, dims=['dim_1'])
            mu = pm.Deterministic('mu', 1 + x, dims=['dim_1'])
            pm.HalfNormal('sigma', 1)
            pm.Normal('y_obs', mu=mu, sigma=1, dims=['dim_1'])

            # This should succeed
            idata = pm.sample(
                draws=10,
                tune=10,
                chains=1,
                random_seed=42,
                progressbar=False,
                compute_convergence_checks=False,
            )

            # Verify we got valid output
            assert 'posterior' in idata.groups()
            assert 'x' in idata.posterior.data_vars
            # (chains, draws, dim_1)
            assert idata.posterior['x'].shape == (1, 10, 3)

    def test_validation_skipped_when_not_creating_idata(self):
        """Test validation is skipped when return_inferencedata=False."""
        # When return_inferencedata=False, the validation shouldn't
        # run because the user won't be creating InferenceData
        coords = {'dim_1': ['a', 'b', 'c'], 'other': [1, 2, 3, 4]}
        with pm.Model(coords=coords):
            x = pm.Normal('x', mu=0, sigma=1, dims=['dim_1'])
            pm.Deterministic('mu', 1 + x, dims=['other'])
            pm.HalfNormal('sigma', 1)
            pm.Normal('y_obs', mu=x, sigma=1, dims=['dim_1'])

            # With return_inferencedata=False, validation should be
            # skipped (though sampling itself might still fail)
            try:
                pm.sample(
                    draws=10,
                    tune=10,
                    chains=1,
                    random_seed=42,
                    return_inferencedata=False,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
            except ValueError as e:
                # If it fails, it shouldn't be our validation
                assert "Pre-sampling validation failed" not in str(e)

    def test_scalar_variables_work_correctly(self):
        """Test that scalar variables (no dims) work correctly."""
        with pm.Model():
            x = pm.Normal('x', mu=0, sigma=1)
            mu = pm.Deterministic('mu', 1 + x)
            pm.HalfNormal('sigma', 1)
            pm.Normal('y_obs', mu=mu, sigma=1)

            # This should succeed
            idata = pm.sample(
                draws=10,
                tune=10,
                chains=1,
                random_seed=42,
                progressbar=False,
                compute_convergence_checks=False,
            )

            assert 'posterior' in idata.groups()
            assert 'x' in idata.posterior.data_vars
            assert idata.posterior['x'].shape == (1, 10)

    def test_multiple_dims_validation(self):
        """Test validation with multiple dimensions."""
        coords = {'dim_1': [1, 2], 'dim_2': ['a', 'b', 'c']}
        with pm.Model(coords=coords):
            # Create a 2D variable with correct dims
            x = pm.Normal('x', mu=0, sigma=1, dims=['dim_1', 'dim_2'])

            # Create a deterministic with wrong dims order
            # x has shape (2, 3) with dims=['dim_1', 'dim_2']
            # If we declare dims=['dim_2', 'dim_1'], we'd be saying
            # shape is (3, 2)
            pm.Deterministic('y', x, dims=['dim_2', 'dim_1'])

            # This should catch the mismatch
            match_text = "Pre-sampling validation failed"
            with pytest.raises(ValueError, match=match_text):
                pm.sample(draws=10, tune=10, chains=1, random_seed=42)
