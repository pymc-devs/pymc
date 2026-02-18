import pytest
from unittest.mock import patch, MagicMock
import pymc as pm
import numpy as np

@pytest.fixture
def continuous_model():
    with pm.Model() as model:
        pm.Normal("x")
    return model

@pytest.fixture
def discrete_model():
    with pm.Model() as model:
        pm.Bernoulli("y", p=0.5)
    return model

def test_selects_nutpie_when_appropriate(continuous_model):
    """Test that nutpie is selected when installed and model is continuous."""
    with patch("pymc.sampling.mcmc._nutpie_is_installed", return_value=True), \
         patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
        
        pm.sample(model=continuous_model, tune=10, draws=10, chains=1)
        
        mock_sample_external.assert_called_once()
        assert mock_sample_external.call_args[1]["sampler"] == "nutpie"

def test_fallbacks_to_pymc_when_nutpie_missing(continuous_model):
    """Test fallback to PyMC when nutpie is not installed."""
    with patch("pymc.sampling.mcmc._nutpie_is_installed", return_value=False), \
         patch("pymc.sampling.mcmc.init_nuts") as mock_init_nuts:
        
        # We mock init_nuts to avoid actual sampling, which is slow/complex
        mock_init_nuts.return_value = ([], MagicMock()) 
        
        # We also need to mock _sample_external_nuts to ensure it's NOT called
        with patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
            try:
                pm.sample(model=continuous_model, tune=10, draws=10, chains=1)
            except Exception:
                pass
            
            mock_sample_external.assert_not_called()
            mock_init_nuts.assert_called()

def test_fallbacks_to_pymc_for_discrete_model(discrete_model):
    """Test fallback to PyMC when model has discrete variables."""
    with patch("pymc.sampling.mcmc._nutpie_is_installed", return_value=True), \
         patch("pymc.sampling.mcmc.init_nuts") as mock_init_nuts: # Mock init_nuts? No, discrete uses a different path usually 
                                                                    # Actually, discrete uses CompoundStep, not exclusive_nuts
        
        # For discrete, exclusive_nuts will be False, so it goes to the "else" block (lines 886+)
        # which calls instantiate_steppers, etc.
        # We just want to ensure _sample_external_nuts is NOT called.
         with patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
            # We might need to mock more to prevent actual sampling
            with patch("pymc.sampling.mcmc.instantiate_steppers"), \
                 patch("pymc.sampling.mcmc.init_traces"):
                try:
                    pm.sample(model=discrete_model, tune=10, draws=10, chains=1)
                except Exception:
                     # It might fail later in the pipeline, but we just check the sampler selection
                     pass
            
            mock_sample_external.assert_not_called()

def test_respects_explicit_pymc(continuous_model):
    """Test that explicit nuts_sampler='pymc' bypasses auto-selection."""
    with patch("pymc.sampling.mcmc._nutpie_is_installed", return_value=True), \
         patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
        
        # Should NOT call external sampler
        with patch("pymc.sampling.mcmc.init_nuts") as mock_init_nuts:
            mock_init_nuts.return_value = ([], MagicMock())
            try:
                pm.sample(model=continuous_model, nuts_sampler="pymc", tune=10, draws=10, chains=1)
            except Exception:
                pass
            mock_sample_external.assert_not_called()

def test_respects_explicit_nutpie(continuous_model):
    """Test that explicit nuts_sampler='nutpie' works even if conditions might be edge-case (though we can't easily force an edge case that isn't already handled by exclusive_nuts check in external path)."""
    with patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
        pm.sample(model=continuous_model, nuts_sampler="nutpie", tune=10, draws=10, chains=1)
        mock_sample_external.assert_called_once()
        assert mock_sample_external.call_args[1]["sampler"] == "nutpie"
