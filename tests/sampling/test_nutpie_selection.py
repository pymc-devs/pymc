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
        
        mock_init_nuts.return_value = ([], MagicMock()) 
        
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
         patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
         
         with patch("pymc.sampling.mcmc.instantiate_steppers"), \
              patch("pymc.sampling.mcmc.init_traces"):
                try:
                    pm.sample(model=discrete_model, tune=10, draws=10, chains=1)
                except Exception:
                     pass
            
                mock_sample_external.assert_not_called()

def test_respects_explicit_pymc(continuous_model):
    """Test that explicit nuts_sampler='pymc' bypasses auto-selection."""
    with patch("pymc.sampling.mcmc._nutpie_is_installed", return_value=True), \
         patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
        
        with patch("pymc.sampling.mcmc.init_nuts") as mock_init_nuts:
            mock_init_nuts.return_value = ([], MagicMock())
            try:
                pm.sample(model=continuous_model, nuts_sampler="pymc", tune=10, draws=10, chains=1)
            except Exception:
                pass
            mock_sample_external.assert_not_called()

def test_respects_explicit_nutpie(continuous_model):
    """Test that explicit nuts_sampler='nutpie' works."""
    with patch("pymc.sampling.mcmc._sample_external_nuts") as mock_sample_external:
        pm.sample(model=continuous_model, nuts_sampler="nutpie", tune=10, draws=10, chains=1)
        mock_sample_external.assert_called_once()
        assert mock_sample_external.call_args[1]["sampler"] == "nutpie"
