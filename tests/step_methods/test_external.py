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

from pymc import Model, Normal, sample
from pymc.step_methods.external import NutPie
from pymc.step_methods.external.nutpie import NUTPIE_AVAILABLE


@pytest.mark.skipif(not NUTPIE_AVAILABLE, reason="NutPie not installed")
def test_nutpie_integration():
    """Test basic usage of NutPie as a PyMC step method."""
    with Model() as model:
        x = Normal("x", mu=0, sigma=1)

        # Create NutPie sampler with numba backend
        nutpie_sampler = NutPie(backend="numba")

        # NutPie sampler has the is_external attribute set to True

        # Sample using external sampler
        trace = sample(
            draws=10,  # Use fewer draws for faster testing
            tune=10,
            step=nutpie_sampler,
            chains=1,  # Use just one chain for simplicity
            random_seed=42,
            progressbar=False,
        )

    # Check that the sampling worked
    assert "x" in trace.posterior
    assert trace.posterior.x.shape == (1, 10)

    # Check that the sampler stats were recorded
    expected_stats = ["diverging", "energy"]
    for stat in expected_stats:
        assert stat in trace.sample_stats


@pytest.mark.skipif(not NUTPIE_AVAILABLE, reason="NutPie not installed")
def test_nutpie_jax_backend():
    """Test NutPie with JAX backend."""
    try:
        import importlib.util

        jax_available = importlib.util.find_spec("jax") is not None
    except ImportError:
        jax_available = False

    if not jax_available:
        pytest.skip("JAX not installed")

    with Model() as model:
        x = Normal("x", mu=0, sigma=1)

        # Create NutPie sampler with JAX backend
        nutpie_sampler = NutPie(backend="jax")

        # Sample using external sampler
        trace = sample(
            draws=10,
            tune=10,
            step=nutpie_sampler,
            chains=1,
            random_seed=42,
            progressbar=False,
        )

    # Check that the sampling worked
    assert "x" in trace.posterior
    assert trace.posterior.x.shape == (1, 10)


@pytest.mark.skipif(not NUTPIE_AVAILABLE, reason="NutPie not installed")
def test_nutpie_custom_params():
    """Test NutPie with custom parameters."""
    with Model() as model:
        x = Normal("x", mu=0, sigma=1)

        # Create NutPie sampler with custom parameters
        nutpie_sampler = NutPie(
            backend="numba",
            target_accept=0.9,
            max_treedepth=8,
        )

        # Sample using external sampler
        trace = sample(
            draws=10,
            tune=10,
            step=nutpie_sampler,
            chains=1,
            random_seed=42,
            progressbar=False,
        )

    # Check that the sampling worked
    assert "x" in trace.posterior
    assert trace.posterior.x.shape == (1, 10)
