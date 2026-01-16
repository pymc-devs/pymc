#   Copyright 2025 - present The PyMC Developers
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
"""Tests for marimo notebook detection and backend."""

from unittest.mock import patch

import pytest

import pymc as pm

from pymc.progress_bar import ProgressBarManager


class TestMarimoProgressBackend:
    """Tests for MarimoProgressBackend."""

    @pytest.fixture(autouse=True)
    def require_marimo(self):
        pytest.importorskip("marimo")

    @pytest.fixture
    def step_method(self):
        """Create a step method for testing."""
        with pm.Model():
            x = pm.Normal("x")
            step = pm.NUTS([x])
        return step

    def test_init_split_mode(self, step_method):
        """Test initialization in split mode (default)."""
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            assert manager._is_marimo is True
            assert type(manager._backend).__name__ == "MarimoProgressBackend"

    def test_init_combined_mode(self, step_method):
        """Test initialization in combined mode."""
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar="combined",
            )
            assert manager.combined_progress is True
            assert manager.full_stats is False
            assert manager._is_marimo is True

    def test_render_html_structure(self, step_method):
        """Test that rendered HTML has expected structure."""
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            # Access the marimo backend
            backend = manager._backend
            assert type(backend).__name__ == "MarimoProgressBackend"

            # Initialize chain state manually for testing
            backend._chain_state = [
                {"draws": 50, "total": 150, "failing": False, "stats": {}},
                {"draws": 75, "total": 150, "failing": True, "stats": {"divergences": 1}},
            ]
            backend._start_times = [0, 0]

            html = backend._render_html()

            assert "pymc-progress-table" in html
            assert "pymc-progress-bar" in html
            # Chain 1 should have failing indicator
            assert "failing" in html

    def test_render_html_with_stats(self, step_method):
        """Test HTML rendering includes stats when full_stats is True."""
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=1,
                draws=100,
                tune=50,
                progressbar="split+stats",
            )
            backend = manager._backend
            assert type(backend).__name__ == "MarimoProgressBackend"

            backend._chain_state = [
                {
                    "draws": 50,
                    "total": 150,
                    "failing": False,
                    "stats": {"step_size": 0.25, "divergences": 0},
                },
            ]
            backend._start_times = [0]

            html = backend._render_html()

            # Should include stat values
            assert "0.25" in html or "Step" in html

    def test_backend_is_enabled(self, step_method):
        """Test that marimo backend is always enabled."""
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=1,
                draws=100,
                tune=50,
                progressbar=True,
            )
            backend = manager._backend
            assert type(backend).__name__ == "MarimoProgressBackend"
            assert backend.is_enabled is True
