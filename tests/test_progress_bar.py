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
from unittest.mock import MagicMock, patch

import pytest

import pymc as pm

from pymc.progress_bar import (
    ProgressBarManager,
    in_marimo_notebook,
)


def test_progressbar_nested_compound():
    # Regression test for https://github.com/pymc-devs/pymc/issues/7721

    with pm.Model():
        a = pm.Poisson("a", mu=10)
        b = pm.Binomial("b", n=a, p=0.8)
        c = pm.Poisson("c", mu=11)
        d = pm.Dirichlet("d", a=[c, b])

        step = pm.CompoundStep(
            [
                pm.CompoundStep([pm.Metropolis(a), pm.Metropolis(b), pm.Metropolis(c)]),
                pm.NUTS([d]),
            ]
        )

        kwargs = {
            "draws": 10,
            "tune": 10,
            "chains": 2,
            "compute_convergence_checks": False,
            "step": step,
        }

        # We don't parametrize to avoid recompiling the model functions
        for cores in (1, 2):
            pm.sample(**kwargs, cores=cores, progressbar=True)  # default is split+stats
            pm.sample(**kwargs, cores=cores, progressbar="combined")
            pm.sample(**kwargs, cores=cores, progressbar="split")
            pm.sample(**kwargs, cores=cores, progressbar=False)


class TestMarimoDetection:
    """Tests for marimo notebook environment detection."""

    def test_in_marimo_notebook_not_installed(self):
        """Test that in_marimo_notebook returns False when marimo is not installed."""
        with patch.dict("sys.modules", {"marimo": None}):
            # Force reimport by clearing from cache
            import importlib

            import pymc.progress_bar

            importlib.reload(pymc.progress_bar)
            # When marimo import fails, should return False
            assert in_marimo_notebook() is False

    def test_in_marimo_notebook_not_running(self):
        """Test that in_marimo_notebook returns False when not in a marimo notebook."""
        mock_marimo = MagicMock()
        mock_marimo.running_in_notebook.return_value = False

        with patch.dict("sys.modules", {"marimo": mock_marimo}):
            assert in_marimo_notebook() is False

    def test_in_marimo_notebook_running(self):
        """Test that in_marimo_notebook returns True when in a marimo notebook."""
        mock_marimo = MagicMock()
        mock_marimo.running_in_notebook.return_value = True

        with patch.dict("sys.modules", {"marimo": mock_marimo}):
            assert in_marimo_notebook() is True


class TestProgressBarManagerEnvironmentDetection:
    """Tests for ProgressBarManager environment detection."""

    @pytest.fixture
    def step_method(self):
        """Create a step method for testing."""
        with pm.Model():
            x = pm.Normal("x")
            step = pm.NUTS([x])
        return step

    def test_detects_non_marimo_environment(self, step_method):
        """Test that _is_marimo is False when not in marimo."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=False):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            assert manager._is_marimo is False

    def test_detects_marimo_environment(self, step_method):
        """Test that _is_marimo is True when in marimo."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            assert manager._is_marimo is True

    def test_disabled_progressbar_not_marimo(self, step_method):
        """Test that _is_marimo is False when progressbar=False regardless of environment."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=False,
            )
            # When progressbar is disabled, _is_marimo should be False
            # because we use the Rich backend (which handles disabled state)
            assert manager._is_marimo is False


class TestProgressBarManagerMarimoMode:
    """Tests for ProgressBarManager when running in marimo mode."""

    @pytest.fixture
    def step_method(self):
        """Create a step method for testing."""
        with pm.Model():
            x = pm.Normal("x")
            step = pm.NUTS([x])
        return step

    def test_init_split_mode(self, step_method):
        """Test initialization in split mode (default)."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            assert manager.combined_progress is False
            assert manager.full_stats is True
            assert manager._show_progress is True
            assert manager.chains == 2
            assert manager.total_draws == 150  # draws + tune
            assert manager._is_marimo is True

    def test_init_combined_mode(self, step_method):
        """Test initialization in combined mode."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
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

    def test_init_disabled(self, step_method):
        """Test initialization with progressbar disabled."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=False,
            )
            assert manager._show_progress is False

    def test_compute_draw_speed(self):
        """Test speed calculation."""
        speed, unit = ProgressBarManager.compute_draw_speed(elapsed=1.0, draws=100)
        assert speed == 100.0
        assert unit == "draws/s"

        speed, unit = ProgressBarManager.compute_draw_speed(elapsed=100.0, draws=1)
        assert speed == 100.0
        assert unit == "s/draw"

    def test_format_time(self):
        """Test time formatting."""
        assert ProgressBarManager._format_time(65) == "1:05"
        assert ProgressBarManager._format_time(3665) == "1:01:05"
        assert ProgressBarManager._format_time(0) == "0:00"

    def test_abbreviate_stat_name(self):
        """Test stat name abbreviation."""
        assert ProgressBarManager._abbreviate_stat_name("divergences") == "Div"
        assert ProgressBarManager._abbreviate_stat_name("step_size") == "Step"
        assert ProgressBarManager._abbreviate_stat_name("unknown_stat") == "Unknow"

    def test_render_html_structure(self, step_method):
        """Test that rendered HTML has expected structure."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
            manager = ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            # Initialize chain state manually for testing
            manager._chain_state = [
                {"draws": 50, "total": 150, "failing": False, "stats": {}},
                {"draws": 75, "total": 150, "failing": True, "stats": {"divergences": 1}},
            ]
            manager._start_times = [0, 0]

            html = manager._render_html()

            assert "pymc-progress-table" in html
            assert "pymc-progress-bar" in html
            # Chain 1 should have failing indicator
            assert "failing" in html
