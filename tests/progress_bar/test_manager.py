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
"""Tests for ProgressBarManager and integration tests."""

from unittest.mock import patch

import pytest

import pymc as pm

from pymc.progress_bar import ProgressBarManager
from pymc.progress_bar.progress import (
    abbreviate_stat_name,
    compute_draw_speed,
    format_time,
)


def test_progressbar_nested_compound():
    """Regression test for https://github.com/pymc-devs/pymc/issues/7721"""
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


class TestProgressBarManagerEnvironmentDetection:
    """Tests for ProgressBarManager environment detection."""

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

    def test_detects_non_marimo_environment(self, step_method):
        """Test that _is_marimo is False when not in marimo."""
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=False):
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
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
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
        with patch("pymc.progress_bar.marimo_progress.in_marimo_notebook", return_value=True):
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


class TestProgressBarManagerConfiguration:
    """Tests for ProgressBarManager configuration parsing."""

    @pytest.fixture
    def step_method(self):
        """Create a step method for testing."""
        with pm.Model():
            x = pm.Normal("x")
            step = pm.NUTS([x])
        return step

    def test_init_split_mode(self, step_method):
        """Test initialization in split mode (default)."""
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

    def test_init_combined_mode(self, step_method):
        """Test initialization in combined mode."""
        manager = ProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar="combined",
        )
        assert manager.combined_progress is True
        assert manager.full_stats is False

    def test_init_combined_stats_mode(self, step_method):
        """Test initialization in combined+stats mode."""
        manager = ProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar="combined+stats",
        )
        assert manager.combined_progress is True
        assert manager.full_stats is True

    def test_init_split_stats_mode(self, step_method):
        """Test initialization in split+stats mode."""
        manager = ProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar="split+stats",
        )
        assert manager.combined_progress is False
        assert manager.full_stats is True

    def test_init_disabled(self, step_method):
        """Test initialization with progressbar disabled."""
        manager = ProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar=False,
        )
        assert manager._show_progress is False

    def test_invalid_progressbar_value(self, step_method):
        """Test that invalid progressbar value raises error."""
        with pytest.raises(ValueError, match="Invalid value for `progressbar`"):
            ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar="invalid",
            )


class TestUtilityFunctions:
    """Tests for shared utility functions."""

    def test_compute_draw_speed_fast(self):
        """Test speed calculation for fast sampling."""
        speed, unit = compute_draw_speed(elapsed=1.0, draws=100)
        assert speed == 100.0
        assert unit == "draws/s"

    def test_compute_draw_speed_slow(self):
        """Test speed calculation for slow sampling."""
        speed, unit = compute_draw_speed(elapsed=100.0, draws=1)
        assert speed == 100.0
        assert unit == "s/draw"

    def test_compute_draw_speed_zero_draws(self):
        """Test speed calculation with zero draws."""
        speed, unit = compute_draw_speed(elapsed=1.0, draws=0)
        assert speed == 0.0
        assert unit == "draws/s"

    def test_format_time_seconds(self):
        """Test time formatting for seconds."""
        assert format_time(45) == "0:45"

    def test_format_time_minutes(self):
        """Test time formatting for minutes."""
        assert format_time(65) == "1:05"
        assert format_time(125) == "2:05"

    def test_format_time_hours(self):
        """Test time formatting for hours."""
        assert format_time(3665) == "1:01:05"

    def test_format_time_zero(self):
        """Test time formatting for zero."""
        assert format_time(0) == "0:00"

    def test_abbreviate_stat_name_known(self):
        """Test abbreviation of known stat names."""
        assert abbreviate_stat_name("divergences") == "Div"
        assert abbreviate_stat_name("step_size") == "Step"
        assert abbreviate_stat_name("tree_depth") == "Depth"
        assert abbreviate_stat_name("mean_tree_accept") == "Accept"

    def test_abbreviate_stat_name_unknown(self):
        """Test abbreviation of unknown stat names."""
        assert abbreviate_stat_name("unknown_stat") == "Unknow"
        assert abbreviate_stat_name("xy") == "Xy"
