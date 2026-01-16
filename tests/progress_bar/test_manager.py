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

        for cores in (1, 2):
            pm.sample(**kwargs, cores=cores, progressbar=True)
            pm.sample(**kwargs, cores=cores, progressbar="combined")
            pm.sample(**kwargs, cores=cores, progressbar="split")
            pm.sample(**kwargs, cores=cores, progressbar=False)


class TestProgressBarManagerConfiguration:
    @pytest.fixture
    def step_method(self):
        with pm.Model():
            x = pm.Normal("x")
            step = pm.NUTS([x])
        return step

    def test_init_split_mode(self, step_method):
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
        assert manager.total_draws == 150

    def test_init_combined_mode(self, step_method):
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
        manager = ProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar=False,
        )
        assert manager._show_progress is False

    def test_invalid_progressbar_value(self, step_method):
        with pytest.raises(ValueError, match="Invalid value for `progressbar`"):
            ProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar="invalid",
            )


class TestUtilityFunctions:
    def test_compute_draw_speed_fast(self):
        speed, unit = compute_draw_speed(elapsed=1.0, draws=100)
        assert speed == 100.0
        assert unit == "draws/s"

    def test_compute_draw_speed_slow(self):
        speed, unit = compute_draw_speed(elapsed=100.0, draws=1)
        assert speed == 100.0
        assert unit == "s/draw"

    def test_compute_draw_speed_zero_draws(self):
        speed, unit = compute_draw_speed(elapsed=1.0, draws=0)
        assert speed == 0.0
        assert unit == "draws/s"

    def test_format_time_seconds(self):
        assert format_time(45) == "0:45"

    def test_format_time_minutes(self):
        assert format_time(65) == "1:05"
        assert format_time(125) == "2:05"

    def test_format_time_hours(self):
        assert format_time(3665) == "1:01:05"

    def test_format_time_zero(self):
        assert format_time(0) == "0:00"

    def test_abbreviate_stat_name_known(self):
        assert abbreviate_stat_name("divergences") == "Div"
        assert abbreviate_stat_name("step_size") == "Step"
        assert abbreviate_stat_name("tree_depth") == "Depth"
        assert abbreviate_stat_name("mean_tree_accept") == "Accept"

    def test_abbreviate_stat_name_unknown(self):
        assert abbreviate_stat_name("unknown_stat") == "Unknow"
        assert abbreviate_stat_name("xy") == "Xy"
