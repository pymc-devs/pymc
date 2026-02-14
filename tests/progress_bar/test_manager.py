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

from pymc.progress_bar import MCMCProgressBarManager


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
        manager = MCMCProgressBarManager(
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
        manager = MCMCProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar="combined",
        )
        assert manager.combined_progress is True
        assert manager.full_stats is False

    def test_init_combined_stats_mode(self, step_method):
        manager = MCMCProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar="combined+stats",
        )
        assert manager.combined_progress is True
        assert manager.full_stats is True

    def test_init_split_stats_mode(self, step_method):
        manager = MCMCProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar="split+stats",
        )
        assert manager.combined_progress is False
        assert manager.full_stats is True

    def test_init_disabled(self, step_method):
        manager = MCMCProgressBarManager(
            step_method=step_method,
            chains=2,
            draws=100,
            tune=50,
            progressbar=False,
        )
        assert manager._show_progress is False

    def test_invalid_progressbar_value(self, step_method):
        with pytest.raises(ValueError, match="Invalid value for `progressbar`"):
            MCMCProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar="invalid",
            )
