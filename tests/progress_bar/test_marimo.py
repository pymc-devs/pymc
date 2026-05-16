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

from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pymc as pm

from pymc.progress_bar import MCMCProgressBarManager, NutpieProgressBarManager
from pymc.progress_bar.marimo_progress import MarimoProgressBackend


class TestMarimoProgressBackend:
    @pytest.fixture(autouse=True)
    def require_marimo(self):
        pytest.importorskip("marimo")

    @pytest.fixture
    def step_method(self):
        with pm.Model():
            x = pm.Normal("x")
            step = pm.NUTS([x])
        return step

    def test_init_split_mode(self, step_method):
        with patch("pymc.progress_bar.progress.in_marimo_notebook", return_value=True):
            manager = MCMCProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            assert isinstance(manager._backend, MarimoProgressBackend)

    def test_init_combined_mode(self, step_method):
        with patch("pymc.progress_bar.progress.in_marimo_notebook", return_value=True):
            manager = MCMCProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar="combined",
            )
            assert isinstance(manager._backend, MarimoProgressBackend)
            assert manager.combined_progress is True
            assert manager.full_stats is False

    def test_render_html_structure(self, step_method):
        with patch("pymc.progress_bar.progress.in_marimo_notebook", return_value=True):
            manager = MCMCProgressBarManager(
                step_method=step_method,
                chains=2,
                draws=100,
                tune=50,
                progressbar=True,
            )
            backend = manager._backend
            assert isinstance(manager._backend, MarimoProgressBackend)

            backend._chain_state = [
                {"completed": 50, "total": 150, "failing": False, "stats": {}},
                {"completed": 75, "total": 150, "failing": True, "stats": {"divergences": 1}},
            ]
            backend._start_times = [0, 0]
            backend._end_times = [None, None]

            html = backend._render_html()

            assert "pymc-progress-table" in html
            assert "pymc-progress-bar" in html
            assert "failing" in html

    def test_render_html_with_stats(self, step_method):
        with patch("pymc.progress_bar.progress.in_marimo_notebook", return_value=True):
            manager = MCMCProgressBarManager(
                step_method=step_method,
                chains=1,
                draws=100,
                tune=50,
                progressbar="split+stats",
            )
            backend = manager._backend
            assert isinstance(manager._backend, MarimoProgressBackend)

            backend._task_state = [
                {
                    "completed": 50,
                    "total": 150,
                    "failing": False,
                    "stats": {"step_size": 0.25, "divergences": 0},
                },
            ]
            backend._start_times = [0]
            backend._end_times = [None]

            html = backend._render_html()

            assert "0.25" in html or "Step" in html

    def test_is_last_sets_completed_to_total(self):
        backend = MarimoProgressBackend(
            step_name="Draws", n_bars=2, total=150, combined=False, full_stats=False
        )
        backend._initialize_tasks()

        for _ in range(150):
            backend.update(task_id=0, advance=1, failing=False, stats={}, is_last=False)
        backend.update(task_id=0, advance=1, failing=False, stats={}, is_last=True)

        assert backend._task_state[0]["completed"] == 150
        assert backend._task_state[1]["completed"] == 0

    def test_elapsed_freezes_after_completion(self):
        """Completed chains must not show drifting speed/elapsed on re-renders."""
        backend = MarimoProgressBackend(
            step_name="Draw", n_bars=2, total=10, combined=False, full_stats=False
        )
        backend._initialize_tasks()

        # Complete chain 0
        for i in range(10):
            backend.update(task_id=0, advance=1, failing=False, stats={}, is_last=i == 9)

        html_at_finish = backend._render_task_row(0, backend._task_state[0], [])

        # Let wall-clock advance so unfrozen elapsed would visibly drift
        sleep(0.3)

        html_after = backend._render_task_row(0, backend._task_state[0], [])
        assert html_at_finish == html_after

    def test_marimo_smc_progress(self):
        backend = MarimoProgressBackend(
            step_name="Stage", n_bars=1, total=1.0, combined=False, full_stats=False
        )
        backend._initialize_tasks()

        assert backend._task_state[0]["total"] == 1.0
        assert backend._task_state[0]["completed"] == 0

        betas = [0.3, 0.7, 1.0]
        old = 0.0
        for beta in betas:
            backend.update(
                task_id=0, advance=beta - old, failing=False, stats={}, is_last=beta >= 1.0
            )
            old = beta

        assert backend._task_state[0]["completed"] == 1.0

    def test_nutpie_elapsed_freezes_after_completion(self):
        """Completed nutpie chains must not show drifting elapsed in marimo."""
        with patch("pymc.progress_bar.progress.in_marimo_notebook", return_value=True):
            manager = NutpieProgressBarManager(chains=2, draws=100, progressbar=True)
        assert isinstance(manager._backend, MarimoProgressBackend)
        total = 1100

        backend = manager._backend
        backend._initialize_tasks()

        def cp(finished, runtime_ms, started=True):
            return SimpleNamespace(
                finished_draws=finished,
                total_draws=total,
                runtime_ms=runtime_ms,
                started=started,
                divergent_draws=[],
                step_size=0.5,
                latest_num_steps=7,
            )

        # Chain 0 finishes, chain 1 halfway
        manager.update([cp(total, runtime_ms=5000), cp(500, runtime_ms=2500)])

        html_at_finish = backend._render_task_row(0, backend._task_state[0], [])

        # Let wall-clock advance so unfrozen elapsed would visibly drift
        sleep(0.3)

        # More callbacks arrive while chain 1 is still running
        manager.update([cp(total, runtime_ms=5000), cp(800, runtime_ms=4000)])

        # Chain 0's row must be identical — elapsed and speed frozen
        html_after = backend._render_task_row(0, backend._task_state[0], [])
        assert html_at_finish == html_after
