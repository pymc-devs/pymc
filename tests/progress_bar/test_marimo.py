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

from unittest.mock import patch

import pytest

import pymc as pm

from pymc.progress_bar import MCMCProgressBarManager
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

            html = backend._render_html()

            assert "0.25" in html or "Step" in html
