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
"""Progress bar utilities for PyMC sampling.

This module provides progress bar functionality that works across different environments:
- Terminal: Rich-based progress bars
- Jupyter notebooks: HTML-based progress with IPython display
- Marimo notebooks: HTML-based progress with marimo output

The implementation is modeled after nutpie's progress bar approach
"""

from __future__ import annotations

import time

from typing import TYPE_CHECKING, Any, Literal

from jinja2 import Template
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Column

if TYPE_CHECKING:
    from pymc.step_methods.compound import BlockedStep, CompoundStep

ProgressBarType = Literal[
    "combined",
    "split",
    "combined+stats",
    "stats+combined",
    "split+stats",
    "stats+split",
]


# =============================================================================
# Environment Detection
# =============================================================================


def in_notebook() -> bool:
    """Check if code is running in a Jupyter notebook.

    Adapted from fastprogress.
    """

    def in_colab():
        """Check if the code is running in Google Colaboratory."""
        try:
            from google import colab  # noqa: F401

            return True
        except ImportError:
            return False

    if in_colab():
        return True
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == "ZMQInteractiveShell":  # Jupyter notebook, Spyder or qtconsole
            try:
                from IPython.display import (
                    HTML,  # noqa: F401
                    clear_output,  # noqa: F401
                    display,  # noqa: F401
                )

                return True
            except ImportError:
                import warnings

                warnings.warn(
                    "Couldn't import ipywidgets properly, progress bar will be disabled",
                    stacklevel=2,
                )
                return False
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def in_marimo_notebook() -> bool:
    """Check if code is running in a marimo notebook."""
    try:
        import marimo as mo

        return mo.running_in_notebook()
    except ImportError:
        return False


# =============================================================================
# Marimo-specific output functions
# =============================================================================


def _mo_write_internal(cell_id, stream, value: object) -> None:
    """Write to marimo cell given cell_id and stream."""
    from marimo._messaging.cell_output import CellChannel
    from marimo._messaging.ops import CellOp
    from marimo._messaging.tracebacks import write_traceback
    from marimo._output import formatting

    output = formatting.try_format(value)
    if output.traceback is not None:
        write_traceback(output.traceback)
    CellOp.broadcast_output(
        channel=CellChannel.OUTPUT,
        mimetype=output.mimetype,
        data=output.data,
        cell_id=cell_id,
        status=None,
        stream=stream,
    )


def _mo_create_replace():
    """Create mo.output.replace with current context pinned.

    This captures the cell context at creation time, allowing updates
    from background threads to correctly target the originating cell.
    """
    from marimo._output import formatting
    from marimo._runtime.context import get_context
    from marimo._runtime.context.types import ContextNotInitializedError

    try:
        ctx = get_context()
    except ContextNotInitializedError:
        return None

    cell_id = ctx.execution_context.cell_id
    execution_context = ctx.execution_context
    stream = ctx.stream

    def replace(value):
        execution_context.output = [formatting.as_html(value)]
        _mo_write_internal(cell_id=cell_id, value=value, stream=stream)

    return replace


# =============================================================================
# HTML Templates and Styles (modeled after nutpie)
# =============================================================================

_PROGRESS_STYLE = """
<style>
    .pymc-progress {
        max-width: 900px;
        margin: 10px auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
        font-size: 14px;
    }
    .pymc-progress table {
        width: 100%;
        border-collapse: collapse;
    }
    .pymc-progress th, .pymc-progress td {
        padding: 8px 10px;
        text-align: left;
        border-bottom: 1px solid #888;
    }
    .pymc-progress progress {
        width: 100%;
        height: 15px;
        border-radius: 5px;
    }
    progress::-webkit-progress-bar {
        background-color: #eee;
        border-radius: 5px;
    }
    progress::-webkit-progress-value {
        background-color: #1764f4;
        border-radius: 5px;
    }
    progress::-moz-progress-bar {
        background-color: #1764f4;
        border-radius: 5px;
    }
    .pymc-progress .progress-cell {
        width: 100%;
    }
    .pymc-progress p strong { font-size: 16px; font-weight: bold; }
    .pymc-progress .failing progress::-webkit-progress-value {
        background-color: #d9534f;
    }
    .pymc-progress .failing progress::-moz-progress-bar {
        background-color: #d9534f;
    }
    .pymc-progress .finished progress::-webkit-progress-value {
        background-color: #5cb85c;
    }
    .pymc-progress .finished progress::-moz-progress-bar {
        background-color: #5cb85c;
    }

    @media (prefers-color-scheme: dark) {
        .pymc-progress {
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .pymc-progress table, .pymc-progress th, .pymc-progress td {
            border-color: #555;
            color: #ccc;
        }
        .pymc-progress th {
            background-color: #2a2a2a;
        }
        .pymc-progress progress::-webkit-progress-bar {
            background-color: #444;
        }
        .pymc-progress progress::-webkit-progress-value {
            background-color: #3178c6;
        }
        .pymc-progress progress::-moz-progress-bar {
            background-color: #3178c6;
        }
    }
</style>
"""

# Jinja2 template for progress bar HTML
_PROGRESS_TEMPLATE = Template("""
<div class="pymc-progress">
    <p><strong>Sampler Progress</strong></p>
    <p>Total Chains: {{ num_chains }} | Active: {{ running_chains }} | Finished: {{ finished_chains }}</p>
    <p>Sampling for {{ time_sampling }}{% if time_remaining %} | ETA: {{ time_remaining }}{% endif %}</p>

    <progress max="{{ total_draws }}" value="{{ total_finished_draws }}"></progress>

    <table>
        <thead>
            <tr>
                <th style="width: 30%;">Progress</th>
                <th>Draws</th>
                {% for col in stat_columns %}
                <th>{{ col.header }}</th>
                {% endfor %}
                <th>Speed</th>
            </tr>
        </thead>
        <tbody>
            {% for chain in chains %}
            <tr>
                <td class="progress-cell {{ chain.state_class }}">
                    <progress max="{{ chain.total_draws }}" value="{{ chain.finished_draws }}"></progress>
                </td>
                <td>{{ chain.finished_draws }}/{{ chain.total_draws }}</td>
                {% for stat in chain.stats %}
                <td>{{ stat }}</td>
                {% endfor %}
                <td>{{ chain.speed }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
""")


# =============================================================================
# Rich Terminal Progress Bar Components
# =============================================================================


class RecolorOnFailureBarColumn(BarColumn):
    """Rich colorbar that changes color when a chain has detected a failure."""

    def __init__(self, *args, failing_color="red", **kwargs):
        from matplotlib.colors import to_rgb

        self.failing_color = failing_color
        self.failing_rgb = [int(x * 255) for x in to_rgb(self.failing_color)]

        super().__init__(*args, **kwargs)

        self.default_complete_style = self.complete_style
        self.default_finished_style = self.finished_style

    def callbacks(self, task: Task):
        if task.fields.get("failing", False):
            self.complete_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
            self.finished_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
        else:
            self.complete_style = self.default_complete_style
            self.finished_style = self.default_finished_style


class TerminalProgress(Progress):
    """Rich Progress subclass that supports column headers and dynamic styling."""

    def __init__(self, *args, disable: bool = False, **kwargs):
        self.is_enabled = not disable
        if self.is_enabled:
            super().__init__(*args, **kwargs)

    def __enter__(self):
        if self.is_enabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_enabled:
            super().__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, *args, **kwargs):  # type: ignore[override]
        if self.is_enabled:
            return super().add_task(*args, **kwargs)
        return None

    def advance(self, task_id, advance: int = 1) -> None:
        if self.is_enabled and task_id is not None:
            super().advance(task_id, advance)

    def update(self, task_id, **kwargs) -> None:
        if self.is_enabled and task_id is not None:
            super().update(task_id, **kwargs)

    def get_renderables(self):
        """Override to call callbacks before rendering."""
        if not self.is_enabled:
            return
        for task in self.tasks:
            if task.visible:
                for column in self.columns:
                    if hasattr(column, "callbacks"):
                        column.callbacks(task)
        yield from super().get_renderables()


class ProgressBarManager:
    """Manage progress bars displayed during sampling.

    This class handles progress bar display across different environments:
    - Terminal: Rich-based progress bars
    - Jupyter: HTML-based with IPython display
    - Marimo: HTML-based with marimo output

    The implementation matches nutpie's approach for marimo compatibility.
    """

    def __init__(
        self,
        step_method: BlockedStep | CompoundStep,
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarType = True,
        progressbar_theme=None,  # Kept for backward compatibility, but unused in HTML mode
    ):
        """Initialize the progress bar manager.

        Parameters
        ----------
        step_method : BlockedStep or CompoundStep
            The step method being used to sample
        chains : int
            Number of chains being sampled
        draws : int
            Number of draws per chain (excluding tune)
        tune : int
            Number of tuning steps per chain
        progressbar : bool or ProgressBarType
            Whether and how to display the progress bar.
        progressbar_theme : optional
            Theme for Rich progress bars (terminal only).
        """
        self.show_progress, self.combined_progress, self.full_stats = self._parse_progressbar_arg(
            progressbar
        )

        self.chains = chains
        self.total_draws_per_chain = draws + tune
        self.total_draws = self.total_draws_per_chain * chains
        self.tune = tune
        self.draws = draws

        self.stat_columns, self.progress_stats = step_method._progressbar_config(chains)
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self._chain_states: list[dict[str, Any]] = [
            {
                "finished_draws": 0,
                "total_draws": self.total_draws_per_chain,
                "failing": False,
                "finished": False,
                "stats": {k: v[i] for k, v in self.progress_stats.items()},
                "speed": 0.0,
                "speed_unit": "draws/s",
            }
            for i in range(chains)
        ]

        self._start_time: float | None = None
        self._total_finished_draws = 0

        self._display_id = None
        self._mo_replace = None
        self._terminal_progress: TerminalProgress | None = None

        if not self.show_progress:
            self._mode = "none"
        elif in_marimo_notebook():
            self._mode = "marimo"
            self._mo_replace = _mo_create_replace()
        elif in_notebook():
            self._mode = "jupyter"
        else:
            self._mode = "terminal"
            self._terminal_progress = self._create_terminal_progress()

    def _parse_progressbar_arg(
        self, progressbar: bool | ProgressBarType
    ) -> tuple[bool, bool, bool]:
        """Parse the progressbar argument.

        Returns
        -------
        show_progress : bool
            Whether to show progress at all
        combined_progress : bool
            Whether to show combined progress (single bar) or split (per-chain)
        full_stats : bool
            Whether to show full stats or just timing
        """
        match progressbar:
            case True:
                return True, False, True
            case False:
                return False, False, True
            case "combined":
                return True, True, False
            case "split":
                return True, False, False
            case "combined+stats" | "stats+combined":
                return True, True, True
            case "split+stats" | "stats+split":
                return True, False, True
            case _:
                raise ValueError(
                    f"Invalid value for `progressbar`: {progressbar}. "
                    "Valid values are True, False, 'combined', 'split', "
                    "'combined+stats', 'stats+combined', 'split+stats', 'stats+split'."
                )

    def _create_terminal_progress(self) -> TerminalProgress:
        """Create a Rich-based terminal progress bar."""
        # Build columns: Progress bar, Draws, [Stats...], Speed, Elapsed, Remaining
        columns = [
            RecolorOnFailureBarColumn(
                table_column=Column(header="Progress", ratio=2),
                complete_style=Style.parse("rgb(31,119,180)"),  # Blue
                finished_style=Style.parse("rgb(44,160,44)"),  # Green
            ),
            TextColumn(
                "{task.fields[draws]}/{task.fields[draws_total]}",
                table_column=Column(header="Draws", ratio=1),
            ),
        ]

        if self.full_stats and self.stat_columns:
            columns.extend(self.stat_columns)

        columns.extend(
            [
                TextColumn(
                    "{task.fields[speed]:.2f} {task.fields[speed_unit]}",
                    table_column=Column(header="Speed", ratio=1),
                ),
                TimeElapsedColumn(table_column=Column(header="Elapsed", ratio=1)),
                TimeRemainingColumn(table_column=Column(header="Remaining", ratio=1)),
            ]
        )

        return TerminalProgress(
            *columns,
            console=Console(),
            disable=not self.show_progress,
        )

    def __enter__(self):
        """Enter the progress bar context."""
        self._start_time = time.time()

        if self._mode == "jupyter":
            import IPython.display

            IPython.display.display(IPython.display.HTML(_PROGRESS_STYLE))
            self._display_id = IPython.display.display(
                IPython.display.HTML(self._render_html()), display_id=True
            )
        elif self._mode == "marimo":
            import marimo as mo

            if self._mo_replace:
                self._mo_replace(mo.Html(f"{_PROGRESS_STYLE}\n{self._render_html()}"))
        elif self._mode == "terminal" and self._terminal_progress:
            self._terminal_progress.__enter__()
            self._init_terminal_tasks()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the progress bar context."""
        # Final update to show completion state
        if self._mode in ("jupyter", "marimo"):
            self._update_html()

        if self._mode == "terminal" and self._terminal_progress:
            self._terminal_progress.__exit__(exc_type, exc_val, exc_tb)

    def _init_terminal_tasks(self):
        """Initialize Rich progress tasks for terminal display."""
        if not self._terminal_progress:
            return

        self._tasks = []
        if self.combined_progress:
            # Single combined task
            task_id = self._terminal_progress.add_task(
                "Sampling",
                total=self.total_draws - 1,
                draws=0,
                draws_total=self.total_draws,
                speed=0.0,
                speed_unit="draws/s",
                failing=False,
                **{k: v[0] for k, v in self.progress_stats.items()},
            )
            self._tasks = [task_id]
        else:
            # Per-chain tasks
            for i in range(self.chains):
                task_id = self._terminal_progress.add_task(
                    f"Chain {i}",
                    total=self.total_draws_per_chain - 1,
                    draws=0,
                    draws_total=self.total_draws_per_chain,
                    speed=0.0,
                    speed_unit="draws/s",
                    failing=False,
                    **{k: v[i] for k, v in self.progress_stats.items()},
                )
                self._tasks.append(task_id)

    def update(
        self,
        chain_idx: int,
        is_last: bool,
        draw: int,
        tuning: bool,
        stats: list[dict[str, Any]],
    ):
        """Update the progress bar with new draw information.

        Parameters
        ----------
        chain_idx : int
            Index of the chain being updated
        is_last : bool
            Whether this is the last draw for the chain
        draw : int
            Current draw number (0-indexed)
        tuning : bool
            Whether the chain is still tuning
        stats : list of dict
            Statistics from the step method
        """
        if not self.show_progress:
            return

        self._total_finished_draws += 1
        chain_state = self._chain_states[chain_idx]
        chain_state["finished_draws"] = draw + 1
        chain_state["finished"] = is_last

        elapsed = time.time() - self._start_time if self._start_time else 1e-6
        speed, speed_unit = self._compute_speed(elapsed, draw + 1)
        chain_state["speed"] = speed
        chain_state["speed_unit"] = speed_unit

        failing = False
        all_step_stats = {}
        for update_fn, step_stats in zip(self.update_stats_functions, stats, strict=True):
            step_stat_update = update_fn(step_stats)
            for key, val in step_stat_update.items():
                if key == "failing":
                    failing |= val
                elif self.full_stats:
                    if key not in all_step_stats:
                        all_step_stats[key] = val

        chain_state["failing"] = failing
        chain_state["stats"].update(all_step_stats)

        if self._mode == "terminal":
            self._update_terminal(chain_idx, draw, is_last, failing, all_step_stats)
        elif self._mode in ("jupyter", "marimo"):
            self._update_html()

    def _compute_speed(self, elapsed: float, draws: int) -> tuple[float, str]:
        """Compute sampling speed."""
        speed = draws / max(elapsed, 1e-6)
        if speed > 1 or speed == 0:
            return speed, "draws/s"
        else:
            return 1 / speed, "s/draw"

    def _update_terminal(
        self,
        chain_idx: int,
        draw: int,
        is_last: bool,
        failing: bool,
        stats: dict[str, Any],
    ):
        """Update Rich terminal progress."""
        if not self._terminal_progress or not self._tasks:
            return

        chain_state = self._chain_states[chain_idx]

        if self.combined_progress:
            task_idx = 0
            completed = self._total_finished_draws
        else:
            task_idx = chain_idx
            completed = draw + 1

        self._terminal_progress.update(
            self._tasks[task_idx],
            completed=completed,
            draws=chain_state["finished_draws"],
            draws_total=chain_state["total_draws"],
            speed=chain_state["speed"],
            speed_unit=chain_state["speed_unit"],
            failing=failing,
            refresh=is_last,
            **stats,
        )

    def _update_html(self):
        """Update HTML progress display (Jupyter or Marimo)."""
        html = self._render_html()

        if self._mode == "jupyter" and self._display_id:
            import IPython.display

            self._display_id.update(IPython.display.HTML(html))
        elif self._mode == "marimo" and self._mo_replace:
            import marimo as mo

            self._mo_replace(mo.Html(f"{_PROGRESS_STYLE}\n{html}"))

    def _render_html(self) -> str:
        """Render the HTML progress display."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        if self._total_finished_draws > 0:
            rate = self._total_finished_draws / max(elapsed, 1e-6)
            remaining_draws = self.total_draws - self._total_finished_draws
            eta_seconds = remaining_draws / rate if rate > 0 else 0
            time_remaining = self._format_time(eta_seconds)
        else:
            time_remaining = None

        finished_chains = sum(1 for c in self._chain_states if c["finished"])
        running_chains = self.chains - finished_chains

        stat_column_headers = []
        if self.full_stats and self.stat_columns:
            for col in self.stat_columns:
                # Extract header from TextColumn's get_table_column() method
                table_col = col.get_table_column()
                if table_col and table_col.header:
                    stat_column_headers.append({"header": table_col.header})

        chains_data = []
        for i, chain_state in enumerate(self._chain_states):
            if chain_state["finished"]:
                state_class = "finished"
            elif chain_state["failing"]:
                state_class = "failing"
            else:
                state_class = ""

            stat_values = []
            if self.full_stats:
                for col in self.stat_columns:
                    # Try to extract the field name and format from the TextColumn
                    stat_values.append(self._format_stat_for_column(col, chain_state["stats"]))

            speed = chain_state["speed"]
            speed_unit = chain_state["speed_unit"]
            speed_str = f"{speed:.2f} {speed_unit}"

            chains_data.append(
                {
                    "finished_draws": chain_state["finished_draws"],
                    "total_draws": chain_state["total_draws"],
                    "state_class": state_class,
                    "stats": stat_values,
                    "speed": speed_str,
                }
            )

        return _PROGRESS_TEMPLATE.render(
            num_chains=self.chains,
            running_chains=running_chains,
            finished_chains=finished_chains,
            time_sampling=self._format_time(elapsed),
            time_remaining=time_remaining,
            total_draws=self.total_draws,
            total_finished_draws=self._total_finished_draws,
            stat_columns=stat_column_headers,
            chains=chains_data,
        )

    def _format_stat_for_column(self, column: TextColumn, stats: dict[str, Any]) -> str:
        """Format a stat value based on the column definition."""
        # The TextColumn has a format string like "{task.fields[divergences]}"
        text_format = column.text_format

        # Extract field names from format string
        # Pattern: {task.fields[name]} or {task.fields[name]:format}
        import re

        matches = re.findall(r"\{task\.fields\[(\w+)\](?::([^}]*))?\}", text_format)

        if not matches:
            return ""

        field_name, fmt = matches[0]
        value = stats.get(field_name, "")

        if fmt and value != "":
            try:
                return f"{value:{fmt}}"
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# Keep the old CustomProgress class for backward compatibility with SMC
class CustomProgress(Progress):
    """A child of Progress that allows to disable progress bars and its container.

    Kept for backward compatibility with SMC sampling.
    """

    def __init__(self, *args, disable=False, include_headers=False, **kwargs):
        self.is_enabled = not disable
        self.include_headers = include_headers

        if self.is_enabled:
            super().__init__(*args, **kwargs)

    def __enter__(self):
        if self.is_enabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_enabled:
            super().__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, *args, **kwargs):
        if self.is_enabled:
            return super().add_task(*args, **kwargs)
        return None  # type: ignore[return-value]

    def advance(self, task_id, advance=1) -> None:
        if self.is_enabled and task_id is not None:
            super().advance(task_id, advance)

    def update(self, task_id, **kwargs):
        if self.is_enabled and task_id is not None:
            super().update(task_id, **kwargs)


# Export default theme for backward compatibility
default_progress_theme = None  # No longer used, kept for import compatibility
