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
from collections.abc import Callable, Iterable
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from rich.box import SIMPLE_HEAD
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
from rich.table import Column, Table
from rich.theme import Theme

if TYPE_CHECKING:
    from pymc.step_methods.compound import BlockedStep, CompoundStep


def in_marimo_notebook() -> bool:
    """Check if running inside a marimo notebook."""
    try:
        import marimo as mo

        return mo.running_in_notebook()
    except (ImportError, AttributeError):
        return False


def _mo_write_internal(cell_id: str, value: object) -> None:
    """Write to marimo cell given cell_id.

    Adapted from marimo's internal write_internal function.
    """
    from marimo._messaging.cell_output import CellChannel
    from marimo._messaging.notification_utils import CellNotificationUtils
    from marimo._messaging.tracebacks import write_traceback
    from marimo._output import formatting

    output = formatting.try_format(value)
    if output.traceback is not None:
        write_traceback(output.traceback)
    CellNotificationUtils.broadcast_output(
        channel=CellChannel.OUTPUT,
        mimetype=output.mimetype,
        data=output.data,
        cell_id=cell_id,
        status=None,
    )


def _mo_create_replace() -> Callable[[object], None] | None:
    """Create mo.output.replace with current context pinned.

    This captures the cell context at creation time so that updates from
    callbacks work correctly even when called from different execution contexts.

    Adapted from marimo's output.replace implementation.
    """
    from marimo._output import formatting
    from marimo._runtime.context import get_context
    from marimo._runtime.context.types import ContextNotInitializedError

    try:
        ctx = get_context()
    except ContextNotInitializedError:
        return None

    if ctx.execution_context is None:
        return None

    cell_id = ctx.execution_context.cell_id
    execution_context = ctx.execution_context

    def replace(value: object) -> None:
        execution_context.output = [formatting.as_html(value)]
        _mo_write_internal(cell_id=cell_id, value=value)

    return replace


ProgressBarType = Literal[
    "combined",
    "split",
    "combined+stats",
    "stats+combined",
    "split+stats",
    "stats+split",
]
default_progress_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "green",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


class CustomProgress(Progress):
    """A child of Progress that allows to disable progress bars and its container.

    The implementation simply checks an `is_enabled` flag and generates the progress bar only if
    it's `True`.
    """

    def __init__(self, *args, disable=False, include_headers=False, **kwargs):
        self.is_enabled = not disable
        self.include_headers = include_headers

        if self.is_enabled:
            super().__init__(*args, **kwargs)

    def __enter__(self):
        """Enter the context manager."""
        if self.is_enabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self.is_enabled:
            super().__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, *args, **kwargs):
        if self.is_enabled:
            return super().add_task(*args, **kwargs)
        return None

    def advance(self, task_id, advance=1) -> None:
        if self.is_enabled:
            super().advance(task_id, advance)
        return None

    def update(
        self,
        task_id,
        *,
        total=None,
        completed=None,
        advance=None,
        description=None,
        visible=None,
        refresh=False,
        **fields,
    ):
        if self.is_enabled:
            super().update(
                task_id,
                total=total,
                completed=completed,
                advance=advance,
                description=description,
                visible=visible,
                refresh=refresh,
                **fields,
            )
        return None

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        """Get a table to render the Progress display.

        Unlike the parent method, this one returns a full table (not a grid), allowing for column headings.

        Parameters
        ----------
        tasks: Iterable[Task]
            An iterable of Task instances, one per row of the table.

        Returns
        -------
        table: Table
            A table instance.
        """

        def call_column(column, task):
            # Subclass rich.BarColumn and add a callback method to dynamically update the display
            if hasattr(column, "callbacks"):
                column.callbacks(task)

            return column(task)

        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )
        if self.include_headers:
            table = Table(
                *table_columns,
                padding=(0, 1),
                expand=self.expand,
                show_header=True,
                show_edge=True,
                box=SIMPLE_HEAD,
            )
        else:
            table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)

        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        (
                            column.format(task=task)
                            if isinstance(column, str)
                            else call_column(column, task)
                        )
                        for column in self.columns
                    )
                )

        return table


class RecolorOnFailureBarColumn(BarColumn):
    """Rich colorbar that changes color when a chain has detected a failure."""

    def __init__(self, *args, failing_color="red", **kwargs):
        from matplotlib.colors import to_rgb

        self.failing_color = failing_color
        self.failing_rgb = [int(x * 255) for x in to_rgb(self.failing_color)]

        super().__init__(*args, **kwargs)

        self.default_complete_style = self.complete_style
        self.default_finished_style = self.finished_style

    def callbacks(self, task: "Task"):
        if task.fields["failing"]:
            self.complete_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
            self.finished_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
        else:
            # Recovered from failing yay
            self.complete_style = self.default_complete_style
            self.finished_style = self.default_finished_style


class ProgressBarManager:
    """Manage progress bars displayed during sampling."""

    def __init__(
        self,
        step_method: "BlockedStep | CompoundStep",
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarType = True,
        progressbar_theme: Theme | None = None,
    ):
        """
        Manage progress bars displayed during sampling.

        When sampling, Step classes are responsible for computing and exposing statistics that can be reported on
        progress bars. Each Step implements two class methods: :meth:`pymc.step_methods.BlockedStep._progressbar_config`
        and :meth:`pymc.step_methods.BlockedStep._make_progressbar_update_functions`. `_progressbar_config` reports which
        columns should be displayed on the progress bar, and `_make_progressbar_update_functions` computes the statistics
        that will be displayed on the progress bar.

        Parameters
        ----------
        step_method: BlockedStep or CompoundStep
            The step method being used to sample
        chains: int
            Number of chains being sampled
        draws: int
            Number of draws per chain
        tune: int
            Number of tuning steps per chain
        progressbar: bool or ProgressType, optional
            How and whether to display the progress bar. If False, no progress bar is displayed. Otherwise, you can ask
            for one of the following:
            - "combined": A single progress bar that displays the total progress across all chains. Only timing
                information is shown.
            - "split": A separate progress bar for each chain. Only timing information is shown.
            - "combined+stats" or "stats+combined": A single progress bar displaying the total progress across all
                chains. Aggregate sample statistics are also displayed.
            - "split+stats" or "stats+split": A separate progress bar for each chain. Sample statistics for each chain
                are also displayed.

            If True, the default is "split+stats" is used.

        progressbar_theme: Theme, optional
            The theme to use for the progress bar. Defaults to the default theme.
        """
        if progressbar_theme is None:
            progressbar_theme = default_progress_theme

        match progressbar:
            case True:
                self.combined_progress = False
                self.full_stats = True
                show_progress = True
            case False:
                self.combined_progress = False
                self.full_stats = True
                show_progress = False
            case "combined":
                self.combined_progress = True
                self.full_stats = False
                show_progress = True
            case "split":
                self.combined_progress = False
                self.full_stats = False
                show_progress = True
            case "combined+stats" | "stats+combined":
                self.combined_progress = True
                self.full_stats = True
                show_progress = True
            case "split+stats" | "stats+split":
                self.combined_progress = False
                self.full_stats = True
                show_progress = True
            case _:
                raise ValueError(
                    "Invalid value for `progressbar`. Valid values are True (default), False (no progress bar), "
                    "one of 'combined', 'split', 'split+stats', or 'combined+stats."
                )

        progress_columns, progress_stats = step_method._progressbar_config(chains)

        self._progress = self.create_progress_bar(
            progress_columns,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )
        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self._show_progress = show_progress
        self.completed_draws = 0
        self.total_draws = draws + tune
        self.desc = "Sampling chain"
        self.chains = chains

        self._tasks: list[Task] | None = None  # type: ignore[annotation-unchecked]

    def __enter__(self):
        self._initialize_tasks()

        return self._progress.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._progress.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_tasks(self):
        if self.combined_progress:
            self.tasks = [
                self._progress.add_task(
                    self.desc.format(self),
                    completed=0,
                    draws=0,
                    total=self.total_draws * self.chains - 1,
                    chain_idx=0,
                    sampling_speed=0,
                    speed_unit="draws/s",
                    failing=False,
                    **{stat: value[0] for stat, value in self.progress_stats.items()},
                )
            ]

        else:
            self.tasks = [
                self._progress.add_task(
                    self.desc.format(self),
                    completed=0,
                    draws=0,
                    total=self.total_draws - 1,
                    chain_idx=chain_idx,
                    sampling_speed=0,
                    speed_unit="draws/s",
                    failing=False,
                    **{stat: value[chain_idx] for stat, value in self.progress_stats.items()},
                )
                for chain_idx in range(self.chains)
            ]

    @staticmethod
    def compute_draw_speed(elapsed, draws):
        speed = draws / max(elapsed, 1e-6)

        if speed > 1 or speed == 0:
            unit = "draws/s"
        else:
            unit = "s/draw"
            speed = 1 / speed

        return speed, unit

    def update(self, chain_idx, is_last, draw, tuning, stats):
        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        elapsed = self._progress.tasks[chain_idx].elapsed
        speed, unit = self.compute_draw_speed(elapsed, draw)

        failing = False
        all_step_stats = {}

        chain_progress_stats = [
            update_stats_fn(step_stats)
            for update_stats_fn, step_stats in zip(self.update_stats_functions, stats, strict=True)
        ]
        for step_stats in chain_progress_stats:
            for key, val in step_stats.items():
                if key == "failing":
                    failing |= val
                    continue
                if not self.full_stats:
                    # Only care about the "failing" flag
                    continue

                if key in all_step_stats:
                    # TODO: Figure out how to integrate duplicate / non-scalar keys, ignoring them for now
                    continue
                else:
                    all_step_stats[key] = val

        self._progress.update(
            self.tasks[chain_idx],
            completed=draw,
            draws=draw,
            sampling_speed=speed,
            speed_unit=unit,
            failing=failing,
            **all_step_stats,
        )

        if is_last:
            self._progress.update(
                self.tasks[chain_idx],
                draws=draw + 1 if not self.combined_progress else draw,
                failing=failing,
                **all_step_stats,
                refresh=True,
            )

    def create_progress_bar(self, step_columns, progressbar, progressbar_theme):
        columns = [TextColumn("{task.fields[draws]}", table_column=Column("Draws", ratio=1))]

        if self.full_stats:
            columns += step_columns

        columns += [
            TextColumn(
                "{task.fields[sampling_speed]:0.2f} {task.fields[speed_unit]}",
                table_column=Column("Sampling Speed", ratio=1),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
            TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
        ]

        return CustomProgress(
            RecolorOnFailureBarColumn(
                table_column=Column("Progress", ratio=2),
                failing_color="tab:red",
                complete_style=Style.parse("rgb(31,119,180)"),  # tab:blue
                finished_style=Style.parse("rgb(31,119,180)"),  # tab:blue
            ),
            *columns,
            console=Console(theme=progressbar_theme),
            disable=not progressbar,
            include_headers=True,
        )


# CSS styling for marimo progress bars
_MARIMO_PROGRESS_STYLE = """
<style>
.pymc-progress-container {
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
    font-size: 13px;
    line-height: 1.5;
    padding: 8px 0;
}
.pymc-progress-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0;
    flex-wrap: nowrap;
}
.pymc-chain-label {
    min-width: 70px;
    font-weight: 500;
}
.pymc-progress-bar-container {
    width: 150px;
    height: 16px;
    background-color: #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
    flex-shrink: 0;
}
.pymc-progress-bar {
    height: 100%;
    background-color: #1f77b4;
    transition: width 0.1s ease-out;
}
.pymc-progress-bar.failing {
    background-color: #d62728;
}
.pymc-progress-bar.finished {
    background-color: #2ca02c;
}
.pymc-progress-text {
    min-width: 120px;
    white-space: nowrap;
}
.pymc-stats {
    display: flex;
    gap: 12px;
    flex-wrap: nowrap;
}
.pymc-stat {
    white-space: nowrap;
}
.pymc-stat-label {
    color: #666;
}
.pymc-timing {
    white-space: nowrap;
    color: #666;
}
.pymc-warning {
    color: #d62728;
    margin-left: 4px;
}
@media (prefers-color-scheme: dark) {
    .pymc-progress-bar-container {
        background-color: #444;
    }
    .pymc-stat-label, .pymc-timing {
        color: #aaa;
    }
}
</style>
"""


class MarimoProgressBarManager:
    """Progress bar manager for marimo notebooks using HTML rendering.

    This class provides the same interface as ProgressBarManager but renders
    progress using HTML/CSS via marimo's output system instead of Rich's
    terminal-based rendering.
    """

    def __init__(
        self,
        step_method: "BlockedStep | CompoundStep",
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarType = True,
        progressbar_theme: Theme | None = None,  # Unused, kept for API compatibility
    ):
        """Initialize the marimo progress bar manager.

        Parameters are the same as ProgressBarManager for API compatibility.
        """
        # Pin marimo context immediately (before any work starts)
        self._mo_replace = _mo_create_replace()

        match progressbar:
            case True:
                self.combined_progress = False
                self.full_stats = True
                self._show_progress = True
            case False:
                self.combined_progress = False
                self.full_stats = True
                self._show_progress = False
            case "combined":
                self.combined_progress = True
                self.full_stats = False
                self._show_progress = True
            case "split":
                self.combined_progress = False
                self.full_stats = False
                self._show_progress = True
            case "combined+stats" | "stats+combined":
                self.combined_progress = True
                self.full_stats = True
                self._show_progress = True
            case "split+stats" | "stats+split":
                self.combined_progress = False
                self.full_stats = True
                self._show_progress = True
            case _:
                raise ValueError(
                    "Invalid value for `progressbar`. Valid values are True (default), False (no progress bar), "
                    "one of 'combined', 'split', 'split+stats', or 'combined+stats."
                )

        # Get stat column names from step method config
        _, progress_stats = step_method._progressbar_config(chains)
        self.progress_stats = progress_stats
        self.stat_names = list(progress_stats.keys())
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self.chains = chains
        self.total_draws = draws + tune
        self.completed_draws = 0

        # Per-chain state
        self._chain_state: list[dict[str, Any]] = []
        self._start_times: list[float] = []

    def __enter__(self):
        """Enter the context manager and initialize chain state."""
        if not self._show_progress:
            return self

        import marimo as mo

        mo.output.clear()

        # Initialize per-chain state
        if self.combined_progress:
            self._chain_state = [
                {
                    "draws": 0,
                    "total": self.total_draws * self.chains,
                    "failing": False,
                    "stats": {},
                }
            ]
            self._start_times = [perf_counter()]
        else:
            self._chain_state = [
                {
                    "draws": 0,
                    "total": self.total_draws,
                    "failing": False,
                    "stats": {},
                }
                for _ in range(self.chains)
            ]
            self._start_times = [perf_counter() for _ in range(self.chains)]

        self._render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager with final render."""
        if self._show_progress:
            self._render()
        return False

    @staticmethod
    def compute_draw_speed(elapsed: float, draws: int) -> tuple[float, str]:
        """Compute sampling speed and appropriate unit."""
        speed = draws / max(elapsed, 1e-6)

        if speed > 1 or speed == 0:
            unit = "draws/s"
        else:
            unit = "s/draw"
            speed = 1 / speed

        return speed, unit

    def update(self, chain_idx: int, is_last: bool, draw: int, tuning: bool, stats: list) -> None:
        """Update progress bar with new sampling statistics."""
        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        # Extract stats from step methods
        failing = False
        all_step_stats: dict[str, Any] = {}

        chain_progress_stats = [
            update_stats_fn(step_stats)
            for update_stats_fn, step_stats in zip(self.update_stats_functions, stats, strict=True)
        ]
        for step_stats in chain_progress_stats:
            for key, val in step_stats.items():
                if key == "failing":
                    failing |= val
                    continue
                if not self.full_stats:
                    continue
                if key not in all_step_stats:
                    all_step_stats[key] = val

        # Update chain state
        self._chain_state[chain_idx]["draws"] = draw
        self._chain_state[chain_idx]["failing"] = failing
        self._chain_state[chain_idx]["stats"] = all_step_stats

        if is_last:
            self._chain_state[chain_idx]["draws"] = draw + 1 if not self.combined_progress else draw

        self._render()

    def _render(self) -> None:
        """Render HTML progress display to marimo output."""
        if self._mo_replace is None:
            return

        import marimo as mo

        html = self._render_html()
        self._mo_replace(mo.Html(html))

    def _render_html(self) -> str:
        """Generate HTML for all progress bars."""
        rows = []
        for i, state in enumerate(self._chain_state):
            rows.append(self._render_chain_row(i, state))

        rows_html = "\n".join(rows)
        return f"{_MARIMO_PROGRESS_STYLE}\n<div class='pymc-progress-container'>{rows_html}</div>"

    def _render_chain_row(self, chain_idx: int, state: dict[str, Any]) -> str:
        """Render a single chain's progress row."""
        draws = state["draws"]
        total = state["total"]
        failing = state["failing"]
        stats = state["stats"]

        # Calculate progress
        pct = (draws / total * 100) if total > 0 else 0
        elapsed = perf_counter() - self._start_times[chain_idx]
        speed, unit = self.compute_draw_speed(elapsed, draws)

        # Format elapsed time
        elapsed_str = self._format_time(elapsed)

        # Determine bar class
        bar_class = "pymc-progress-bar"
        if failing:
            bar_class += " failing"
        elif pct >= 100:
            bar_class += " finished"

        # Chain label
        if self.combined_progress:
            label = "All chains"
        else:
            label = f"Chain {chain_idx}"

        # Warning indicator
        warning = '<span class="pymc-warning">⚠</span>' if failing else ""

        # Stats display
        stats_html = ""
        if self.full_stats and stats:
            stat_items = []
            for key, val in stats.items():
                if isinstance(val, float):
                    formatted = f"{val:.3f}"
                else:
                    formatted = str(val)
                # Abbreviate stat names
                short_key = self._abbreviate_stat_name(key)
                stat_items.append(
                    f'<span class="pymc-stat"><span class="pymc-stat-label">{short_key}:</span> {formatted}</span>'
                )
            stats_html = f'<div class="pymc-stats">{" ".join(stat_items)}</div>'

        return f"""
        <div class="pymc-progress-row">
            <span class="pymc-chain-label">{label}{warning}</span>
            <div class="pymc-progress-bar-container">
                <div class="{bar_class}" style="width: {pct:.1f}%"></div>
            </div>
            <span class="pymc-progress-text">{draws}/{total} ({pct:.0f}%)</span>
            {stats_html}
            <span class="pymc-timing">{speed:.1f} {unit} | {elapsed_str}</span>
        </div>
        """

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format elapsed time as mm:ss or hh:mm:ss."""
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    @staticmethod
    def _abbreviate_stat_name(name: str) -> str:
        """Abbreviate common statistic names for compact display."""
        abbreviations = {
            "divergences": "Div",
            "diverging": "Div",
            "step_size": "Step",
            "tree_size": "Tree",
            "tree_depth": "Depth",
            "n_steps": "Steps",
            "energy_error": "E-err",
            "max_energy_error": "Max-E",
            "mean_tree_accept": "Accept",
            "scaling": "Scale",
            "tune": "Tune",
        }
        return abbreviations.get(name, name[:6].capitalize())


def create_progress_bar_manager(
    step_method: "BlockedStep | CompoundStep",
    chains: int,
    draws: int,
    tune: int,
    progressbar: bool | ProgressBarType = True,
    progressbar_theme: Theme | None = None,
) -> ProgressBarManager | MarimoProgressBarManager:
    """Create the appropriate progress bar manager for the current environment.

    Automatically detects marimo notebooks and returns a MarimoProgressBarManager,
    otherwise returns the standard Rich-based ProgressBarManager.

    Parameters
    ----------
    step_method : BlockedStep or CompoundStep
        The step method being used to sample
    chains : int
        Number of chains being sampled
    draws : int
        Number of draws per chain
    tune : int
        Number of tuning steps per chain
    progressbar : bool or ProgressBarType, optional
        How and whether to display the progress bar
    progressbar_theme : Theme, optional
        The theme to use for the progress bar (Rich backend only)

    Returns
    -------
    ProgressBarManager or MarimoProgressBarManager
        The appropriate progress bar manager for the environment
    """
    if progressbar and in_marimo_notebook():
        return MarimoProgressBarManager(
            step_method=step_method,
            chains=chains,
            draws=draws,
            tune=tune,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )
    return ProgressBarManager(
        step_method=step_method,
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        progressbar_theme=progressbar_theme,
    )
