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
    """Write to marimo cell given cell_id."""
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
    """Manage progress bars displayed during sampling.

    This class automatically detects the execution environment and uses the
    appropriate rendering backend:
    - Rich-based terminal rendering for standard Python/Jupyter environments
    - HTML rendering for marimo notebooks

    The `_is_marimo` attribute indicates which backend is active.
    """

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

        self._show_progress = show_progress
        self._is_marimo = in_marimo_notebook()

        # Get progress bar config from step method
        progress_columns, progress_stats = step_method._progressbar_config(chains)
        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self.completed_draws = 0
        self.total_draws = draws + tune
        self.desc = "Sampling chain"
        self.chains = chains

        if self._is_marimo:
            # Marimo-specific initialization
            self._mo_replace = _mo_create_replace()
            self._chain_state: list[dict[str, Any]] = []
            self._start_times: list[float] = []
        else:
            # Rich-specific initialization
            self._progress = self.create_progress_bar(
                progress_columns,
                progressbar=progressbar,
                progressbar_theme=progressbar_theme,
            )
            self.tasks: list[Task] | None = None  # type: ignore[annotation-unchecked]

    def __enter__(self):
        self._initialize_tasks()

        if self._is_marimo:
            if self._show_progress:
                import marimo as mo

                mo.output.clear()
                self._render()
            return self
        else:
            return self._progress.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_marimo:
            if self._show_progress:
                self._render()
            return False
        else:
            return self._progress.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_tasks(self):
        """Initialize progress tracking for all chains."""
        if self._is_marimo:
            self._initialize_tasks_marimo()
        else:
            self._initialize_tasks_rich()

    def _initialize_tasks_rich(self):
        """Initialize Rich progress bar tasks."""
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

    def _initialize_tasks_marimo(self):
        """Initialize marimo chain state."""
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

    @staticmethod
    def compute_draw_speed(elapsed, draws):
        """Compute sampling speed and appropriate unit."""
        speed = draws / max(elapsed, 1e-6)

        if speed > 1 or speed == 0:
            unit = "draws/s"
        else:
            unit = "s/draw"
            speed = 1 / speed

        return speed, unit

    def _extract_stats(self, stats) -> tuple[bool, dict[str, Any]]:
        """Extract and process stats from step methods.

        Returns
        -------
        failing : bool
            Whether any step method reported a failure condition
        all_step_stats : dict
            Aggregated statistics from all step methods
        """
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
                    # Only care about the "failing" flag
                    continue
                if key in all_step_stats:
                    # TODO: Figure out how to integrate duplicate / non-scalar keys, ignoring them for now
                    continue
                else:
                    all_step_stats[key] = val

        return failing, all_step_stats

    def update(self, chain_idx, is_last, draw, tuning, stats):
        """Update progress bar with new sampling statistics."""
        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        # Extract stats (shared between both backends)
        failing, all_step_stats = self._extract_stats(stats)

        # Dispatch to appropriate backend
        if self._is_marimo:
            self._update_marimo(chain_idx, is_last, draw, failing, all_step_stats)
        else:
            self._update_rich(chain_idx, is_last, draw, failing, all_step_stats)

    def _update_rich(self, chain_idx, is_last, draw, failing, all_step_stats):
        """Update Rich progress bar."""
        elapsed = self._progress.tasks[chain_idx].elapsed
        speed, unit = self.compute_draw_speed(elapsed, draw)

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

    def _update_marimo(self, chain_idx, is_last, draw, failing, all_step_stats):
        """Update marimo HTML progress."""
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
        """Generate HTML for all progress bars as a table with headers."""
        # Build header row - get stat columns from first chain's state
        stat_keys = []
        if self.full_stats and self._chain_state and self._chain_state[0]["stats"]:
            stat_keys = list(self._chain_state[0]["stats"].keys())

        header_cells = ["Progress", "Draws"]
        header_cells += [self._abbreviate_stat_name(k) for k in stat_keys]
        header_cells += ["Speed", "Elapsed"]

        header_row = "<tr>" + "".join(f"<th>{h}</th>" for h in header_cells) + "</tr>"

        # Build data rows
        data_rows = []
        for i, state in enumerate(self._chain_state):
            data_rows.append(self._render_chain_row(i, state, stat_keys))

        rows_html = "\n".join(data_rows)
        return f"{_MARIMO_PROGRESS_STYLE}\n<table class='pymc-progress-table'><thead>{header_row}</thead><tbody>{rows_html}</tbody></table>"

    def _render_chain_row(self, chain_idx: int, state: dict[str, Any], stat_keys: list[str]) -> str:
        """Render a single chain's progress as a table row."""
        draws = state["draws"]
        total = state["total"]
        failing = state["failing"]
        stats = state["stats"]

        # Calculate progress
        pct = (draws / total * 100) if total > 0 else 0
        elapsed = perf_counter() - self._start_times[chain_idx]
        speed, unit = self.compute_draw_speed(elapsed, draws)

        # Determine bar class
        bar_class = "pymc-progress-bar"
        if failing:
            bar_class += " failing"
        elif pct >= 100:
            bar_class += " finished"

        # Build cells
        cells = [
            f'<td><div class="pymc-progress-bar-container"><div class="{bar_class}" style="width: {pct:.1f}%"></div></div></td>',
            f"<td>{draws}/{total} ({pct:.0f}%)</td>",
        ]

        # Add stat cells in consistent order
        for key in stat_keys:
            val = stats.get(key, "")
            if isinstance(val, float):
                cells.append(f"<td>{val:.3f}</td>")
            else:
                cells.append(f"<td>{val}</td>")

        # Speed and elapsed
        cells.append(f"<td>{speed:.1f} {unit}</td>")
        cells.append(f"<td>{self._format_time(elapsed)}</td>")

        return "<tr>" + "".join(cells) + "</tr>"

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
.pymc-progress-table {
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
    font-size: 13px;
    border-collapse: collapse;
}
.pymc-progress-table th {
    text-align: left;
    padding: 4px 10px;
    font-weight: 500;
    color: #666;
    border-bottom: 1px solid #ddd;
}
.pymc-progress-table td {
    padding: 6px 10px;
    white-space: nowrap;
}
.pymc-progress-bar-container {
    width: 120px;
    height: 14px;
    background-color: #e0e0e0;
    border-radius: 3px;
    overflow: hidden;
}
.pymc-progress-bar {
    height: 100%;
    background-color: #1f77b4;
    transition: width 0.1s ease-out;
}
.pymc-progress-bar.failing { background-color: #d62728; }
.pymc-progress-bar.finished { background-color: #2ca02c; }
@media (prefers-color-scheme: dark) {
    .pymc-progress-table th { color: #aaa; border-bottom-color: #444; }
    .pymc-progress-bar-container { background-color: #444; }
}
</style>
"""


class MarimoSimpleProgress:
    """Simple marimo-aware progress bar for forward sampling functions.

    This provides a similar interface to CustomProgress but renders using
    HTML in marimo notebooks for better display.
    """

    def __init__(self, description: str, total: int, disable: bool = False):
        """Initialize the simple progress bar.

        Parameters
        ----------
        description : str
            Description text to show with the progress bar
        total : int
            Total number of steps
        disable : bool
            If True, disable the progress bar
        """
        self.description = description
        self.total = total
        self.completed = 0
        self.is_enabled = not disable

        self._mo_replace: Callable[[object], None] | None = None
        self._start_time: float = 0.0
        self._task_id = 0  # Dummy task ID for interface compatibility

    def __enter__(self):
        """Enter the context manager."""
        if not self.is_enabled:
            return self

        self._mo_replace = _mo_create_replace()
        self._start_time = perf_counter()

        import marimo as mo

        mo.output.clear()
        self._render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager with final render."""
        if self.is_enabled:
            self._render()
        return False

    def add_task(
        self, description: str, completed: int = 0, total: int | None = None, **kwargs
    ) -> int:
        """Add a task (interface compatibility with CustomProgress).

        Parameters are accepted but mostly ignored since MarimoSimpleProgress
        only supports a single task initialized at construction.
        """
        if not self.is_enabled:
            return self._task_id
        self.description = description
        self.completed = completed
        if total is not None:
            self.total = total
        self._render()
        return self._task_id

    def advance(self, task_id: int | None = None, advance: int = 1) -> None:
        """Advance the progress bar.

        Parameters
        ----------
        task_id : int, optional
            Ignored (for interface compatibility)
        advance : int
            Amount to advance by
        """
        if not self.is_enabled:
            return
        self.completed += advance
        self._render()

    def update(
        self,
        task_id: int | None = None,
        completed: int | None = None,
        refresh: bool = False,
        **kwargs,
    ) -> None:
        """Update the progress bar state.

        Parameters
        ----------
        task_id : int, optional
            Ignored (for interface compatibility)
        completed : int, optional
            Set completed count
        refresh : bool
            If True, force a render
        **kwargs
            Additional arguments ignored for compatibility
        """
        if not self.is_enabled:
            return
        if completed is not None:
            self.completed = completed
        if refresh:
            self._render()

    def _render(self) -> None:
        """Render HTML progress to marimo output."""
        if self._mo_replace is None:
            return

        import marimo as mo

        html = self._render_html()
        self._mo_replace(mo.Html(html))

    def _render_html(self) -> str:
        """Generate HTML for the progress bar as a table."""
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        elapsed = perf_counter() - self._start_time

        # Calculate speed
        if elapsed > 0 and self.completed > 0:
            speed = self.completed / elapsed
            if speed > 1:
                speed_str = f"{speed:.1f} samples/s"
            else:
                speed_str = f"{1 / speed:.1f} s/sample"
        else:
            speed_str = "-- samples/s"

        # Estimate remaining time
        if self.completed > 0 and self.completed < self.total:
            remaining = (self.total - self.completed) / (self.completed / elapsed)
            remaining_str = ProgressBarManager._format_time(remaining)
        else:
            remaining_str = "--:--"

        elapsed_str = ProgressBarManager._format_time(elapsed)

        bar_class = "pymc-progress-bar"
        if pct >= 100:
            bar_class += " finished"

        return f"""{_MARIMO_PROGRESS_STYLE}
<table class="pymc-progress-table">
<thead><tr><th>Progress</th><th>Samples</th><th>Speed</th><th>Elapsed</th><th>Remaining</th></tr></thead>
<tbody><tr>
<td><div class="pymc-progress-bar-container"><div class="{bar_class}" style="width: {pct:.1f}%"></div></div></td>
<td>{self.completed}/{self.total} ({pct:.0f}%)</td>
<td>{speed_str}</td>
<td>{elapsed_str}</td>
<td>{remaining_str}</td>
</tr></tbody>
</table>"""


def create_simple_progress(
    description: str,
    total: int,
    progressbar: bool = True,
    progressbar_theme: Theme | None = None,
) -> CustomProgress | MarimoSimpleProgress:
    """Create a simple progress bar appropriate for the current environment.

    Automatically detects marimo notebooks and returns a MarimoSimpleProgress,
    otherwise returns a standard Rich-based CustomProgress.

    This is intended for use by forward sampling functions like
    sample_posterior_predictive and sample_prior_predictive.

    Parameters
    ----------
    description : str
        Description text for the progress bar
    total : int
        Total number of samples/steps
    progressbar : bool
        Whether to show the progress bar
    progressbar_theme : Theme, optional
        Theme for the Rich progress bar (non-marimo only)

    Returns
    -------
    CustomProgress or MarimoSimpleProgress
        The appropriate progress bar for the environment
    """
    if progressbar and in_marimo_notebook():
        return MarimoSimpleProgress(
            description=description,
            total=total,
            disable=not progressbar,
        )

    if progressbar_theme is None:
        progressbar_theme = default_progress_theme

    return CustomProgress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("/"),
        TimeElapsedColumn(),
        console=Console(theme=progressbar_theme),
        disable=not progressbar,
    )
