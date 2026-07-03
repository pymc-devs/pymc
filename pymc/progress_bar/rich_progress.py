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

import warnings

from collections.abc import Iterable
from sys import stderr
from typing import Any, Self

from rich.box import SIMPLE_HEAD
from rich.console import Console, ConsoleOptions, RenderResult
from rich.progress import (
    BarColumn,
    Progress,
    ProgressBar,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.segment import Segment
from rich.style import Style
from rich.table import Column, Table
from rich.theme import Theme

default_progress_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "#1764f4",
        "bar.pulse": "grey23",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


class CustomProgress(Progress):
    def __init__(self, *args, include_headers: bool = False, **kwargs):
        self.include_headers = include_headers
        super().__init__(*args, **kwargs)

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        """Get a table to render the Progress display.

        Unlike the parent method, this returns a full table (not a grid),
        allowing for column headings.

        Parameters
        ----------
        tasks : Iterable[Task]
            An iterable of Task instances, one per row of the table.

        Returns
        -------
        Table
            A table instance.
        """

        def call_column(column, task):
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


class MarkerProgressBar(ProgressBar):
    """A progress bar with a thin gap at a given position (e.g. tune/draw boundary)."""

    def __init__(self, *args, marker_pos: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.marker_pos = marker_pos

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.marker_pos is None or not self.total:
            yield from super().__rich_console__(console, options)
            return

        width = min(self.width or options.max_width, options.max_width)
        # Map marker_pos (in task units) to a column index (in characters).
        marker_col = max(0, min(int(width * self.marker_pos / self.total), width - 1))

        # Rich yields ~3-5 bulk Segments (e.g. "━━━━━" for completed, "━━━" for
        # remaining). We iterate those and split only the one that spans the
        # marker column, replacing one character with a thin-space gap.
        pos = 0
        seg: Segment
        for seg in super().__rich_console__(console, options):  # type: ignore[assignment]
            seg_len = len(seg.text)
            seg_end = pos + seg_len
            if pos <= marker_col < seg_end:
                i = marker_col - pos
                if i > 0:
                    yield Segment(seg.text[:i], seg.style)
                yield Segment(" ", Style.null())  # thin space (U+2009)
                if i + 1 < seg_len:
                    yield Segment(seg.text[i + 1 :], seg.style)
            else:
                yield seg
            pos = seg_end


class CustomBarColumn(BarColumn):
    """Bar column that recolors on divergences and renders a separator marker."""

    def __init__(self, *args, failing_color: str = "red", **kwargs):
        from matplotlib.colors import to_rgb

        self.failing_color = failing_color
        self.failing_rgb = [int(x * 255) for x in to_rgb(self.failing_color)]

        super().__init__(*args, **kwargs)

        self.default_complete_style = self.complete_style
        self.default_finished_style = self.finished_style

    def callbacks(self, task: Task):
        """Update bar color based on failure state."""
        if task.fields["failing"]:
            self.complete_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
            self.finished_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
        else:
            self.complete_style = self.default_complete_style
            self.finished_style = self.default_finished_style

    def render(self, task: Task) -> ProgressBar:
        return MarkerProgressBar(
            total=max(0, task.total) if task.total is not None else None,
            completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started,
            animation_time=task.get_time(),
            style=self.style,
            complete_style=self.complete_style,
            finished_style=self.finished_style,
            pulse_style=self.pulse_style,
            marker_pos=task.fields.get("marker_pos"),
        )


class RichProgressBackend:
    """Rich-based progress bar backend for terminal rendering.

    This backend uses the Rich library to render progress bars in terminals
    and Jupyter notebooks.
    """

    def __init__(
        self,
        step_name: str,
        n_bars: int,
        total: int | float | None,
        combined: bool,
        full_stats: bool,
        progress_columns: list,
        progress_stats: dict[str, list[Any]],
        theme: Theme | None = None,
    ):
        """Initialize the Rich progress backend.

        Parameters
        ----------
        step_name : str
            Name of the unit of iteration (e.g., "draws", "particles")
        n_bars : int
            Number of progress bars to display
        total : int
            Total number of iterations per bar
        combined : bool
            Whether to show a single combined progress bar
        full_stats : bool
            Whether to show detailed statistics
        progress_columns : list
            Rich column definitions from step method
        progress_stats : dict
            Initial values for statistics by task
        theme : Theme, optional
            Rich theme for styling
        """
        self.step_name = step_name
        self.n_bars = n_bars
        self.initial_total = total
        self.combined = combined
        self.full_stats = full_stats
        self.progress_stats = progress_stats

        self._progress = self._create_progress_bar(
            progress_columns=progress_columns,
            theme=default_progress_theme if theme is None else theme,
        )
        self._tasks: list[TaskID | None] = []

    def _create_progress_bar(
        self,
        progress_columns: list,
        theme: Theme,
    ) -> CustomProgress:
        columns: list[ProgressColumn] = [
            TextColumn(
                "{" + f"task.fields[{self.step_name.lower()}]" + "}",
                table_column=Column(self.step_name.title(), ratio=1),
            )
        ]

        if self.full_stats:
            columns += progress_columns

        columns += [
            TextColumn(
                "{task.fields[sampling_speed]:0.2f} {task.fields[speed_unit]}",
                table_column=Column("Speed", ratio=2, no_wrap=True),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
            TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
        ]

        return CustomProgress(
            CustomBarColumn(
                bar_width=None,
                table_column=Column("Progress", ratio=2),
                failing_color="tab:red",
                complete_style=Style.parse("rgb(31,119,180)"),
                finished_style=Style.parse("rgb(31,119,180)"),
            ),
            *columns,
            console=Console(theme=theme),
            include_headers=True,
            expand=True,
        )

    def __enter__(self) -> Self:
        self._progress.__enter__()
        self._initialize_tasks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._progress.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_tasks(self) -> None:
        # ``start=False`` defers the per-task clock until ``start_task`` is
        # called on the first ``update``. With ``cores < chains``, later chains
        # would otherwise be timed against the first chain's start.
        self._tasks = [
            self._progress.add_task(
                "Sampling",
                start=False,
                completed=0,
                total=self.initial_total,
                task_idx=task_idx,
                sampling_speed=0,
                speed_unit="draws/s",
                failing=False,
                **{stat: value[task_idx] for stat, value in self.progress_stats.items()},
            )
            for task_idx in range(self.n_bars)
        ]
        self._started: list[bool] = [False] * self.n_bars

    def update(
        self,
        task_id: int,
        advance: int | float,
        failing: bool,
        stats: dict[str, Any],
        is_last: bool,
        total: int | None = None,
    ) -> None:
        """Update a progress bar.

        Parameters
        ----------
        task_id : int
            Index of the progress bar being updated
        advance : int or float
            Amount to advance the progress bar by
        failing : bool
            Whether the process has encountered failures
        stats : dict
            Statistics to display
        is_last : bool
            Whether this is the final update
        total : int, optional
            Updated total for the bar. If None, the existing total is kept —
            use this when the total is fixed at construction time. Pass an
            integer when the total is only known later (e.g. nutpie chooses
            its own tune count) or can change mid-run.
        """
        rich_task_id = self._tasks[task_id]
        if rich_task_id is None:
            return

        if not self._started[task_id]:
            self._progress.start_task(rich_task_id)
            self._started[task_id] = True

        if is_last:
            if total is None:
                total = self._progress.tasks[task_id].total  # type: ignore[assignment]
            self._progress.update(
                rich_task_id,
                completed=total,
                failing=failing,
                refresh=True,
                total=total,
                **stats,
            )
            return

        self._progress.advance(rich_task_id, advance=advance)

        task = self._progress.tasks[task_id]
        completed = task.completed
        elapsed = task.elapsed if task.elapsed is not None else 0.0

        action = self.step_name.lower()
        # Wait for a small window of elapsed time before computing speed so the
        # first reading isn't ``completed / ~0 ≈ infinity``.
        if elapsed > 0.25:
            speed = completed / elapsed
        else:
            speed = 0
        if speed > 1 or speed == 0:
            unit = f"{action}s/s"
        else:
            unit = f"s/{action}"
            speed = 1 / speed

        self._progress.update(
            rich_task_id,
            sampling_speed=speed,
            speed_unit=unit,
            failing=failing,
            total=total,
            **stats,
        )


def RichSimpleProgress(theme: Theme | None):
    return CustomProgress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("/"),
        TimeElapsedColumn(),
        console=Console(file=stderr, theme=default_progress_theme if theme is None else theme),
    )


def __getattr__(name: str):
    if name == "RecolorOnFailureBarColumn":
        warnings.warn(
            "RecolorOnFailureBarColumn has been renamed to CustomBarColumn.",
            DeprecationWarning,
            stacklevel=2,
        )
        return CustomBarColumn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
