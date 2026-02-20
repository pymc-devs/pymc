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

from collections.abc import Iterable
from typing import Any, Self

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Column, Table
from rich.theme import Theme

default_progress_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "#1764f4",
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


class RecolorOnFailureBarColumn(BarColumn):
    """Rich colorbar that changes color when a chain has detected a failure."""

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


class RichProgressBackend:
    """Rich-based progress bar backend for terminal rendering.

    This backend uses the Rich library to render progress bars in terminals
    and Jupyter notebooks.
    """

    def __init__(
        self,
        step_name: str,
        n_bars: int,
        total: int | float,
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
        self.total = total
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
                table_column=Column("Sampling Speed", ratio=1),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
            TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
        ]

        return CustomProgress(
            RecolorOnFailureBarColumn(
                table_column=Column("Progress", ratio=2),
                failing_color="tab:red",
                complete_style=Style.parse("rgb(31,119,180)"),
                finished_style=Style.parse("rgb(31,119,180)"),
            ),
            *columns,
            console=Console(theme=theme),
            include_headers=True,
        )

    def __enter__(self) -> Self:
        self._progress.__enter__()
        self._initialize_tasks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._progress.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_tasks(self) -> None:
        if self.combined:
            self._tasks = [
                self._progress.add_task(
                    "Sampling",
                    completed=0,
                    total=self.total * self.n_bars - 1,
                    task_idx=0,
                    sampling_speed=0,
                    speed_unit="draws/s",
                    failing=False,
                    **{stat: value[0] for stat, value in self.progress_stats.items()},
                )
            ]
        else:
            self._tasks = [
                self._progress.add_task(
                    "Sampling",
                    completed=0,
                    total=self.total - 1,
                    task_idx=task_idx,
                    sampling_speed=0,
                    speed_unit="draws/s",
                    failing=False,
                    **{stat: value[task_idx] for stat, value in self.progress_stats.items()},
                )
                for task_idx in range(self.n_bars)
            ]

    def update(
        self,
        task_id: int,
        advance: int | float,
        failing: bool,
        stats: dict[str, Any],
        is_last: bool,
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
        """
        rich_task_id = self._tasks[task_id]
        if rich_task_id is None:
            return

        self._progress.advance(rich_task_id, advance=advance)

        task = self._progress.tasks[task_id]
        completed = task.completed
        elapsed = task.elapsed if task.elapsed is not None else 0.0

        action = self.step_name.lower()
        speed = completed / max(elapsed, 1e-6)
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
            **stats,
        )

        if is_last:
            # Ensure bar is fully filled on completion
            remaining = task.total - task.completed if task.total else 0
            if remaining > 0:
                self._progress.advance(rich_task_id, advance=remaining)
            self._progress.update(
                rich_task_id,
                failing=failing,
                **stats,
                refresh=True,
            )


def RichSimpleProgress(theme: Theme | None):
    return CustomProgress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("/"),
        TimeElapsedColumn(),
        console=Console(theme=default_progress_theme if theme is None else theme),
    )
