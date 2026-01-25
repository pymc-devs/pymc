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
"""Core progress bar types, utilities, and manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from rich.console import Console
from rich.progress import BarColumn, Task, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.style import Style
from rich.table import Column
from rich.theme import Theme

from pymc.progress_bar.marimo_progress import (
    MarimoProgressBackend,
    MarimoSimpleProgress,
    in_marimo_notebook,
)
from pymc.progress_bar.rich_progress import (
    CustomProgress,
    RecolorOnFailureBarColumn,
    RichProgressBackend,
    default_progress_theme,
)
from pymc.progress_bar.utils import abbreviate_stat_name, compute_draw_speed, format_time

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


class ProgressBackend(Protocol):
    """Protocol defining the interface for progress bar rendering backends.

    Any backend that implements this protocol can be used with ProgressBarManager.
    """

    @property
    def is_enabled(self) -> bool: ...

    def __enter__(self) -> ProgressBackend: ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None: ...

    def update(
        self,
        chain_idx: int,
        draw: int,
        failing: bool,
        stats: dict[str, Any],
        is_last: bool,
    ) -> None: ...


class ProgressBarManager:
    """Manage progress bars displayed during sampling.

    This class automatically detects the execution environment and uses the
    appropriate rendering backend:
    - Rich-based terminal rendering for standard Python/Jupyter environments
    - HTML rendering for marimo notebooks

    The manager handles configuration parsing and stat extraction, delegating
    the actual rendering to the appropriate backend.
    """

    def __init__(
        self,
        step_method: BlockedStep | CompoundStep,
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarType = True,
        progressbar_theme: Theme | None = None,
        progressbar_css: str | None = None,
    ):
        """Initialize the progress bar manager.

        Parameters
        ----------
        step_method : BlockedStep or CompoundStep
            The step method being used for sampling
        chains : int
            Number of chains being sampled
        draws : int
            Number of draws per chain
        tune : int
            Number of tuning steps per chain
        progressbar : bool or ProgressBarType
            Progress bar display mode
        progressbar_theme : Theme, optional
            Rich theme for terminal progress bars
        progressbar_css : str, optional
            Path to custom CSS file for marimo progress bars
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
                    "Invalid value for `progressbar`. Valid values are True (default), "
                    "False (no progress bar), one of 'combined', 'split', 'split+stats', "
                    "or 'combined+stats'."
                )

        self._show_progress = show_progress

        progress_columns, progress_stats = step_method._progressbar_config(chains)
        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self.completed_draws = 0
        self.total_draws = draws + tune
        self.chains = chains
        self._backend: ProgressBackend
        if in_marimo_notebook() and show_progress:
            self._backend = MarimoProgressBackend(
                chains=chains,
                total_draws=self.total_draws,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                progress_stats=progress_stats,
                css_file=progressbar_css,
            )
        else:
            self._backend = RichProgressBackend(
                chains=chains,
                total_draws=self.total_draws,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                progress_columns=progress_columns,
                progress_stats=progress_stats,
                theme=progressbar_theme,
                disable=not show_progress,
            )

    @property
    def _is_marimo(self) -> bool:
        return type(self._backend).__name__ == "MarimoProgressBackend"

    def __enter__(self):
        return self._backend.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._backend.__exit__(exc_type, exc_val, exc_tb)

    def _extract_stats(self, stats) -> tuple[bool, dict[str, Any]]:
        """Extract and process stats from step methods."""
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
                if key in all_step_stats:
                    continue
                else:
                    all_step_stats[key] = val

        return failing, all_step_stats

    def update(self, chain_idx: int, is_last: bool, draw: int, tuning: bool, stats) -> None:
        """Update progress bar with new sampling statistics."""
        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        failing, all_step_stats = self._extract_stats(stats)

        self._backend.update(
            chain_idx=chain_idx,
            draw=draw,
            failing=failing,
            stats=all_step_stats,
            is_last=is_last,
        )


def create_simple_progress(
    description: str,
    total: int,
    progressbar: bool = True,
    progressbar_theme: Theme | None = None,
    progressbar_css: str | None = None,
) -> CustomProgress | MarimoSimpleProgress:
    """Create a simple progress bar appropriate for the current environment."""
    if progressbar and in_marimo_notebook():
        return MarimoSimpleProgress(
            description=description,
            total=total,
            disable=not progressbar,
            css_file=progressbar_css,
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


class SMCProgressBarManager(ProgressBarManager):
    """Progress bar manager specialized for SMC sampling.

    Unlike MCMC which tracks a fixed number of draws, SMC tracks progress
    via the beta parameter (inverse temperature) which goes from 0 to 1.
    """

    def __init__(
        self,
        kernel,
        chains: int,
        progressbar: bool = True,
        progressbar_theme: Theme | None = None,
    ):
        """
        Initialize SMC progress bar manager.

        Parameters
        ----------
        kernel : SMC_KERNEL
            The SMC kernel being used
        chains : int
            Number of chains being sampled
        progressbar : bool, optional
            Whether to display the progress bar
        progressbar_theme : Theme, optional
            The theme to use for the progress bar
        """
        if progressbar_theme is None:
            progressbar_theme = default_progress_theme

        # For SMC, we don't use combined/split modes since chains run independently
        self.combined_progress = False
        self.full_stats = True
        self._show_progress = progressbar

        progress_columns, progress_stats = kernel._progressbar_config(chains)

        self._progress = self._create_smc_progress_bar(
            progress_columns,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )
        self.progress_stats = progress_stats
        self.update_stats_functions = kernel._make_progressbar_update_functions()

        self.chains = chains
        self.desc = "Sampling chain"

        # SMC-specific: track beta instead of draws
        # Beta goes from 0 to 1, so total is 1.0
        self.total_beta = 1.0

        self._tasks: list[Task] | None = None  # type: ignore[annotation-unchecked]

    def __enter__(self):
        """Enter context manager and return self so our update method is used."""
        self._initialize_tasks()
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        return self._progress.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_tasks(self):
        """Initialize progress bar tasks for each chain."""
        self.tasks = [
            self._progress.add_task(
                f"Chain {chain_idx}",
                completed=0.0,
                total=self.total_beta,
                chain_idx=chain_idx,
                sampling_speed=0,
                speed_unit="stages/s",
                failing=False,
                **{stat: value[chain_idx] for stat, value in self.progress_stats.items()},
            )
            for chain_idx in range(self.chains)
        ]

    def update(self, chain_idx, stage, beta, old_beta=None, is_last=False):
        """Update progress bar for a specific chain.

        Parameters
        ----------
        chain_idx : int
            Index of the chain being updated
        stage : int
            Current stage number
        beta : float
            Current beta value (0 to 1)
        old_beta : float, optional
            Previous beta value, used to compute advancement
        is_last : bool, optional
            Whether this is the last update for this chain. Ensures the bar is fully filled at completion.
        """
        if not self._show_progress:
            return

        # Fill progressbar based on change in beta
        if old_beta is not None:
            advance = beta - old_beta
        else:
            advance = 0.0

        elapsed = self._progress.tasks[chain_idx].elapsed
        speed, unit = compute_draw_speed(elapsed, stage, action_name="stage", time_unit="s")

        failing = False  # SMC samplers have no failing concept
        all_step_stats = {}
        stats = [{"stage": stage, "beta": beta}]

        chain_progress_stats = [
            update_stats_fn(step_stats)
            for update_stats_fn, step_stats in zip(self.update_stats_functions, stats, strict=True)
        ]

        for step_stats in chain_progress_stats:
            for key, val in step_stats.items():
                all_step_stats[key] = val

        self._progress.update(
            self.tasks[chain_idx],
            completed=beta,
            advance=advance if advance > 0 else None,
            sampling_speed=speed,
            speed_unit=unit,
            failing=failing,
            **all_step_stats,
        )

        if is_last:
            # Final update to ensure the bar is completely full
            self._progress.update(
                self.tasks[chain_idx],
                completed=self.total_beta,
                failing=failing,
                refresh=True,
                **all_step_stats,
            )

    @staticmethod
    def _create_smc_progress_bar(step_columns, progressbar, progressbar_theme):
        """Create progress bar with SMC-specific columns."""
        # Add kernel-specific columns (Stage, Beta)
        columns = step_columns.copy()

        # Add speed and timing columns
        columns += [
            TextColumn(
                "{task.fields[sampling_speed]:0.2f} {task.fields[speed_unit]}",
                table_column=Column("Stage Speed", ratio=1),
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


__all__ = [
    "ProgressBarManager",
    "SMCProgressBarManager",
    "abbreviate_stat_name",
    "compute_draw_speed",
    "create_simple_progress",
    "default_progress_theme",
    "format_time",
]
