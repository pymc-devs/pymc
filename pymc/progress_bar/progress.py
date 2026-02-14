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
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

from rich.theme import Theme

from pymc.progress_bar.marimo_progress import (
    MarimoProgressBackend,
    MarimoSimpleProgress,
    in_marimo_notebook,
)
from pymc.progress_bar.rich_progress import (
    RichProgressBackend,
    RichSimpleProgress,
)

if TYPE_CHECKING:
    from pymc.smc.kernels import SMC_KERNEL
    from pymc.step_methods.compound import BlockedStep, CompoundStep


ProgressBarOptions = Literal[
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

    def __enter__(self) -> Self: ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None: ...

    def update(
        self,
        task_id: int,
        advance: int | float,
        failing: bool,
        stats: dict[str, Any],
        is_last: bool,
    ) -> None: ...


class NullProgressBackend(nullcontext):
    def update(self, *args, **kwargs): ...


class ProgressBarManager(ABC):
    """Abstract base class for progress bar managers.

    This class handles the common logic for progress bar management including:
    - Environment detection (terminal vs marimo notebook)
    - Backend selection and initialization
    - Configuration parsing

    Subclasses implement sampling-specific logic like how to track progress
    and how to extract statistics from step methods.
    """

    _backend: ProgressBackend
    _show_progress: bool
    combined_progress: bool
    full_stats: bool
    n_bars: int
    step_name: str = "Completed"

    def __init__(
        self,
        n_bars: int,
        progressbar: bool | ProgressBarOptions = True,
        progressbar_theme: Theme | str | None = None,
    ):
        """Initialize the progress bar manager.

        Parameters
        ----------
        n_bars : int
            Number of progress bars to draw
        progressbar : bool or ProgressBarOptions
            Progress bar display mode
        progressbar_theme : Theme or str, optional
            Rich theme for terminal progress bars, or CSS string for marimo
        """
        self.n_bars = n_bars
        self._progressbar_theme = progressbar_theme

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

    def _create_backend(
        self,
        total: int | float,
        progress_columns: list,
        progress_stats: dict[str, list[Any]],
    ) -> ProgressBackend:
        """Create the appropriate backend based on environment.

        Parameters
        ----------
        total : int
            Total number of progress units (draws, etc.)
        progress_columns : list
            Column definitions for the progress bar
        progress_stats : dict
            Initial statistics values by chain

        Returns
        -------
        ProgressBackend
            The appropriate backend for the current environment
        """
        if not self._show_progress:
            return NullProgressBackend()
        elif in_marimo_notebook():
            return MarimoProgressBackend(
                step_name=self.step_name,
                n_bars=1 if self.combined_progress else self.n_bars,
                total=total,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                css_theme=self._progressbar_theme
                if isinstance(self._progressbar_theme, str)
                else None,
            )
        else:
            return RichProgressBackend(
                step_name=self.step_name,
                n_bars=1 if self.combined_progress else self.n_bars,
                total=total,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                progress_columns=progress_columns,
                progress_stats=progress_stats,
                theme=self._progressbar_theme
                if isinstance(self._progressbar_theme, Theme)
                else None,
            )

    def __enter__(self) -> Self:
        self._backend.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._backend.__exit__(exc_type, exc_val, exc_tb)

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update progress bar with new sampling statistics.

        Subclasses must implement this with their specific signature.
        """
        ...


class MCMCProgressBarManager(ProgressBarManager):
    """Progress bar manager for MCMC sampling.

    Tracks progress via draw count with support for tuning phases.
    """

    step_name: str = "Draws"

    def __init__(
        self,
        step_method: BlockedStep | CompoundStep,
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarOptions = True,
        progressbar_theme: Theme | str | None = None,
    ):
        """Initialize the MCMC progress bar manager.

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
        progressbar : bool or ProgressBarOptions
            Progress bar display mode
        progressbar_theme : Theme or str, optional
            Rich theme for terminal progress bars, or CSS string for marimo
        """
        super().__init__(
            n_bars=chains,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )

        progress_columns, progress_stats = step_method._progressbar_config(chains)
        progress_stats["draws"] = [0] * chains

        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self.completed_draws = 0
        self.total_draws = draws + tune

        self._backend = self._create_backend(
            total=self.total_draws * chains if self.combined_progress else self.total_draws,
            progress_columns=progress_columns,
            progress_stats=progress_stats,
        )

    @property
    def chains(self) -> int:
        """Number of chains being sampled."""
        return self.n_bars

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
        """Update progress bar with new sampling statistics.

        Parameters
        ----------
        chain_idx : int
            Index of the chain being updated
        is_last : bool
            Whether this is the final update
        draw : int
            Current draw number
        tuning : bool
            Whether the chain is in tuning phase
        stats : list
            Statistics from each step method
        """
        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        failing, all_step_stats = self._extract_stats(stats)
        all_step_stats["draws"] = draw

        self._backend.update(
            task_id=chain_idx,
            advance=1,
            failing=failing,
            stats=all_step_stats,
            is_last=is_last,
        )


class SMCProgressBarManager(ProgressBarManager):
    """Progress bar manager for SMC sampling.

    Unlike MCMC which tracks a fixed number of draws, SMC tracks progress
    via the beta parameter (inverse temperature) which goes from 0 to 1.
    """

    step_name: str = "Stage"

    def __init__(
        self,
        kernel: SMC_KERNEL,
        chains: int,
        progressbar: bool = True,
        progressbar_theme: Theme | str | None = None,
    ):
        """Initialize the SMC progress bar manager.

        Parameters
        ----------
        kernel : SMC_KERNEL
            The SMC kernel being used
        chains : int
            Number of processes being run.
        progressbar : bool
            Whether to display the progress bar
        progressbar_theme : Theme or str, optional
            Rich theme for terminal progress bars, or CSS string for marimo
        """
        # SMC doesn't use combined/split modes - each chain runs independently
        # and beta goes from 0 to 1, so we force split mode with full stats
        super().__init__(
            n_bars=chains,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )
        # Override base class settings for SMC-specific behavior
        self.combined_progress = False
        self.full_stats = True

        progress_columns, progress_stats = kernel._progressbar_config(chains)
        progress_stats["stage"] = [0] * chains

        self.progress_stats = progress_stats
        self.update_stats_functions = kernel._make_progressbar_update_functions()

        self._backend = self._create_backend(
            total=1.0,
            progress_columns=progress_columns,
            progress_stats=progress_stats,
        )

    @property
    def chains(self):
        return self.n_bars

    def update(
        self,
        chain_idx: int,
        stage: int,
        beta: float,
        old_beta: float | None = None,
        is_last: bool = False,
    ) -> None:
        """Update a progress bar.

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
        is_last : bool
            Whether this is the last update for this chain
        """
        if not self._show_progress:
            return

        stats = [{"beta": beta}]
        all_step_stats: dict[str, Any] = {}

        chain_progress_stats = [
            update_stats_fn(step_stats)
            for update_stats_fn, step_stats in zip(self.update_stats_functions, stats, strict=True)
        ]

        for step_stats in chain_progress_stats:
            for key, val in step_stats.items():
                all_step_stats[key] = val

        all_step_stats["stage"] = stage

        advance = beta - (old_beta if old_beta is not None else 0.0)

        self._backend.update(
            task_id=chain_idx,
            advance=advance,
            failing=False,
            stats=all_step_stats,
            is_last=is_last,
        )


class SimpleProgressBackend(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None: ...

    def add_task(self, description: str, completed: int, total: int) -> int: ...

    def advance(self, task_id: int | None, advance: int = 1) -> None: ...

    def update(self, task_id: int, refresh: bool, completed: int) -> None: ...


class NullSimpleProgress(nullcontext):
    def add_task(self, *args, **kwargs) -> int:
        return 0

    def advance(self, *args, **kwargs) -> None:
        return

    def update(self, *args, **kwargs) -> None:
        return


def create_simple_progress(
    progressbar: bool = True,
    progressbar_theme: Theme | str | None = None,
) -> SimpleProgressBackend:
    """Create a simple progress bar appropriate for the current environment."""
    if not progressbar:
        return NullSimpleProgress()

    elif in_marimo_notebook():
        return MarimoSimpleProgress(
            theme=progressbar_theme if isinstance(progressbar_theme, str) else None
        )

    else:
        return RichSimpleProgress(
            theme=progressbar_theme if isinstance(progressbar_theme, Theme) else None
        )
