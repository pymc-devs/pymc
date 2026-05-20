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
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

from rich.progress import TextColumn
from rich.table import Column
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
        total: int | None = None,
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
        total: int | float | None,
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

    step_name: str = "Draw"

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
        progress_stats["draw"] = [0] * chains

        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self.completed_draws = 0
        total_draws = draws + tune
        self.total_draws = total_draws * chains if self.combined_progress else total_draws

        self._backend = self._create_backend(
            total=self.total_draws,
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

        if not is_last:
            self.completed_draws += 1

        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        failing, all_step_stats = self._extract_stats(stats)
        all_step_stats["draw"] = draw + 1 if not self.combined_progress else draw

        self._backend.update(
            task_id=chain_idx,
            advance=1,
            failing=failing,
            stats=all_step_stats,
            is_last=is_last,
            total=self.total_draws,
        )


class NutpieProgressBarManager(ProgressBarManager):
    """Progress bar manager for nutpie NUTS sampling.

    Bridges ``nutpie.sample``'s ``progress_callback`` (a callable that receives a
    list of ``nutpie.ChainProgress`` objects) to PyMC's progress bar backends,
    so nutpie draws through the same UI as the pymc sampler.
    """

    step_name: str = "Draw"

    def __enter__(self) -> Self:
        self._backend.__enter__()
        # nutpie's progress callback fires from a Rust thread where marimo's
        # cell context is absent.  Replace the backend's output function with
        # a thread-safe version that pins the cell identity.
        if isinstance(self._backend, MarimoProgressBackend):
            from nutpie.sample import _mo_create_replace

            replace = _mo_create_replace()
            if replace is not None:
                self._backend._mo_replace = replace
        return self

    def __init__(
        self,
        chains: int,
        draws: int,
        progressbar: bool | ProgressBarOptions = True,
        progressbar_theme: Theme | str | None = None,
    ):
        super().__init__(
            n_bars=chains,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )
        # Used to compute delta draws between calls
        self._previous_finished = [0] * chains
        self._chain_completed = [False] * chains

        progress_columns = [
            TextColumn("{task.fields[divergences]}", table_column=Column("Divergences", ratio=1)),
            TextColumn("{task.fields[step_size]:0.3f}", table_column=Column("Step size", ratio=1)),
            TextColumn("{task.fields[tree_size]}", table_column=Column("Grad evals", ratio=1)),
        ]
        progress_stats = {
            stat: [0] * chains for stat in ("divergences", "step_size", "tree_size", "draw")
        }
        self._backend = self._create_backend(
            total=None,  # Will be set on first callback
            progress_columns=progress_columns,
            progress_stats=progress_stats,
        )

    def update(self, chain_progresses) -> None:
        """Consume a list of ``nutpie.ChainProgress`` objects and advance each bar."""
        if not self._show_progress:
            return

        if self.combined_progress:
            stats = {
                "divergences": 0,
                "step_size": 0,
                "tree_size": 0,
                "draw": 0,
            }
            delta = 0
            for chain_idx, cp in enumerate(chain_progresses):
                cp_finished_draws = cp.finished_draws
                delta += cp_finished_draws - self._previous_finished[chain_idx]
                self._previous_finished[chain_idx] = cp_finished_draws
                stats["divergences"] += len(cp.divergent_draws)
                stats["tree_size"] = max(stats["tree_size"], cp.latest_num_steps)
                stats["draw"] += cp_finished_draws
            bar_total: int = cp.total_draws * len(chain_progresses)
            # Use the slowest chain's runtime as the combined bar's elapsed.
            max_runtime_ms = max(cp.runtime_ms for cp in chain_progresses)
            if max_runtime_ms > 0:
                self._set_task_elapsed(0, max_runtime_ms / 1000.0)
            self._backend.update(
                task_id=0,
                advance=delta,
                failing=stats["divergences"] > 0,
                stats=stats,
                is_last=stats["draw"] >= bar_total,
                total=bar_total,
            )
        else:
            for chain_idx, cp in enumerate(chain_progresses):
                # With ``cores < chains`` queued chains haven't started yet;
                # skip them so their bar doesn't show progress or elapsed time.
                if not cp.started or self._chain_completed[chain_idx]:
                    continue
                cp_finished_draws = cp.finished_draws
                delta = cp_finished_draws - self._previous_finished[chain_idx]
                self._previous_finished[chain_idx] = cp_finished_draws
                is_last = cp_finished_draws >= cp.total_draws
                if is_last:
                    self._chain_completed[chain_idx] = True
                # Use nutpie's per-chain runtime as the source of truth for
                # elapsed/speed, so reads aren't skewed by the wait time
                # before this chain started.
                if cp.runtime_ms > 0:
                    self._set_task_elapsed(chain_idx, cp.runtime_ms / 1000.0)
                stats = {
                    "divergences": len(cp.divergent_draws),
                    "step_size": cp.step_size,
                    "tree_size": cp.latest_num_steps,
                    "draw": cp_finished_draws,
                }
                self._backend.update(
                    task_id=chain_idx,
                    advance=delta,
                    failing=bool(cp.divergent_draws),
                    stats=stats,
                    is_last=is_last,
                    total=cp.total_draws,
                )

    def _set_task_elapsed(self, task_id: int, elapsed_seconds: float) -> None:
        """Override the backend's elapsed clock to match nutpie's per-chain runtime."""
        backend = self._backend
        now = perf_counter()
        if isinstance(backend, RichProgressBackend):
            rich_task_id = backend._tasks[task_id]
            if rich_task_id is None:
                return
            backend._progress.tasks[task_id].start_time = now - elapsed_seconds
            backend._started[task_id] = True
        elif isinstance(backend, MarimoProgressBackend):
            backend._start_times[task_id] = now - elapsed_seconds


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
