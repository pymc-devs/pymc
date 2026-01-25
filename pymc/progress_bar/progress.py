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
        chain_idx: int,
        draw: int,
        failing: bool,
        stats: dict[str, Any],
        is_last: bool,
    ) -> None: ...


class NullProgressBackend(nullcontext):
    def update(self, *args, **kwargs): ...


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
        progressbar: bool | ProgressBarOptions = True,
        progressbar_theme: Theme | str | None = None,
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

        if not show_progress:
            self._backend = NullProgressBackend()
        elif in_marimo_notebook():
            self._backend = MarimoProgressBackend(
                chains=chains,
                total_draws=self.total_draws,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                css_theme=progressbar_theme if isinstance(progressbar_theme, str) else None,
            )
        else:
            self._backend = RichProgressBackend(
                chains=chains,
                total_draws=self.total_draws,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                progress_columns=progress_columns,
                progress_stats=progress_stats,
                theme=progressbar_theme if isinstance(progressbar_theme, Theme) else None,
            )

    def __enter__(self) -> ProgressBackend:
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
