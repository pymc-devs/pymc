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
from rich.progress import BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.theme import Theme

if TYPE_CHECKING:
    from pymc.progress_bar.marimo_progress import MarimoProgressBackend, MarimoSimpleProgress
    from pymc.progress_bar.rich_progress import CustomProgress, RichProgressBackend
    from pymc.step_methods.compound import BlockedStep, CompoundStep


# Type alias for progress bar display options
ProgressBarType = Literal[
    "combined",
    "split",
    "combined+stats",
    "stats+combined",
    "split+stats",
    "stats+split",
]

# Default theme for Rich progress bars
default_progress_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "green",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


class ProgressBackend(Protocol):
    """Protocol defining the interface for progress bar rendering backends.

    Backends handle the actual rendering of progress bars, whether to terminal
    (Rich) or HTML (marimo). The ProgressBarManager delegates rendering to
    the appropriate backend based on the execution environment.
    """

    @property
    def is_enabled(self) -> bool:
        """Whether the progress bar is enabled."""
        ...

    def __enter__(self) -> ProgressBackend:
        """Enter the context manager."""
        ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> bool:
        """Exit the context manager."""
        ...

    def update(
        self,
        chain_idx: int,
        draw: int,
        failing: bool,
        stats: dict[str, Any],
        is_last: bool,
    ) -> None:
        """Update progress for a specific chain.

        Parameters
        ----------
        chain_idx : int
            Index of the chain being updated
        draw : int
            Current draw number
        failing : bool
            Whether the chain has encountered failures (e.g., divergences)
        stats : dict
            Statistics to display (e.g., step_size, divergences)
        is_last : bool
            Whether this is the final update for the chain
        """
        ...


def compute_draw_speed(elapsed: float, draws: int) -> tuple[float, str]:
    """Compute sampling speed and appropriate unit.

    Parameters
    ----------
    elapsed : float
        Elapsed time in seconds
    draws : int
        Number of draws completed

    Returns
    -------
    speed : float
        The computed speed value
    unit : str
        Either "draws/s" or "s/draw" depending on speed
    """
    speed = draws / max(elapsed, 1e-6)

    if speed > 1 or speed == 0:
        unit = "draws/s"
    else:
        unit = "s/draw"
        speed = 1 / speed

    return speed, unit


def format_time(seconds: float) -> str:
    """Format elapsed time as mm:ss or hh:mm:ss.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Formatted time string
    """
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def abbreviate_stat_name(name: str) -> str:
    """Abbreviate common statistic names for compact display.

    Parameters
    ----------
    name : str
        Full statistic name

    Returns
    -------
    str
        Abbreviated name (max 6 chars for unknown names)
    """
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
    ):
        """Initialize the progress bar manager.

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
            How and whether to display the progress bar. Options:
            - True: Default "split+stats" mode
            - False: No progress bar
            - "combined": Single progress bar for all chains
            - "split": Separate progress bar per chain
            - "combined+stats" or "stats+combined": Combined with statistics
            - "split+stats" or "stats+split": Split with statistics
        progressbar_theme : Theme, optional
            Rich theme for progress bar colors. Defaults to default_progress_theme.
        """
        if progressbar_theme is None:
            progressbar_theme = default_progress_theme

        # Parse progress bar configuration
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

        # Get progress bar config from step method
        progress_columns, progress_stats = step_method._progressbar_config(chains)
        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self.completed_draws = 0
        self.total_draws = draws + tune
        self.chains = chains

        # Import backends here to avoid circular imports
        from pymc.progress_bar.marimo_progress import MarimoProgressBackend, in_marimo_notebook
        from pymc.progress_bar.rich_progress import RichProgressBackend

        # Select and create appropriate backend
        if in_marimo_notebook() and show_progress:
            self._backend: RichProgressBackend | MarimoProgressBackend = MarimoProgressBackend(
                chains=chains,
                total_draws=self.total_draws,
                combined=self.combined_progress,
                full_stats=self.full_stats,
                progress_stats=progress_stats,
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
        """Whether the marimo backend is active (for backward compatibility)."""
        from pymc.progress_bar.marimo_progress import MarimoProgressBackend

        return isinstance(self._backend, MarimoProgressBackend)

    def __enter__(self):
        """Enter the context manager."""
        return self._backend.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        return self._backend.__exit__(exc_type, exc_val, exc_tb)

    def _extract_stats(self, stats) -> tuple[bool, dict[str, Any]]:
        """Extract and process stats from step methods.

        Parameters
        ----------
        stats : list
            Raw statistics from step methods

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
                    # TODO: Figure out how to integrate duplicate / non-scalar keys
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
            Whether this is the final draw
        draw : int
            Current draw number
        tuning : bool
            Whether currently in tuning phase (unused but kept for interface)
        stats : list
            Statistics from the step methods
        """
        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        # Extract stats (shared between both backends)
        failing, all_step_stats = self._extract_stats(stats)

        # Delegate to backend
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
    from pymc.progress_bar.marimo_progress import MarimoSimpleProgress, in_marimo_notebook
    from pymc.progress_bar.rich_progress import CustomProgress

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
