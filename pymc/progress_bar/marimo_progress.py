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
from collections.abc import Callable
from time import perf_counter
from typing import Any, Self

from pymc.progress_bar.marimo_progress_css import DEFAULT_CSS
from pymc.progress_bar.utils import compute_draw_speed


def format_time(seconds: float) -> str:
    """Format elapsed time as mm:ss or hh:mm:ss."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def in_marimo_notebook() -> bool:
    """Check if running inside a marimo notebook.

    Returns
    -------
    bool
        True if running in a marimo notebook, False otherwise.
    """
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
        cell_id=cell_id,  # type: ignore[arg-type]
        status=None,
    )


def _mo_create_replace() -> Callable[[object], None] | None:
    """Create mo.output.replace with current cell context pinned."""
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


class MarimoProgressBackend:
    """Marimo-based progress bar backend for HTML rendering.

    This backend renders progress bars as HTML tables in marimo notebooks,
    providing a rich visual display with automatic updates.
    """

    def __init__(
        self,
        chains: int,
        total_draws: int,
        combined: bool,
        full_stats: bool,
        css_theme: str | None = None,
    ):
        self.chains = chains
        self.total_draws = total_draws
        self.combined = combined
        self.full_stats = full_stats
        self._css_theme = DEFAULT_CSS if css_theme is None else css_theme

        self._mo_replace: Callable[[object], None] | None = None
        self._chain_state: list[dict[str, Any]] = []
        self._start_times: list[float] = []

    @property
    def is_enabled(self) -> bool:
        """Whether the progress bar is enabled (always True for marimo backend)."""
        return True

    def __enter__(self):
        """Enter the context manager and initialize display."""
        import marimo as mo

        self._mo_replace = _mo_create_replace()
        self._initialize_tasks()
        mo.output.clear()
        self._render()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager with final render."""
        self._render()

    def _initialize_tasks(self) -> None:
        """Initialize progress tracking state for all chains."""
        if self.combined:
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
            Whether the chain has encountered failures
        stats : dict
            Statistics to display
        is_last : bool
            Whether this is the final update
        """
        self._chain_state[chain_idx]["draws"] = draw
        self._chain_state[chain_idx]["failing"] = failing
        self._chain_state[chain_idx]["stats"] = stats

        if is_last:
            self._chain_state[chain_idx]["draws"] = draw + 1 if not self.combined else draw

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
        stat_keys = []
        if self.full_stats and self._chain_state and self._chain_state[0]["stats"]:
            stat_keys = list(self._chain_state[0]["stats"].keys())

        header_cells = ["Progress", "Draws"]

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
        header_cells += [abbreviations.get(k, k[:6].capitalize()) for k in stat_keys]

        header_cells += ["Speed", "Elapsed"]

        header_row = "<tr>" + "".join(f"<th>{h}</th>" for h in header_cells) + "</tr>"

        data_rows = []
        for i, state in enumerate(self._chain_state):
            data_rows.append(self._render_chain_row(i, state, stat_keys))

        rows_html = "\n".join(data_rows)
        return f"""
            <style>{self._css_theme}</style>
            <table class='pymc-progress-table'>
                <thead>{header_row}</thead>
                <tbody>{rows_html}</tbody>
            </table>
        """

    def _render_chain_row(self, chain_idx: int, state: dict[str, Any], stat_keys: list[str]) -> str:
        """Render a single chain's progress as a table row."""
        draws = state["draws"]
        total = state["total"]
        failing = state["failing"]
        stats = state["stats"]

        pct = (draws / total * 100) if total > 0 else 0
        elapsed = perf_counter() - self._start_times[chain_idx]
        speed, unit = compute_draw_speed(elapsed, draws)

        bar_class = "pymc-progress-bar"
        if failing:
            bar_class += " failing"
        elif pct >= 100:
            bar_class += " finished"

        cells = [
            f'<td><div class="pymc-progress-bar-container"><div class="{bar_class}" style="width: {pct:.1f}%"></div></div></td>',
            f"<td>{draws}/{total} ({pct:.0f}%)</td>",
        ]

        for key in stat_keys:
            val = stats.get(key, "")
            if isinstance(val, float):
                cells.append(f"<td>{val:.3f}</td>")
            else:
                cells.append(f"<td>{val}</td>")

        cells.append(f"<td>{speed:.1f} {unit}</td>")
        cells.append(f"<td>{format_time(elapsed)}</td>")

        return "<tr>" + "".join(cells) + "</tr>"


class MarimoSimpleProgress:
    """Simple marimo-aware progress bar for forward sampling functions.

    This provides a similar interface to CustomProgress but renders using
    HTML in marimo notebooks for better display.
    """

    def __init__(self, theme: str | None = None):
        self.description = ""
        self.total = 0
        self.completed = 0
        self._css_theme = DEFAULT_CSS if theme is None else theme

        self._mo_replace: Callable[[object], None] | None = None
        self._start_time: float = 0.0
        self._task_id = 0

    def __enter__(self) -> Self:
        """Enter the context manager."""
        self._mo_replace = _mo_create_replace()
        self._start_time = perf_counter()

        import marimo as mo

        mo.output.clear()
        self._render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager with final render."""
        self._render()

    def add_task(
        self, description: str, completed: int = 0, total: int | None = None, **kwargs
    ) -> int:
        """Add a task (interface compatibility with CustomProgress).

        Kwargs are ignored since MarimoSimpleProgress
        only supports a single task initialized at construction.
        """
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
        self.completed += advance
        self._render()

    def update(
        self,
        task_id: int | None = None,
        refresh: bool = False,
        completed: int | None = None,
        **kwargs,
    ) -> None:
        """Update the progress bar state.

        Parameters
        ----------
        task_id : int, optional
            Ignored (for interface compatibility)
        refresh : bool
            If True, force a render
        completed : int, optional
            Set completed count
        **kwargs
            Additional arguments ignored for compatibility
        """
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

        if elapsed > 0 and self.completed > 0:
            speed = self.completed / elapsed
            if speed > 1:
                speed_str = f"{speed:.1f} samples/s"
            else:
                speed_str = f"{1 / speed:.1f} s/sample"
        else:
            speed_str = "-- samples/s"

        if self.completed > 0 and self.completed < self.total:
            remaining = (self.total - self.completed) / (self.completed / elapsed)
            remaining_str = format_time(remaining)
        else:
            remaining_str = "--:--"

        elapsed_str = format_time(elapsed)

        bar_class = "pymc-progress-bar"
        if pct >= 100:
            bar_class += " finished"

        return f"""<style>{self._css_theme}</style>
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
