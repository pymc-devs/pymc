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
"""Shared utilities for progress bar rendering (no internal dependencies)."""

from rich.theme import Theme

default_progress_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "#1764f4",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


def compute_draw_speed(elapsed: float, draws: int) -> tuple[float, str]:
    """Compute sampling speed and appropriate unit (draws/s or s/draw)."""
    speed = draws / max(elapsed, 1e-6)

    if speed > 1 or speed == 0:
        unit = "draws/s"
    else:
        unit = "s/draw"
        speed = 1 / speed

    return speed, unit


def format_time(seconds: float) -> str:
    """Format elapsed time as mm:ss or hh:mm:ss."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def abbreviate_stat_name(name: str) -> str:
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
