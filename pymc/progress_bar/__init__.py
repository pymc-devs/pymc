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
"""Progress bar module for PyMC sampling.

This module provides progress bar rendering during MCMC sampling, with support
for multiple backends:
- Rich-based terminal rendering (default)
- HTML rendering for marimo notebooks (auto-detected)

Public API
----------
ProgressBarManager
    Main class for managing progress bars during sampling.
create_simple_progress
    Factory function for simple progress bars (forward sampling).
ProgressBarType
    Type alias for progress bar configuration options.
default_progress_theme
    Default Rich theme for progress bar colors.
in_marimo_notebook
    Function to detect marimo notebook environment.
"""

from pymc.progress_bar.marimo_progress import in_marimo_notebook
from pymc.progress_bar.progress import (
    ProgressBarManager,
    ProgressBarType,
    create_simple_progress,
    default_progress_theme,
)
from pymc.progress_bar.rich_progress import CustomProgress

__all__ = [
    "CustomProgress",
    "ProgressBarManager",
    "ProgressBarType",
    "create_simple_progress",
    "default_progress_theme",
    "in_marimo_notebook",
]
