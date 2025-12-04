#   Copyright 2024 - present The PyMC Developers
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

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeAlias

import numpy as np

# -------------------------
# Coordinate typing helpers
# -------------------------

# User-facing coordinate values (before normalization)
CoordValue: TypeAlias = Sequence[Hashable] | np.ndarray | None
Coords: TypeAlias = Mapping[str, CoordValue]

# After normalization / internal representation
StrongCoordValue: TypeAlias = tuple[Hashable, ...] | None
StrongCoords: TypeAlias = Mapping[str, StrongCoordValue]
