#   Copyright 2022 The PyMC Developers
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

"""Helper functions for MCMC, prior and posterior predictive sampling."""

from typing import Union

from typing_extensions import TypeAlias

from pymc.backends.base import BaseTrace, MultiTrace
from pymc.backends.ndarray import NDArray

Backend: TypeAlias = Union[BaseTrace, MultiTrace, NDArray]

__all__ = ()
