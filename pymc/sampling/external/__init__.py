#   Copyright 2026 - present The PyMC Developers
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
"""External whole-model samplers that can be passed to ``pm.sample(external_sampler=...)``.

Based on the API drafted in https://github.com/pymc-devs/pymc/pull/7880.
"""

from pymc.sampling.external.base import ExternalSampler
from pymc.sampling.external.blackjax import Blackjax

__all__ = ["Blackjax", "ExternalSampler"]
