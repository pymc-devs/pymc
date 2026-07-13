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
"""Whole-model samplers that can be passed to ``pm.sample(sampler=...)``.

Every sampler exposes two equivalent surfaces: a configured object for the
``pm.sample`` funnel (``pm.sample(sampler=pm.nutpie.nuts(...))``) and a flat
per-algorithm entry point (``pm.nutpie.nuts.sample(...)``).
"""

from pymc.sampling.samplers.base import ExternalSampler, Sampler, SamplerEntry
from pymc.sampling.samplers.step import StepSampler

__all__ = ["ExternalSampler", "Sampler", "SamplerEntry", "StepSampler"]
