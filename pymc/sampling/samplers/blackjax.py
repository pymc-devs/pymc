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
"""Samplers backed by blackjax: ``pm.blackjax.nuts``.

Support for the full blackjax algorithm family (mclmc, mala, ...) is added
on top of this in a follow-up.
"""

from pymc.sampling.samplers.base import SamplerEntry
from pymc.sampling.samplers.jax_nuts import BlackjaxNUTS

__all__ = ["BlackjaxNUTS", "nuts"]

nuts = SamplerEntry(
    "blackjax.nuts",
    BlackjaxNUTS,
    doc="NUTS via blackjax: `pm.blackjax.nuts(**config)` configures a sampler for "
    "`pm.sample(sampler=...)`; `pm.blackjax.nuts.sample(...)` draws in one flat call.",
)
