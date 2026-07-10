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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pymc.model.core import Model, modelcontext
from pymc.util import RandomState
from pymc.vartypes import discrete_types


class ExternalSampler(ABC):
    """Base class for samplers that sample the whole model outside of PyMC's own machinery.

    Unlike step methods, external samplers cannot be combined with other samplers;
    they are responsible for all free variables of the model.

    Sampler-specific configuration belongs to the subclass constructor. ``sample``
    only receives the run-level arguments that ``pm.sample`` forwards; the
    optional ``jitter``/``jitter_max_retries`` keywords (derived from
    ``pm.sample``'s ``init`` string) may be ignored by samplers that control
    initialization themselves.
    """

    def __init__(self, model: Model | None = None):
        self.model = modelcontext(model)

    @abstractmethod
    def sample(
        self,
        *,
        tune: int,
        draws: int,
        chains: int,
        initvals: dict[str, Any] | Sequence[dict[str, Any] | None] | None,
        random_seed: RandomState,
        progressbar: bool,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict[str, Any] | None = None,
        compute_convergence_checks: bool = True,
        jitter: bool | None = None,
        jitter_max_retries: int = 10,
        **kwargs,
    ):
        pass


def require_continuous_model(model: Model, *, sampler_name: str) -> None:
    """Raise if the model has discrete free random variables."""
    if any(var.dtype in discrete_types for var in model.free_RVs):
        raise ValueError(
            f"The {sampler_name} external sampler can only sample models "
            "where all free variables are continuous."
        )
