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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pytensor.scalar import discrete_dtypes

from pymc.model.core import modelcontext
from pymc.util import RandomSeed


class ExternalSampler(ABC):
    def __init__(self, model=None):
        model = modelcontext(model)
        self.model = model

    @abstractmethod
    def sample(
        self,
        *,
        tune: int,
        draws: int,
        chains: int,
        initvals: dict[str, Any] | Sequence[dict[str, Any]],
        random_seed: RandomSeed,
        progressbar: bool,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict[str, Any] | None = None,
        compute_convergence_checks: bool,
        **kwargs,
    ):
        pass


class NUTSExternalSampler(ExternalSampler):
    def __init__(self, model=None):
        super().__init__(model)
        if any(var.dtype in discrete_dtypes for var in model.free_RVs):
            raise ValueError("External NUTS samplers can only sample continuous variables")
