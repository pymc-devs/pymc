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

from pymc.model.core import modelcontext
from pymc.util import get_value_vars_from_user_vars


class ExternalSampler(ABC):
    def __init__(self, vars=None, model=None):
        model = modelcontext(model)
        if vars is None:
            vars = model.free_RVs
        else:
            vars = get_value_vars_from_user_vars(vars, model=model)
            if set(vars) != set(model.free_RVs):
                raise ValueError(
                    "External samplers must sample all the model free_RVs, not just a subset"
                )
        self.vars = vars
        self.model = model

    @abstractmethod
    def sample(
        self,
        tune,
        draws,
        chains,
        initvals,
        random_seed,
        progressbar,
        var_names,
        idata_kwargs,
        compute_convergence_checks,
        **kwargs,
    ):
        pass
