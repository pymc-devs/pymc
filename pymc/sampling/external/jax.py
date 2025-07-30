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
from collections.abc import Sequence
from typing import Literal

from arviz import InferenceData

from pymc.sampling.external.base import ExternalSampler
from pymc.util import RandomState


class JAXSampler(ExternalSampler):
    nuts_sampler = None  # Should be defined by subclass

    def __init__(
        self,
        vars=None,
        model=None,
        postprocessing_backend: Literal["cpu", "gpu"] | None = None,
        chain_method: Literal["parallel", "vectorized"] = "parallel",
        jitter: bool = True,
        keep_untransformed: bool = False,
        nuts_kwargs: dict | None = None,
    ):
        super().__init__(vars, model)
        self.postprocessing_backend = postprocessing_backend
        self.chain_method = chain_method
        self.jitter = jitter
        self.keep_untransformed = keep_untransformed
        self.nuts_kwargs = nuts_kwargs or {}

    def sample(
        self,
        *,
        tune: int = 1000,
        draws: int = 1000,
        chains: int = 4,
        initvals=None,
        random_seed: RandomState | None = None,
        progressbar: bool = True,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict | None = None,
        compute_convergence_checks: bool = True,
        target_accept: float = 0.8,
        nuts_sampler,
        **kwargs,
    ) -> InferenceData:
        from pymc.sampling.jax import sample_jax_nuts

        return sample_jax_nuts(
            tune=tune,
            draws=draws,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            var_names=var_names,
            progressbar=progressbar,
            idata_kwargs=idata_kwargs,
            compute_convergence_checks=compute_convergence_checks,
            initvals=initvals,
            jitter=self.jitter,
            model=self.model,
            chain_method=self.chain_method,
            postprocessing_backend=self.postprocessing_backend,
            keep_untransformed=self.keep_untransformed,
            nuts_kwargs=self.nuts_kwargs,
            nuts_sampler=self.nuts_sampler,
            **kwargs,
        )


class Numpyro(JAXSampler):
    nuts_sampler = "numpyro"


class Blackjax(JAXSampler):
    nuts_sampler = "blackjax"
