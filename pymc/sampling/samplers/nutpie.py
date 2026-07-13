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
"""The nutpie NUTS implementation as a sampler."""

import warnings

from collections.abc import Sequence
from typing import Any

from pymc.initial_point import StartDict
from pymc.sampling.samplers.base import (
    ExternalSampler,
    SamplerEntry,
    require_continuous_model,
)
from pymc.util import RandomState, _get_seeds_per_chain

__all__ = ["Nutpie", "nuts"]


class Nutpie(ExternalSampler):
    """NUTS from nutpie.

    Parameters
    ----------
    progressbar_theme : Theme, optional
        Theme for the progress bar.
    **nuts_kwargs
        Passed to ``nutpie.sample`` (e.g. ``target_accept``,
        ``max_treedepth``, ``store_unconstrained``).
    """

    package = "nutpie"

    def __init__(self, *, progressbar_theme=None, **nuts_kwargs):
        super().__init__()
        self.progressbar_theme = progressbar_theme
        self.nuts_kwargs = nuts_kwargs

    def sample_from_init(
        self,
        *,
        model=None,
        draws: int = 1000,
        tune: int = 1000,
        chains: int | None = None,
        cores: int | None = None,
        initvals: StartDict | Sequence[StartDict | None] | None = None,
        random_seed: RandomState = None,
        progressbar: bool = True,
        quiet: bool = False,
        discard_tuned_samples: bool = True,
        keep_warning_stat: bool = False,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict[str, Any] | None = None,
        compute_convergence_checks: bool = True,
        compile_kwargs: dict[str, Any] | None = None,
    ):
        """Run nutpie's NUTS on ``model``.

        All run arguments are honored (``compile_kwargs``/``backend`` select
        the numba or jax compilation backend; ``discard_tuned_samples=False``
        stores the warmup draws), except ``keep_warning_stat`` which warns:
        nutpie does not emit PyMC's ``warning`` sampler stat.
        """
        from pymc.model.core import modelcontext
        from pymc.sampling.mcmc import _sample_external_nuts

        model = modelcontext(model)
        require_continuous_model(model, sampler_name="Nutpie")
        if keep_warning_stat:
            warnings.warn(
                "`keep_warning_stat` is ignored: nutpie does not emit the `warning` sampler stat.",
                UserWarning,
                stacklevel=2,
            )
        if chains is None:
            chains = 4
        if cores is None:
            cores = min(4, chains)

        (seed,) = _get_seeds_per_chain(random_seed, 1)
        return _sample_external_nuts(
            sampler="nutpie",
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=[seed],
            initvals=initvals,
            model=model,
            var_names=var_names,
            progressbar=progressbar,
            progressbar_theme=self.progressbar_theme,
            quiet=quiet,
            compute_convergence_checks=compute_convergence_checks,
            discard_tuned_samples=discard_tuned_samples,
            nuts_kwargs=dict(self.nuts_kwargs),
            compile_kwargs={} if compile_kwargs is None else dict(compile_kwargs),
            idata_kwargs=idata_kwargs,
        )


nuts = SamplerEntry(
    "nutpie.nuts",
    Nutpie,
    doc="NUTS via nutpie: `pm.nutpie.nuts(**config)` configures a sampler for "
    "`pm.sample(sampler=...)`; `pm.nutpie.nuts.sample(...)` draws in one flat call.",
)
