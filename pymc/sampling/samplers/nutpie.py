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
from typing import Any, Literal

from pymc.initial_point import StartDict
from pymc.model.core import modelcontext
from pymc.sampling.mcmc import _sample_external_nuts, setup_cores_blas_cores
from pymc.sampling.parallel import _cpu_count, _initialize_multiprocessing_context
from pymc.sampling.samplers.base import (
    ExternalSampler,
    SamplerEntry,
    require_continuous_model,
)
from pymc.util import RandomState, get_random_generator

__all__ = ["Nutpie", "nuts"]


class Nutpie(ExternalSampler):
    """NUTS from nutpie.

    Parameters
    ----------
    backend : "numba" or "jax", optional
        Backend nutpie compiles the model with. Defaults to "jax" if
        PyTensor's configured mode uses the jax linker, "numba" otherwise.
    gradient_backend : "pytensor" or "jax", optional
        How nutpie computes the logp gradient. Defaults to nutpie's own
        choice for ``backend``.
    progressbar_theme : Theme, optional
        Theme for the progress bar.
    **nuts_kwargs
        Passed to ``nutpie.sample`` (e.g. ``target_accept``,
        ``max_treedepth``, ``store_unconstrained``).
    """

    package = "nutpie"

    def __init__(
        self,
        *,
        backend: Literal["numba", "jax"] | None = None,
        gradient_backend: Literal["pytensor", "jax"] | None = None,
        progressbar_theme=None,
        **nuts_kwargs,
    ):
        super().__init__()
        # Named rather than an opaque `compile_kwargs` dict, so the supported
        # `nutpie.compile_pymc_model` options are discoverable from the
        # signature. `None` means "leave it to the existing default" rather
        # than forwarding it.
        self.compile_kwargs = {
            name: value
            for name, value in (("backend", backend), ("gradient_backend", gradient_backend))
            if value is not None
        }
        self.progressbar_theme = progressbar_theme
        self.nuts_kwargs = nuts_kwargs

    def sample_from_init(
        self,
        *,
        model=None,
        draws: int = 1000,
        tune: int | None = 1000,
        chains: int | None = None,
        cores: int | None = None,
        blas_cores: int | None | str = "auto",
        initvals: StartDict | Sequence[StartDict | None] | None = None,
        random_seed: RandomState = None,
        progressbar: bool = True,
        quiet: bool = False,
        discard_tuned_samples: bool = True,
        keep_warning_stat: bool = False,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict[str, Any] | None = None,
        compute_convergence_checks: bool = True,
    ):
        """Run nutpie's NUTS on ``model``.

        All run arguments are honored (``discard_tuned_samples=False`` stores
        the warmup draws), except ``keep_warning_stat`` which warns: nutpie
        does not emit PyMC's ``warning`` sampler stat.
        """
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
            # Matches `pm.sample`: at most 4, never more than the machine has,
            # never more than there are chains to run.
            cores = min(4, _cpu_count(), chains)
        if tune is None:
            tune = 1000

        compile_kwargs = dict(self.compile_kwargs)
        # nutpie samples in-process, so the BLAS/OpenMP limit applies to this
        # process. `mp_ctx` only selects whether limiting is safe (see #7354),
        # and nutpie never forks, so resolve it the same way `pm.sample` did.
        joined_blas_limiter, cores, _ = setup_cores_blas_cores(
            blas_cores,
            chains,
            cores,
            _initialize_multiprocessing_context(None, mode=compile_kwargs.get("mode"), quiet=quiet),
        )

        # Derive one master seed without reinterpreting array-like input as a
        # per-chain seed list (which pm.sample accepts and documents).
        seed = int(get_random_generator(random_seed).integers(2**30))
        with joined_blas_limiter():
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
                progressbar=False if quiet else progressbar,
                progressbar_theme=self.progressbar_theme,
                quiet=quiet,
                compute_convergence_checks=compute_convergence_checks,
                discard_tuned_samples=discard_tuned_samples,
                nuts_kwargs=dict(self.nuts_kwargs),
                compile_kwargs=compile_kwargs,
                idata_kwargs=idata_kwargs,
            )


nuts = SamplerEntry(
    "nutpie.nuts",
    Nutpie,
    doc="NUTS via nutpie: `pm.nutpie.nuts(**config)` configures a sampler for "
    "`pm.sample(sampler=...)`; `pm.nutpie.nuts.sample(...)` draws in one flat call.",
)
