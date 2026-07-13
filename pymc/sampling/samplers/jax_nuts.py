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
"""The numpyro and blackjax NUTS implementations as samplers."""

import warnings

from collections.abc import Sequence
from typing import Any, Literal

from pymc.initial_point import StartDict
from pymc.sampling.samplers.base import ExternalSampler, require_continuous_model
from pymc.util import RandomState, get_random_generator

__all__ = ["BlackjaxNUTS", "NumpyroNUTS"]


class _JAXNUTS(ExternalSampler):
    """Shared implementation of the jax-based external NUTS samplers.

    Parameters
    ----------
    target_accept : float, default 0.8
        Target acceptance rate for step size adaptation.
    chain_method : "parallel" or "vectorized", default "parallel"
        Whether chains run under ``jax.pmap`` or ``jax.vmap``.
    postprocessing_backend : "cpu" or "gpu", optional
        Where to compute the transformation of draws to the constrained space.
    keep_untransformed : bool, default False
        Include unconstrained values in the returned posterior.
    jitter_initial_points : bool, default True
        Add jitter to the initial points.
    **nuts_kwargs
        Passed to the underlying NUTS implementation.
    """

    nuts_sampler: Literal["numpyro", "blackjax"]

    def __init__(
        self,
        *,
        target_accept: float = 0.8,
        chain_method: Literal["parallel", "vectorized"] = "parallel",
        postprocessing_backend: Literal["cpu", "gpu"] | None = None,
        keep_untransformed: bool = False,
        jitter_initial_points: bool = True,
        **nuts_kwargs,
    ):
        super().__init__()
        self.target_accept = target_accept
        self.chain_method = chain_method
        self.postprocessing_backend = postprocessing_backend
        self.keep_untransformed = keep_untransformed
        self.jitter_initial_points = jitter_initial_points
        self.nuts_kwargs = nuts_kwargs

    @staticmethod
    def _validate_compile_kwargs(compile_kwargs: dict | None) -> None:
        """Reject compilation requests this sampler cannot honor.

        The model logp is always compiled with the jax backend, so a
        different requested mode would be silently overridden — raise
        instead. The nutpie sampler, by contrast, consumes these options.
        """
        if not compile_kwargs:
            return
        from pytensor.compile.mode import get_mode
        from pytensor.link.jax.linker import JAXLinker

        unsupported = sorted(set(compile_kwargs) - {"mode"})
        if unsupported:
            raise ValueError(f"`compile_kwargs` {unsupported} are not supported by this sampler.")
        if not isinstance(get_mode(compile_kwargs["mode"]).linker, JAXLinker):
            raise ValueError(
                f"The {type(_JAXNUTS).__name__} sampler always compiles the model with the "
                f"jax backend; `compile_kwargs['mode']={compile_kwargs['mode']!r}` (or the "
                "equivalent `backend` argument) is not supported."
            )

    def sample_from_init(
        self,
        *,
        model=None,
        draws: int = 1000,
        tune: int | None = 1000,
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
        """Run NUTS on ``model``.

        Dispositions specific to this sampler:

        - ``cores``: unused. Chain parallelism is governed by the available
          jax devices and the ``chain_method`` constructor argument.
        - ``compile_kwargs``: validated; a non-jax ``mode`` raises.
        - ``discard_tuned_samples=False``: warns, warmup draws are discarded.
        - ``keep_warning_stat=True``: warns, no ``warning`` stat is emitted.
        """
        from pymc.model.core import modelcontext

        model = modelcontext(model)
        require_continuous_model(model, sampler_name=type(self).__name__)
        self._validate_compile_kwargs(compile_kwargs)
        if not discard_tuned_samples:
            warnings.warn(
                "`discard_tuned_samples=False` is ignored: warmup draws are discarded "
                "to bound memory use.",
                UserWarning,
                stacklevel=2,
            )
        if keep_warning_stat:
            warnings.warn(
                "`keep_warning_stat` is ignored: no `warning` sampler stat is emitted.",
                UserWarning,
                stacklevel=2,
            )
        if chains is None:
            chains = 4
        if tune is None:
            tune = 1000

        from pymc.sampling.jax import sample_jax_nuts

        # Derive one master seed without reinterpreting array-like input as a
        # per-chain seed list (which pm.sample accepts and documents).
        seed = int(get_random_generator(random_seed).integers(2**30))
        return sample_jax_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=self.target_accept,
            random_seed=seed,
            initvals=initvals,
            jitter=self.jitter_initial_points,
            model=model,
            var_names=var_names,
            nuts_kwargs=dict(self.nuts_kwargs),
            progressbar=bool(progressbar),
            quiet=quiet,
            keep_untransformed=self.keep_untransformed,
            chain_method=self.chain_method,
            postprocessing_backend=self.postprocessing_backend,
            idata_kwargs=idata_kwargs,
            compute_convergence_checks=compute_convergence_checks,
            nuts_sampler=self.nuts_sampler,
        )


class NumpyroNUTS(_JAXNUTS):
    """NUTS from numpyro. See ``_JAXNUTS`` for the parameters."""

    package = "numpyro"
    nuts_sampler = "numpyro"


class BlackjaxNUTS(_JAXNUTS):
    """NUTS from blackjax. See ``_JAXNUTS`` for the parameters."""

    package = "blackjax"
    nuts_sampler = "blackjax"
