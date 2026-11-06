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
from collections.abc import Sequence
from typing import Any

from pymc.exceptions import SamplingError
from pymc.initial_point import StartDict
from pymc.model.core import modelcontext
from pymc.pytensorf import resolve_backend_compile_kwargs
from pymc.sampling.mcmc import _sample_with_step_methods, setup_cores_blas_cores
from pymc.sampling.parallel import _cpu_count, _initialize_multiprocessing_context
from pymc.sampling.samplers.base import Sampler
from pymc.util import RandomState, get_random_generator

__all__ = ["StepSampler"]


class StepSampler(Sampler):
    """PyMC's native step-method machinery as a sampler.

    Wraps step assignment, ``CompoundStep`` and the python sampling loop —
    the engine behind plain ``pm.sample()`` — in the shared
    :class:`~pymc.sampling.samplers.base.Sampler` interface, so the internal
    sampler is a regular sampler rather than a special case of ``pm.sample``.

    Parameters
    ----------
    step : step method or iterable of step methods, optional
        A step method or collection of step methods. Variables without a
        step method are assigned automatically at sample time. By default
        NUTS is used for all continuous variables.
    init : str, default "auto"
        NUTS initialization strategy (only used when NUTS is auto-assigned;
        see :func:`pymc.init_nuts`).
    n_init : int, default 200_000
        Number of initialization iterations for ADVI-based ``init`` strategies.
    jitter_max_retries : int, default 10
        Maximum retries when jittered initial points have invalid logp.
    trace : backend, optional
        A backend instance (e.g. ``ZarrTrace``) to store the raw draws in.
    callback : callable, optional
        Called after every draw with the trace and draw.
    progressbar_theme : Theme, optional
        Theme for the progress bar.
    mp_ctx : multiprocessing.context.BaseContext or str, optional
        Multiprocessing context for parallel sampling.
    backend : str, optional
        Computational backend to compile the step methods with, e.g. "numba",
        "c" or "jax". Shortcut for ``compile_kwargs={"mode": ...}``; passing
        both raises.
    compile_kwargs : dict, optional
        Forwarded to :func:`pymc.compile` when building the step methods.
        Unlike the external samplers, this really is a PyTensor
        ``function`` passthrough, so it stays a dict.
    **step_kwargs
        Passed to the automatically assigned step methods; keys may be
        lowercased step-method names holding a dict of arguments (e.g.
        ``nuts={"target_accept": 0.9}``).
    """

    def __init__(
        self,
        step=None,
        *,
        init: str = "auto",
        n_init: int = 200_000,
        jitter_max_retries: int = 10,
        trace=None,
        callback=None,
        progressbar_theme=None,
        mp_ctx=None,
        backend: str | None = None,
        compile_kwargs: dict[str, Any] | None = None,
        **step_kwargs,
    ):
        self.compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
        if isinstance(trace, list):
            raise ValueError("Please use the `var_names` run argument for partial traces.")
        self.step = step
        self.init = init
        self.n_init = n_init
        self.jitter_max_retries = jitter_max_retries
        self.trace = trace
        self.callback = callback
        self.progressbar_theme = progressbar_theme
        self.mp_ctx = mp_ctx
        self.step_kwargs = step_kwargs

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
        """Run the native step-method machinery on ``model``.

        All run arguments are honored; sampler-specific configuration
        (``step``, NUTS initialization, trace backends, callbacks,
        multiprocessing options, compilation) belongs to the constructor.
        """
        model = modelcontext(model)
        if not model.free_RVs:
            raise SamplingError(
                "Cannot sample from the model, since the model does not contain any free variables."
            )

        if cores is None:
            cores = min(4, _cpu_count())
        if chains is None:
            chains = max(2, cores)
        if random_seed == -1:
            raise ValueError(
                "Setting random_seed = -1 is not allowed. Pass `None` to not specify a seed."
            )
        # tune=None is forwarded: the engine resolves it per step method
        # via `get_default_tune_steps`.

        compile_kwargs = dict(self.compile_kwargs)
        mp_ctx = _initialize_multiprocessing_context(
            self.mp_ctx, mode=compile_kwargs.get("mode"), quiet=quiet
        )
        joined_blas_limiter, cores, num_blas_cores_per_worker = setup_cores_blas_cores(
            blas_cores, chains, cores, mp_ctx
        )
        rngs = get_random_generator(random_seed).spawn(chains)
        random_seed_list = [rng.integers(2**30) for rng in rngs]

        return _sample_with_step_methods(
            model=model,
            step=self.step,
            step_kwargs=dict(self.step_kwargs),
            init=self.init,
            n_init=self.n_init,
            jitter_max_retries=self.jitter_max_retries,
            trace=self.trace,
            callback=self.callback,
            progressbar_theme=self.progressbar_theme,
            return_inferencedata=True,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            rngs=rngs,
            random_seed_list=random_seed_list,
            initvals=initvals,
            progressbar=progressbar,
            quiet=quiet,
            discard_tuned_samples=discard_tuned_samples,
            keep_warning_stat=keep_warning_stat,
            var_names=var_names,
            idata_kwargs=idata_kwargs,
            compute_convergence_checks=compute_convergence_checks,
            compile_kwargs=compile_kwargs,
            mp_ctx=mp_ctx,
            joined_blas_limiter=joined_blas_limiter,
            num_blas_cores_per_worker=num_blas_cores_per_worker,
        )
