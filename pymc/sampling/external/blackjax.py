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
"""Generic driver to run any BlackJAX MCMC algorithm on a PyMC model.

BlackJAX exposes all its MCMC algorithms through a uniform interface:
``blackjax.<algorithm>(logdensity_fn, **parameters)`` returns a
``SamplingAlgorithm(init, step)``. This module exploits that uniformity (plus
signature introspection, an idea borrowed from ``bayeux``) so that a single
driver can run every current — and future — blackjax algorithm, instead of
hardcoding one inference loop per algorithm.

Algorithm parameters are supplied in one of three ways:

- statically, by the user (e.g. ``step_size`` for MALA),
- by a single-chain adaptation procedure run during tuning
  (``window_adaptation`` and friends for the HMC family),
- by an algorithm-specific tuner (``mclmc_find_L_and_step_size`` for MCLMC).

SMC (needs a split logprior/loglikelihood), SGMCMC (needs minibatch gradient
estimators), cross-chain adaptation (MEADS/ChEES) and variational algorithms
are out of scope here and raise informative errors.
"""

import inspect
import warnings

from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from typing import Any, Literal

import numpy as np

from pymc.sampling.external.base import ExternalSampler, require_continuous_model
from pymc.util import RandomState, _get_seeds_per_chain

__all__ = ["Blackjax"]


_REQUIRED = "<required>"

# GenerateSamplingAPI instances that do not follow the plain single-chain
# MCMC contract (a logdensity_fn and a step over one chain state).
_UNSUPPORTED_ALGORITHMS = {
    "tempered_smc": "SMC requires a split logprior/loglikelihood; see pymc_extras SMC support",
    "adaptive_tempered_smc": "SMC requires a split logprior/loglikelihood; see pymc_extras SMC support",
    "persistent_sampling_smc": "SMC requires a split logprior/loglikelihood; see pymc_extras SMC support",
    "adaptive_persistent_sampling_smc": "SMC requires a split logprior/loglikelihood; see pymc_extras SMC support",
    "partial_posteriors_smc": "SMC requires a split logprior/loglikelihood; see pymc_extras SMC support",
    "inner_kernel_tuning": "SMC meta-algorithms are not supported",
    "pretuning": "SMC meta-algorithms are not supported",
    "sgld": "SGMCMC requires minibatch gradient estimators, which PyMC does not provide",
    "sghmc": "SGMCMC requires minibatch gradient estimators, which PyMC does not provide",
    "sgnht": "SGMCMC requires minibatch gradient estimators, which PyMC does not provide",
    "csgld": "SGMCMC requires minibatch gradient estimators, which PyMC does not provide",
    "svgd": "SVGD is a particle-based variational method, not an MCMC sampler",
    "elliptical_slice": "requires a Gaussian-prior loglikelihood decomposition of the model",
    "mgrad_gaussian": "requires a latent-Gaussian decomposition of the model",
}

# Adaptation procedures that take (algorithm, logdensity_fn, ...) and tune a
# single chain during warmup. "mclmc" is the bespoke MCLMC tuner.
_WINDOW_LIKE_SCHEMES = ("window", "low_rank", "pathfinder")
_ADAPTATION_SCHEMES = (*_WINDOW_LIKE_SCHEMES, "mclmc")

_DEFAULT_ADAPTATION = {
    "nuts": "window",
    "hmc": "window",
    "dynamic_hmc": "window",
    "mclmc": "mclmc",
}

# Parameters filled in by adaptation, so not required from the user upfront.
_ADAPTED_PARAMETERS = {
    "window": ("step_size", "inverse_mass_matrix"),
    "low_rank": ("step_size", "inverse_mass_matrix"),
    "pathfinder": ("step_size", "inverse_mass_matrix"),
    "mclmc": ("L", "step_size", "inverse_mass_matrix"),
}

# Map blackjax stat names to the ArviZ/PyMC conventions.
_STAT_RENAMES = {
    "is_divergent": "diverging",
    "num_trajectory_expansions": "tree_depth",
    "num_integration_steps": "n_steps",
    "logdensity": "lp",
}


def _import_blackjax():
    # Importing pymc.sampling.jax first sets XLA_FLAGS so that enough host
    # devices exist for pmap-based parallel chains. This must happen before
    # anything initializes the jax backend.
    import pymc.sampling.jax  # noqa: F401

    try:
        import blackjax
    except ImportError as err:
        raise ImportError(
            "The Blackjax external sampler requires blackjax. Install it with "
            "`pip install blackjax`."
        ) from err
    return blackjax


def _fn_params(fn, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return {parameter: default} for ``fn``, with ``_REQUIRED`` for empty defaults."""
    params = {}
    for name, parameter in inspect.signature(fn).parameters.items():
        if name in exclude or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = parameter.default
        params[name] = _REQUIRED if default is inspect.Parameter.empty else default
    return params


def _available_algorithms(blackjax) -> list[str]:
    return sorted(
        name
        for name, obj in vars(blackjax).items()
        if isinstance(obj, blackjax.GenerateSamplingAPI) and name not in _UNSUPPORTED_ALGORITHMS
    )


def _resolve_algorithm(algorithm) -> tuple[str, Any]:
    blackjax = _import_blackjax()

    if isinstance(algorithm, str):
        name = algorithm
        obj = getattr(blackjax, name, None)
        if obj is None:
            raise ValueError(
                f"Unknown blackjax algorithm {name!r}. "
                f"Available algorithms: {_available_algorithms(blackjax)}"
            )
    else:
        obj = algorithm
        names = [n for n, v in vars(blackjax).items() if v is obj]
        name = names[0] if names else getattr(obj, "__name__", str(obj))

    if isinstance(obj, blackjax.GenerateVariationalAPI | blackjax.GeneratePathfinderAPI):
        raise ValueError(
            f"blackjax.{name} is a variational algorithm, not an MCMC sampler, "
            "and cannot be used with `pm.sample`."
        )
    if name in _UNSUPPORTED_ALGORITHMS:
        raise ValueError(f"blackjax.{name} is not supported: {_UNSUPPORTED_ALGORITHMS[name]}.")
    if not isinstance(obj, blackjax.GenerateSamplingAPI):
        raise ValueError(
            f"{algorithm!r} is not a blackjax sampling algorithm. "
            f"Available algorithms: {_available_algorithms(blackjax)}"
        )
    return name, obj


def _adaptation_params(scheme: str | None, blackjax) -> dict[str, Any]:
    if scheme is None:
        return {}
    if scheme == "mclmc":
        return _fn_params(
            blackjax.mclmc_find_L_and_step_size,
            exclude=("mclmc_kernel", "num_steps", "state", "rng_key", "params"),
        )
    factory = _window_like_factory(scheme, blackjax)
    return _fn_params(factory, exclude=("algorithm", "logdensity_fn", "adaptation_info_fn"))


def _window_like_factory(scheme: str, blackjax):
    return {
        "window": blackjax.window_adaptation,
        "low_rank": blackjax.low_rank_window_adaptation,
        "pathfinder": blackjax.pathfinder_adaptation,
    }[scheme]


def _extract_stats(state, info) -> dict[str, Any]:
    """Flatten a blackjax Info namedtuple into per-draw scalar sampling stats."""
    import jax.numpy as jnp

    stats = {}
    for name, value in getattr(info, "_asdict", dict)().items():
        if value is None or hasattr(value, "_asdict") or isinstance(value, list | tuple | dict):
            # Skip nested states/proposals and containers (position/momentum pytrees)
            continue
        try:
            ndim = jnp.ndim(value)
        except TypeError:
            continue
        if ndim == 0:
            stats[_STAT_RENAMES.get(name, name)] = value
    logdensity = getattr(state, "logdensity", None)
    if logdensity is not None:
        stats["lp"] = logdensity
    return stats


class Blackjax(ExternalSampler):
    """Sample a PyMC model with any BlackJAX MCMC algorithm.

    Parameters
    ----------
    algorithm : str or blackjax algorithm, default "nuts"
        Name of a blackjax MCMC algorithm (e.g. ``"nuts"``, ``"hmc"``,
        ``"mclmc"``, ``"mala"``, ``"barker"``, ``"ghmc"``, ...) or the
        blackjax algorithm object itself (e.g. ``blackjax.nuts``).
    adaptation : str or None, optional
        How the algorithm parameters are tuned during warmup. One of
        ``"window"``, ``"low_rank"``, ``"pathfinder"`` (HMC-family window
        adaptation variants), ``"mclmc"`` (the MCLMC-specific tuner), or
        ``None`` (no adaptation; all required algorithm parameters must be
        passed explicitly). Defaults to the appropriate scheme per algorithm
        (e.g. ``"window"`` for nuts/hmc, ``"mclmc"`` for mclmc, ``None`` for
        algorithms without a standard tuner).
    target_accept : float, optional
        Target acceptance rate for adaptation schemes that support it.
    model : Model, optional
        Model to sample from. Taken from the model context if not provided.
    chain_method : "parallel" or "vectorized", default "parallel"
        Whether chains run under ``jax.pmap`` or ``jax.vmap``.
    postprocessing_backend : "cpu" or "gpu", optional
        Where to compute the transformation of draws to the constrained space.
    keep_untransformed : bool, default False
        Include unconstrained values in the returned posterior.
    jitter : bool, default True
        Add jitter to the initial points.
    **kwargs
        Additional keyword arguments are routed automatically — based on the
        blackjax function signatures — either to the algorithm (e.g.
        ``max_num_doublings`` for nuts, ``step_size`` for mala) or to the
        adaptation procedure (e.g. ``is_mass_matrix_diagonal`` for window
        adaptation, ``desired_energy_var`` for the mclmc tuner). Use
        :meth:`get_kwargs` to discover all accepted parameters and their
        defaults. Parameters passed explicitly override adapted values.

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Normal("x", shape=3)
            y = pm.Normal("y", mu=x.sum(), observed=1.0)

            idata = pm.sample(external_sampler=pm.external.Blackjax("mclmc"))
    """

    def __init__(
        self,
        algorithm="nuts",
        *,
        adaptation: str | None | Literal["auto"] = "auto",
        target_accept: float | None = None,
        model=None,
        chain_method: Literal["parallel", "vectorized"] = "parallel",
        postprocessing_backend: Literal["cpu", "gpu"] | None = None,
        keep_untransformed: bool = False,
        jitter: bool = True,
        **kwargs,
    ):
        super().__init__(model)
        require_continuous_model(self.model, sampler_name="Blackjax")
        blackjax = _import_blackjax()

        self.algorithm_name, self._algorithm = _resolve_algorithm(algorithm)

        if adaptation == "auto":
            adaptation = _DEFAULT_ADAPTATION.get(self.algorithm_name)
        if adaptation is not None and adaptation not in _ADAPTATION_SCHEMES:
            raise ValueError(
                f"Unknown adaptation scheme {adaptation!r}. "
                f"Choose one of {_ADAPTATION_SCHEMES} or None. "
                "Cross-chain adaptation (MEADS/ChEES) is not supported yet."
            )
        self.adaptation = adaptation

        self._algorithm_params = _fn_params(
            self._algorithm.differentiable, exclude=("logdensity_fn",)
        )
        self._adaptation_params = _adaptation_params(adaptation, blackjax)

        if adaptation == "mclmc" and self.algorithm_name != "mclmc":
            raise ValueError(
                f"adaptation='mclmc' can only be used with the mclmc algorithm, "
                f"not {self.algorithm_name!r}."
            )
        adapted = _ADAPTED_PARAMETERS.get(adaptation, ()) if adaptation is not None else ()
        if adaptation in _WINDOW_LIKE_SCHEMES and not all(
            name in self._algorithm_params for name in adapted
        ):
            raise ValueError(
                f"adaptation={adaptation!r} tunes {adapted}, but blackjax."
                f"{self.algorithm_name} does not accept these parameters. "
                f"Pass adaptation=None and set parameters explicitly: {self.get_kwargs()}"
            )

        self.algorithm_kwargs = {k: v for k, v in kwargs.items() if k in self._algorithm_params}
        self.adaptation_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._adaptation_params and k not in self._algorithm_params
        }
        unknown = set(kwargs) - set(self.algorithm_kwargs) - set(self.adaptation_kwargs)
        if unknown:
            raise TypeError(
                f"Unknown keyword arguments {sorted(unknown)} for blackjax."
                f"{self.algorithm_name} with adaptation={adaptation!r}. "
                f"Accepted arguments: {self.get_kwargs()}"
            )

        if target_accept is not None:
            if "target_acceptance_rate" not in self._adaptation_params:
                raise ValueError(
                    f"target_accept is not supported by blackjax.{self.algorithm_name} "
                    f"with adaptation={adaptation!r}."
                )
            self.adaptation_kwargs["target_acceptance_rate"] = target_accept

        missing = [
            name
            for name, default in self._algorithm_params.items()
            if default is _REQUIRED and name not in adapted and name not in self.algorithm_kwargs
        ]
        if missing:
            raise ValueError(
                f"blackjax.{self.algorithm_name} with adaptation={adaptation!r} "
                f"requires explicit values for {missing}, e.g. "
                f"`pm.external.Blackjax({self.algorithm_name!r}, {missing[0]}=...)`."
            )

        self.chain_method = chain_method
        self.postprocessing_backend = postprocessing_backend
        self.keep_untransformed = keep_untransformed
        self.jitter = jitter

    def get_kwargs(self) -> dict[str, dict[str, Any]]:
        """Return all accepted keyword arguments and their (current) values.

        Parameters the user passed at construction override the blackjax
        defaults; parameters shown as ``"<required>"`` must be provided unless
        an adaptation scheme tunes them.
        """
        return {
            "algorithm": {
                name: self.algorithm_kwargs.get(name, default)
                for name, default in self._algorithm_params.items()
            },
            "adaptation": {
                name: self.adaptation_kwargs.get(name, default)
                for name, default in self._adaptation_params.items()
            },
        }

    def sample(
        self,
        *,
        tune: int = 1000,
        draws: int = 1000,
        chains: int = 4,
        initvals=None,
        random_seed: RandomState = None,
        progressbar: bool = True,
        var_names: Sequence[str] | None = None,
        idata_kwargs: dict | None = None,
        compute_convergence_checks: bool = True,
        quiet: bool = False,
        jitter: bool | None = None,
        jitter_max_retries: int = 10,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                f"Unsupported arguments {sorted(kwargs)}. The Blackjax external sampler "
                "is configured at construction, e.g. `pm.external.Blackjax(target_accept=0.9)`."
            )
        if tune == 0 and self.adaptation is not None:
            raise ValueError(
                "tune=0 is incompatible with adaptation. Pass explicit algorithm "
                "parameters and `adaptation=None` to sample without tuning."
            )

        import jax

        from pymc.sampling.jax import (
            _get_batched_jittered_initial_points,
            _postprocess_and_build_idata,
            get_jaxified_logp,
        )

        blackjax = _import_blackjax()

        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

        logp_fn = get_jaxified_logp(self.model)
        initial_points = _get_batched_jittered_initial_points(
            model=self.model,
            chains=chains,
            initvals=initvals,
            random_seed=random_seed,
            jitter=self.jitter if jitter is None else jitter,
            jitter_max_retries=jitter_max_retries,
            logp_fn=logp_fn,
        )
        if chains == 1:
            initial_points = [np.stack(init_state) for init_state in zip(initial_points)]

        map_fn: Callable
        if self.chain_method == "parallel":
            if chains > jax.local_device_count():
                # Happens when the jax backend was initialized before pymc could
                # request extra host devices via XLA_FLAGS.
                warnings.warn(
                    f"There are not enough devices to run parallel chains: expected {chains} "
                    f"but got {jax.local_device_count()}. Chains will be drawn vectorized. "
                    "Import pymc before running any jax computation to enable parallel chains, "
                    "or pass chain_method='vectorized' to silence this warning.",
                    UserWarning,
                )
                map_fn = jax.vmap
            else:
                map_fn = jax.pmap
        elif self.chain_method == "vectorized":
            map_fn = jax.vmap
        else:
            raise ValueError(
                "Only supporting the following methods to draw chains: 'parallel' or 'vectorized'"
            )

        keys = jax.random.split(jax.random.PRNGKey(random_seed), chains)
        run_fn = partial(
            self._inference_loop,
            logp_fn=logp_fn,
            tune=tune,
            draws=draws,
            progress_bar=bool(progressbar) and not quiet,
        )

        tic1 = datetime.now()
        raw_mcmc_samples, sample_stats = map_fn(run_fn)(keys, initial_points)
        tic2 = datetime.now()

        return _postprocess_and_build_idata(
            model=self.model,
            raw_mcmc_samples=raw_mcmc_samples,
            sample_stats=sample_stats,
            library=blackjax,
            sampling_time=(tic2 - tic1).total_seconds(),
            tune=tune,
            var_names=var_names,
            keep_untransformed=self.keep_untransformed,
            postprocessing_backend=self.postprocessing_backend,
            idata_kwargs=idata_kwargs,
            compute_convergence_checks=compute_convergence_checks,
            quiet=quiet,
        )

    def _inference_loop(self, seed, init_position, *, logp_fn, tune, draws, progress_bar):
        import jax
        import jax.numpy as jnp

        blackjax = _import_blackjax()

        from blackjax.adaptation.base import get_filter_adapt_info_fn

        adapt_key, init_key, warmup_key, sample_key = jax.random.split(seed, 4)
        algorithm_kwargs = dict(self.algorithm_kwargs)

        if self.adaptation in _WINDOW_LIKE_SCHEMES:
            adapt = _window_like_factory(self.adaptation, blackjax)(
                algorithm=self._algorithm,
                logdensity_fn=logp_fn,
                adaptation_info_fn=get_filter_adapt_info_fn(),
                **self.adaptation_kwargs,
                **algorithm_kwargs,
            )
            (last_state, tuned_params), _ = adapt.run(adapt_key, init_position, num_steps=tune)
            algorithm = self._algorithm(logp_fn, **{**tuned_params, **algorithm_kwargs})
        elif self.adaptation == "mclmc":
            integrator = algorithm_kwargs.get(
                "integrator", self._algorithm_params.get("integrator")
            )
            initial_state = self._algorithm.init(
                position=init_position, logdensity_fn=logp_fn, rng_key=init_key
            )
            last_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=lambda inverse_mass_matrix: self._algorithm.build_kernel(
                    logdensity_fn=logp_fn,
                    integrator=integrator,
                    inverse_mass_matrix=inverse_mass_matrix,
                ),
                num_steps=tune,
                state=initial_state,
                rng_key=adapt_key,
                **self.adaptation_kwargs,
            )
            tuned = {
                "L": tuned_params.L,
                "step_size": tuned_params.step_size,
                "inverse_mass_matrix": tuned_params.inverse_mass_matrix,
            }
            algorithm = self._algorithm(logp_fn, **{**tuned, **algorithm_kwargs})
        else:
            algorithm = self._algorithm(logp_fn, **algorithm_kwargs)
            last_state = algorithm.init(init_position, init_key)
            if tune > 0:
                # No parameters to adapt; use the tuning phase as plain burn-in.
                warmup_keys = jax.random.split(warmup_key, tune)
                last_state, _ = jax.lax.scan(
                    lambda state, key: (algorithm.step(key, state)[0], None),
                    last_state,
                    warmup_keys,
                )

        def _one_step(state, xs):
            _, rng_key = xs
            state, info = algorithm.step(rng_key, state)
            return state, (state.position, _extract_stats(state, info))

        keys = jax.random.split(sample_key, draws)
        scan_fn = blackjax.progress_bar.gen_scan_fn(draws, progress_bar)
        _, (samples, stats) = scan_fn(_one_step, last_state, (jnp.arange(draws), keys))
        return samples, stats


def _make_algorithm_factory(name: str, algorithm) -> Callable[..., Blackjax]:
    def factory(**kwargs) -> Blackjax:
        return Blackjax(algorithm, **kwargs)

    factory.__name__ = name
    factory.__qualname__ = name
    default_adaptation = _DEFAULT_ADAPTATION.get(name)
    factory.__doc__ = (
        f"Create a :class:`~pymc.sampling.external.blackjax.Blackjax` external sampler "
        f"that draws with ``blackjax.{name}`` (default adaptation: {default_adaptation!r}).\n\n"
        "Keyword arguments are routed to the algorithm or the adaptation procedure; "
        "call ``.get_kwargs()`` on the returned sampler to list them all.\n\n"
        f"Usage: ``pm.sample(external_sampler=pm.external.blackjax.{name}(...))``"
    )
    return factory


def __getattr__(name: str):
    """Expose every supported blackjax algorithm as a per-algorithm factory.

    ``pm.external.blackjax.mclmc(**kwargs)`` is equivalent to
    ``pm.external.Blackjax("mclmc", **kwargs)``. The available names are
    enumerated dynamically from the installed blackjax version.
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    blackjax = _import_blackjax()
    if getattr(blackjax, name, None) is not None:
        try:
            resolved_name, resolved = _resolve_algorithm(name)
        except ValueError as err:
            raise AttributeError(str(err)) from None
        return _make_algorithm_factory(resolved_name, resolved)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available blackjax algorithms: {_available_algorithms(blackjax)}"
    )


def __dir__() -> list[str]:
    names = list(globals())
    try:
        names += _available_algorithms(_import_blackjax())
    except ImportError:
        pass
    return sorted(set(names))
