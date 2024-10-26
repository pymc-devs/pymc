#   Copyright 2024 The PyMC Developers
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
import logging
import os
import re

from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from typing import Any, Literal

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt

from arviz.data.base import make_attrs
from jax.lax import scan
from pytensor.compile import SharedVariable, Supervisor, mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.raise_op import Assert
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.type import RandomType

from pymc import Model, modelcontext
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)
from pymc.distributions.multivariate import PosDefMatrix
from pymc.initial_point import StartDict
from pymc.logprob.utils import CheckParameterValue
from pymc.sampling.mcmc import _init_jitter
from pymc.stats.convergence import log_warnings, run_convergence_checks
from pymc.util import (
    RandomSeed,
    RandomState,
    _get_seeds_per_chain,
    get_default_varnames,
)

logger = logging.getLogger(__name__)

xla_flags_env = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(r"--xla_force_host_platform_device_count=.+\s", "", xla_flags_env).split()
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={100}", *xla_flags])

__all__ = (
    "get_jaxified_graph",
    "get_jaxified_logp",
    "sample_blackjax_nuts",
    "sample_numpyro_nuts",
)


@jax_funcify.register(Assert)
@jax_funcify.register(CheckParameterValue)
def jax_funcify_Assert(op, **kwargs):
    # Jax does not allow assert whose values aren't known during JIT compilation
    # within it's JIT-ed code. Hence we need to make a simple pass through
    # version of the Assert Op.
    # https://github.com/google/jax/issues/2273#issuecomment-589098722
    def assert_fn(value, *inps):
        return value

    return assert_fn


@jax_funcify.register(PosDefMatrix)
def jax_funcify_PosDefMatrix(op, **kwargs):
    def posdefmatrix_fn(value, *inps):
        no_pos_def = jnp.any(jnp.isnan(jnp.linalg.cholesky(value)))
        return jnp.invert(no_pos_def)

    return posdefmatrix_fn


def _replace_shared_variables(graph: list[TensorVariable]) -> list[TensorVariable]:
    """Replace shared variables in graph by their constant values.

    Raises
    ------
    ValueError
        If any shared variable contains default_updates
    """
    shared_variables = [var for var in graph_inputs(graph) if isinstance(var, SharedVariable)]

    if any(isinstance(var.type, RandomType) for var in shared_variables):
        raise ValueError(
            "Graph contains shared RandomType variables which cannot be safely replaced"
        )

    if any(var.default_update is not None for var in shared_variables):
        raise ValueError(
            "Graph contains shared variables with default_update which cannot "
            "be safely replaced."
        )

    replacements = {var: pt.constant(var.get_value(borrow=True)) for var in shared_variables}

    new_graph = clone_replace(graph, replace=replacements)
    return new_graph


def get_jaxified_graph(
    inputs: list[TensorVariable] | None = None,
    outputs: list[TensorVariable] | None = None,
) -> list[TensorVariable]:
    """Compile a PyTensor graph into an optimized JAX function."""
    graph = _replace_shared_variables(outputs) if outputs is not None else None

    fgraph = FunctionGraph(inputs=inputs, outputs=graph, clone=True)
    # We need to add a Supervisor to the fgraph to be able to run the
    # JAX sequential optimizer without warnings. We made sure there
    # are no mutable input variables, so we only need to check for
    # "destroyers". This should be automatically handled by PyTensor
    # once https://github.com/aesara-devs/aesara/issues/637 is fixed.
    fgraph.attach_feature(
        Supervisor(
            input
            for input in fgraph.inputs
            if not (hasattr(fgraph, "destroyers") and fgraph.has_destroyers([input]))
        )
    )
    mode.JAX.optimizer.rewrite(fgraph)

    # We now jaxify the optimized fgraph
    return jax_funcify(fgraph)


def get_jaxified_logp(model: Model, negative_logp=True) -> Callable:
    model_logp = model.logp()
    if not negative_logp:
        model_logp = -model_logp
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[model_logp])

    def logp_fn_wrap(x):
        return logp_fn(*x)[0]

    return logp_fn_wrap


def _get_log_likelihood(
    model: Model,
    samples,
    backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "scan",
) -> dict:
    """Compute log-likelihood for all observations."""
    elemwise_logp = model.logp(model.observed_RVs, sum=False)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=elemwise_logp)
    result = _postprocess_samples(
        jax_fn,
        samples,
        backend,
        postprocessing_vectorize=postprocessing_vectorize,
        donate_samples=False,
    )
    return {v.name: r for v, r in zip(model.observed_RVs, result)}


def _device_put(input, device: str):
    return jax.device_put(input, jax.devices(device)[0])


def _postprocess_samples(
    jax_fn: Callable,
    raw_mcmc_samples: list[TensorVariable],
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "vmap",
    donate_samples: bool = False,
) -> list[TensorVariable]:
    if postprocessing_vectorize == "scan":
        t_raw_mcmc_samples = [jnp.swapaxes(t, 0, 1) for t in raw_mcmc_samples]
        jax_vfn = jax.vmap(jax_fn)
        _, outs = scan(
            lambda _, x: ((), jax_vfn(*x)),
            (),
            _device_put(t_raw_mcmc_samples, postprocessing_backend),
        )
        return [jnp.swapaxes(t, 0, 1) for t in outs]
    elif postprocessing_vectorize == "vmap":

        def process_fn(x):
            return jax.vmap(jax.vmap(jax_fn))(*_device_put(x, postprocessing_backend))

        return jax.jit(process_fn, donate_argnums=0 if donate_samples else None)(raw_mcmc_samples)

    else:
        raise ValueError(f"Unrecognized postprocessing_vectorize: {postprocessing_vectorize}")


def _get_batched_jittered_initial_points(
    model: Model,
    chains: int,
    initvals: StartDict | Sequence[StartDict | None] | None,
    random_seed: RandomSeed,
    jitter: bool = True,
    jitter_max_retries: int = 10,
) -> np.ndarray | list[np.ndarray]:
    """Get jittered initial point in format expected by NumPyro MCMC kernel.

    Returns
    -------
    out: list of ndarrays
        list with one item per variable and number of chains as batch dimension.
        Each item has shape `(chains, *var.shape)`
    """
    initial_points = _init_jitter(
        model,
        initvals,
        seeds=_get_seeds_per_chain(random_seed, chains),
        jitter=jitter,
        jitter_max_retries=jitter_max_retries,
    )
    initial_points_values = [list(initial_point.values()) for initial_point in initial_points]
    if chains == 1:
        return initial_points_values[0]
    return [np.stack(init_state) for init_state in zip(*initial_points_values)]


def _blackjax_inference_loop(
    seed, init_position, logprob_fn, draws, tune, target_accept, **adaptation_kwargs
):
    import blackjax

    from blackjax.adaptation.base import get_filter_adapt_info_fn

    algorithm_name = adaptation_kwargs.pop("algorithm", "nuts")
    if algorithm_name == "nuts":
        algorithm = blackjax.nuts
    elif algorithm_name == "hmc":
        algorithm = blackjax.hmc
    else:
        raise ValueError("Only supporting 'nuts' or 'hmc' as algorithm to draw samples.")

    adapt = blackjax.window_adaptation(
        algorithm=algorithm,
        logdensity_fn=logprob_fn,
        target_acceptance_rate=target_accept,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        **adaptation_kwargs,
    )
    (last_state, tuned_params), _ = adapt.run(seed, init_position, num_steps=tune)
    kernel = algorithm(logprob_fn, **tuned_params).step

    def _one_step(state, xs):
        _, rng_key = xs
        state, info = kernel(rng_key, state)
        position = state.position
        stats = {
            "diverging": info.is_divergent,
            "energy": info.energy,
            "tree_depth": info.num_trajectory_expansions,
            "n_steps": info.num_integration_steps,
            "acceptance_rate": info.acceptance_rate,
            "lp": state.logdensity,
        }
        return state, (position, stats)

    progress_bar = adaptation_kwargs.pop("progress_bar", False)

    keys = jax.random.split(seed, draws)
    scan_fn = blackjax.progress_bar.gen_scan_fn(draws, progress_bar)
    _, (samples, stats) = scan_fn(_one_step, last_state, (jnp.arange(draws), keys))

    return samples, stats


def _sample_blackjax_nuts(
    model: Model,
    target_accept: float,
    tune: int,
    draws: int,
    chains: int,
    chain_method: str | None,
    progressbar: bool,
    random_seed: int,
    initial_points,
    nuts_kwargs,
) -> az.InferenceData:
    """
    Draw samples from the posterior using the NUTS method from the ``blackjax`` library.

    Parameters
    ----------
    draws : int, default 1000
        The number of samples to draw. The number of tuned samples are discarded by
        default.
    tune : int, default 1000
        Number of iterations to tune. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number
        specified in the ``draws`` argument.
    chains : int, default 4
        The number of chains to sample.
    target_accept : float in [0, 1].
        The step size is tuned such that we approximate this acceptance rate. Higher
        values like 0.9 or 0.95 often work better for problematic posteriors.
    random_seed : int, RandomState or Generator, optional
        Random seed used by the sampling steps.
    initvals: StartDict or Sequence[Optional[StartDict]], optional
        Initial values for random variables provided as a dictionary (or sequence of
        dictionaries) mapping the random variable (by name or reference) to desired
        starting values.
    jitter: bool, default True
        If True, add jitter to initial points.
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "parallel", and
        "vectorized".
    postprocessing_backend: Optional[Literal["cpu", "gpu"]], default None,
        Specify how postprocessing should be computed. gpu or cpu
    postprocessing_vectorize: Literal["vmap", "scan"], default "scan"
        How to vectorize the postprocessing: vmap or sequential scan
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz.from_dict`. It also accepts a boolean as
        value for the ``log_likelihood`` key to indicate that the pointwise log
        likelihood should not be included in the returned object. Values for
        ``observed_data``, ``constant_data``, ``coords``, and ``dims`` are inferred from
        the ``model`` argument if not provided in ``idata_kwargs``. If ``coords`` and
        ``dims`` are provided, they are used to update the inferred dictionaries.

    Returns
    -------
    InferenceData
        ArviZ ``InferenceData`` object that contains the posterior samples, together
        with their respective sample stats and pointwise log likeihood values (unless
        skipped with ``idata_kwargs``).
    """
    import blackjax

    # Adapted from numpyro
    if chain_method == "parallel":
        map_fn = jax.pmap
    elif chain_method == "vectorized":
        map_fn = jax.vmap
    else:
        raise ValueError(
            "Only supporting the following methods to draw chains:" ' "parallel" or "vectorized"'
        )

    if chains == 1:
        initial_points = [np.stack(init_state) for init_state in zip(initial_points)]

    logprob_fn = get_jaxified_logp(model)

    seed = jax.random.PRNGKey(random_seed)
    keys = jax.random.split(seed, chains)

    nuts_kwargs["progress_bar"] = progressbar
    get_posterior_samples = partial(
        _blackjax_inference_loop,
        logprob_fn=logprob_fn,
        tune=tune,
        draws=draws,
        target_accept=target_accept,
        **nuts_kwargs,
    )

    raw_mcmc_samples, sample_stats = map_fn(get_posterior_samples)(keys, initial_points)
    return raw_mcmc_samples, sample_stats, blackjax


# Adopted from arviz numpyro extractor
def _numpyro_stats_to_dict(posterior):
    """Extract sample_stats from NumPyro posterior."""
    rename_key = {
        "potential_energy": "lp",
        "adapt_state.step_size": "step_size",
        "num_steps": "n_steps",
        "accept_prob": "acceptance_rate",
    }
    data = {}
    for stat, value in posterior.get_extra_fields(group_by_chain=True).items():
        if isinstance(value, dict | tuple):
            continue
        name = rename_key.get(stat, stat)
        value = value.copy()
        data[name] = value
        if stat == "num_steps":
            data["tree_depth"] = np.log2(value).astype(int) + 1
    return data


def _sample_numpyro_nuts(
    model: Model,
    target_accept: float,
    tune: int,
    draws: int,
    chains: int,
    chain_method: str | None,
    progressbar: bool,
    random_seed: int,
    initial_points,
    nuts_kwargs: dict[str, Any],
):
    import numpyro

    from numpyro.infer import MCMC, NUTS

    logp_fn = get_jaxified_logp(model, negative_logp=False)

    nuts_kwargs.setdefault("adapt_step_size", True)
    nuts_kwargs.setdefault("adapt_mass_matrix", True)
    nuts_kwargs.setdefault("dense_mass", False)

    nuts_kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=target_accept,
        **nuts_kwargs,
    )

    pmap_numpyro = MCMC(
        nuts_kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        postprocess_fn=None,
        chain_method=chain_method,
        progress_bar=progressbar,
    )

    map_seed = jax.random.PRNGKey(random_seed)
    if chains > 1:
        map_seed = jax.random.split(map_seed, chains)

    pmap_numpyro.run(
        map_seed,
        init_params=initial_points,
        extra_fields=(
            "num_steps",
            "potential_energy",
            "energy",
            "adapt_state.step_size",
            "accept_prob",
            "diverging",
        ),
    )

    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)
    sample_stats = _numpyro_stats_to_dict(pmap_numpyro)
    return raw_mcmc_samples, sample_stats, numpyro


def sample_jax_nuts(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: RandomState | None = None,
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    jitter: bool = True,
    model: Model | None = None,
    var_names: Sequence[str] | None = None,
    nuts_kwargs: dict | None = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] | None = None,
    postprocessing_chunks=None,
    idata_kwargs: dict | None = None,
    compute_convergence_checks: bool = True,
    nuts_sampler: Literal["numpyro", "blackjax"],
) -> az.InferenceData:
    """
    Draw samples from the posterior using a jax NUTS method.

    Parameters
    ----------
    draws : int, default 1000
        The number of samples to draw. The number of tuned samples are discarded by
        default.
    tune : int, default 1000
        Number of iterations to tune. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number
        specified in the ``draws`` argument.  Tuned samples are discarded.
    chains : int, default 4
        The number of chains to sample.
    target_accept : float in [0, 1].
        The step size is tuned such that we approximate this acceptance rate. Higher
        values like 0.9 or 0.95 often work better for problematic posteriors.
    random_seed : int, RandomState or Generator, optional
        Random seed used by the sampling steps.
    initvals: StartDict or Sequence[Optional[StartDict]], optional
        Initial values for random variables provided as a dictionary (or sequence of
        dictionaries) mapping the random variable (by name or reference) to desired
        starting values.
    jitter: bool, default True
        If True, add jitter to initial points.
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    nuts_kwargs : dict, optional
        Keyword arguments for the underlying nuts sampler
    progressbar : bool, default True
        If True, display a progressbar while sampling
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "parallel", and
        "vectorized".
    postprocessing_backend : Optional[Literal["cpu", "gpu"]], default None,
        Specify how postprocessing should be computed. gpu or cpu
    postprocessing_vectorize : Literal["vmap", "scan"], default "scan"
        How to vectorize the postprocessing: vmap or sequential scan
    postprocessing_chunks : None
        This argument is deprecated
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz.from_dict`. It also accepts a boolean as
        value for the ``log_likelihood`` key to indicate that the pointwise log
        likelihood should not be included in the returned object. Values for
        ``observed_data``, ``constant_data``, ``coords``, and ``dims`` are inferred from
        the ``model`` argument if not provided in ``idata_kwargs``. If ``coords`` and
        ``dims`` are provided, they are used to update the inferred dictionaries.
    compute_convergence_checks : bool, default True
        If True, compute ess and rhat values and warn if they indicate potential sampling issues.
    nuts_sampler : Literal["numpyro", "blackjax"]
        Nuts sampler library to use - do not change - use sample_numpyro_nuts or
        sample_blackjax_nuts as appropriate

    Returns
    -------
    InferenceData
        ArviZ ``InferenceData`` object that contains the posterior samples, together
        with their respective sample stats and pointwise log likeihood values (unless
        skipped with ``idata_kwargs``).
    """
    if postprocessing_chunks is not None:
        import warnings

        warnings.warn(
            "postprocessing_chunks is deprecated due to being unstable, "
            "using postprocessing_vectorize='scan' instead",
            DeprecationWarning,
        )

    if postprocessing_vectorize is not None:
        import warnings

        warnings.warn(
            'postprocessing_vectorize={"scan", "vmap"} will be removed in a future release.',
            FutureWarning,
        )
    else:
        postprocessing_vectorize = "vmap"

    model = modelcontext(model)

    if var_names is not None:
        filtered_var_names = [v for v in model.unobserved_value_vars if v.name in var_names]
    else:
        filtered_var_names = model.unobserved_value_vars

    if nuts_kwargs is None:
        nuts_kwargs = {}
    else:
        nuts_kwargs = nuts_kwargs.copy()

    vars_to_sample = list(
        get_default_varnames(filtered_var_names, include_transformed=keep_untransformed)
    )

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    initial_points = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
        jitter=jitter,
    )

    if nuts_sampler == "numpyro":
        sampler_fn = _sample_numpyro_nuts
    elif nuts_sampler == "blackjax":
        sampler_fn = _sample_blackjax_nuts
    else:
        raise ValueError(f"{nuts_sampler=} not recognized")

    tic1 = datetime.now()
    raw_mcmc_samples, sample_stats, library = sampler_fn(
        model=model,
        target_accept=target_accept,
        tune=tune,
        draws=draws,
        chains=chains,
        chain_method=chain_method,
        progressbar=progressbar,
        random_seed=random_seed,
        initial_points=initial_points,
        nuts_kwargs=nuts_kwargs,
    )
    tic2 = datetime.now()

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            postprocessing_vectorize=postprocessing_vectorize,
        )
    else:
        log_likelihood = None

    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn,
        raw_mcmc_samples,
        postprocessing_backend=postprocessing_backend,
        postprocessing_vectorize=postprocessing_vectorize,
        donate_samples=True,
    )
    del raw_mcmc_samples
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    attrs = {
        "sampling_time": (tic2 - tic1).total_seconds(),
        "tuning_steps": tune,
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    if "coords" in idata_kwargs:
        coords.update(idata_kwargs.pop("coords"))
    if "dims" in idata_kwargs:
        dims.update(idata_kwargs.pop("dims"))

    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=sample_stats,
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=library),
        posterior_attrs=make_attrs(attrs, library=library),
    )
    az_trace = to_trace(posterior=mcmc_samples, **idata_kwargs)

    if compute_convergence_checks:
        warns = run_convergence_checks(az_trace, model)
        log_warnings(warns)

    return az_trace


sample_numpyro_nuts = partial(sample_jax_nuts, nuts_sampler="numpyro")
sample_blackjax_nuts = partial(sample_jax_nuts, nuts_sampler="blackjax")
