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
    """Replace shared variables in graph by their constant values

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
    """Compile an PyTensor graph into an optimized JAX function"""

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


def _get_log_likelihood(model: Model, samples) -> Callable:
    """Generate function to compute log-likelihood for all observations"""
    elemwise_logp = model.logp(model.observed_RVs, sum=False)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=elemwise_logp)
    result = jax.vmap(jax_fn)(*samples)
    return {v.name: r for v, r in zip(model.observed_RVs, result)}


def _get_batched_jittered_initial_points(
    model: Model,
    chains: int,
    initvals: StartDict | Sequence[StartDict | None] | None,
    random_seed: RandomSeed,
    jitter: bool = True,
    jitter_max_retries: int = 10,
) -> np.ndarray | list[np.ndarray]:
    """Get jittered initial point in format expected by NumPyro MCMC kernel

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


@partial(jax.jit, donate_argnums=0)
def _set_tree(store, input, idx):
    """Update pytree of outputs - used for saving results of chunked sampling"""

    def update_fn(save, inp):
        starts = (save.shape[0], idx, *([0] * (len(save.shape) - 2)))
        return jax.lax.dynamic_update_slice(save, inp, starts)

    store = jax.tree.map(update_fn, store, input)
    return store


def _gen_arr(inp, nchunk):
    """Generate output array on cpu for chunked sampling"""
    shape = (inp.shape[0] * nchunk, *inp.shape[1:])
    return jnp.zeros(shape, dtype=inp.dtype, device=jax.devices("cpu")[0])


def _do_chunked_sampling(last_state, output, nchunk, nsteps, sample_fn, progressbar):
    """Run chunked sampling saving to output on the cpu"""
    for i in range(1, nchunk):
        if progressbar:
            logger.info("Sampling chunk %d of %d:" % (i + 1, nchunk))
        last_state, tmpout = sample_fn(last_state)
        output = _set_tree(
            output,
            jax.device_put(tmpout, jax.devices("cpu")[0]),
            nsteps * i,
        )
        del tmpout
    return last_state, output


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
    postprocess_fn,
    nuts_kwargs,
    num_chunks: int = 1,
) -> az.InferenceData:
    import blackjax

    from blackjax.adaptation.base import get_filter_adapt_info_fn

    # Adapted from numpyro
    if chain_method == "parallel":
        map_fn = jax.pmap
    elif chain_method == "vectorized":

        def map_fn(x):
            return jax.jit(jax.vmap(x))
    else:
        raise ValueError(
            "Only supporting the following methods to draw chains:" ' "parallel" or "vectorized"'
        )

    if chains == 1:
        initial_points = [np.stack(init_state) for init_state in zip(initial_points)]

    logprob_fn = get_jaxified_logp(model)

    s1, s2 = jax.random.split(jax.random.PRNGKey(random_seed))
    adapt_seed = jax.random.split(s1, chains)
    sample_seed = jax.random.split(s2, chains)

    algorithm_name = nuts_kwargs.pop("algorithm", "nuts")
    if algorithm_name == "nuts":
        algorithm = blackjax.nuts
    elif algorithm_name == "hmc":
        algorithm = blackjax.hmc
    else:
        raise ValueError("Only supporting 'nuts' or 'hmc' as algorithm to draw samples.")

    assert draws % num_chunks == 0
    nsteps = draws // num_chunks

    # Run adaptation
    adapt = blackjax.window_adaptation(
        algorithm=algorithm,
        logdensity_fn=logprob_fn,
        target_acceptance_rate=target_accept,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        progress_bar=progressbar,
        **nuts_kwargs,
    )

    @map_fn
    def run_adaptation(seed, init_position):
        return adapt.run(seed, init_position, num_steps=tune)

    (last_state, tuned_params), _ = run_adaptation(adapt_seed, initial_points)

    def _one_step(state, x, kernel):
        del x
        state, rng_key = state
        key, _skey = jax.random.split(rng_key)
        state, info = kernel(_skey, state)
        position = state.position
        stats = {
            "diverging": info.is_divergent,
            "energy": info.energy,
            "tree_depth": info.num_trajectory_expansions,
            "n_steps": info.num_integration_steps,
            "acceptance_rate": info.acceptance_rate,
            "lp": state.logdensity,
        }
        return (state, key), (position, stats)

    @map_fn
    @partial(jax.jit, donate_argnums=0)
    def _multi_step(state, imm, ss):
        start_state, key = state
        scan_fn = blackjax.progress_bar.gen_scan_fn(nsteps, progressbar)

        kernel = algorithm(logprob_fn, inverse_mass_matrix=imm, step_size=ss).step

        (last_state, key), (raw_samples, stats) = scan_fn(
            partial(_one_step, kernel=kernel), (start_state, key), jnp.arange(nsteps)
        )
        samples, log_likelihoods = postprocess_fn(raw_samples)
        return (last_state, key), ((samples, log_likelihoods), stats)

    sample_fn = partial(
        _multi_step, imm=tuned_params["inverse_mass_matrix"], ss=tuned_params["step_size"]
    )
    if progressbar:
        logger.info("Sampling chunk %d of %d:" % (1, num_chunks))
    (last_state, seed), (samples, stats) = sample_fn((last_state, sample_seed))
    if num_chunks == 1:
        return samples[0], stats, samples[1], blackjax

    output = _set_tree(
        jax.tree.map(jax.vmap(partial(_gen_arr, nchunk=num_chunks)), (samples, stats)),
        jax.device_put((samples, stats), jax.devices("cpu")[0]),
        0,
    )
    del samples, stats

    last_state, (all_samples, all_stats) = _do_chunked_sampling(
        (last_state, seed), output, num_chunks, nsteps, sample_fn, progressbar
    )
    return all_samples[0], all_stats, all_samples[1], blackjax


def _numpyro_stats_to_dict(posterior):
    """Extract sample_stats from NumPyro posterior."""
    extra_fields = posterior.get_extra_fields(group_by_chain=True)
    data = {
        "lp": extra_fields["potential_energy"],
        "step_size": extra_fields["adapt_state.step_size"],
        "n_steps": extra_fields["num_steps"],
        "acceptance_rate": extra_fields["accept_prob"],
        "tree_depth": jnp.log2(extra_fields["num_steps"]).astype(int) + 1,
        "energy": extra_fields["energy"],
        "diverging": extra_fields["diverging"],
    }
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
    postprocess_fn,
    nuts_kwargs: dict[str, Any],
    num_chunks: int = 1,
):
    import numpyro

    from numpyro.infer import MCMC, NUTS

    assert draws % num_chunks == 0
    nsteps = draws // num_chunks

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
        num_samples=nsteps,
        num_chains=chains,
        postprocess_fn=None,
        chain_method=chain_method,
        progress_bar=progressbar,
    )

    extra_fields = (
        "num_steps",
        "potential_energy",
        "energy",
        "adapt_state.step_size",
        "accept_prob",
        "diverging",
    )

    vmap_postprocess = jax.jit(jax.vmap(postprocess_fn))

    key = jax.random.PRNGKey(random_seed)
    if progressbar:
        logger.info("Sampling chunk %d of %d:" % (1, num_chunks))
    pmap_numpyro.run(key, init_params=initial_points, extra_fields=extra_fields)
    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)
    stats = _numpyro_stats_to_dict(pmap_numpyro)
    samples = vmap_postprocess(raw_mcmc_samples)

    if num_chunks == 1:
        return samples[0], stats, samples[1], numpyro

    def sample_chunk(state):
        pmap_numpyro.post_warmup_state = state
        pmap_numpyro.run(pmap_numpyro.post_warmup_state.rng_key, extra_fields=extra_fields)

        raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)
        sample_stats = _numpyro_stats_to_dict(pmap_numpyro)
        mcmc_samples, likelihoods = vmap_postprocess(raw_mcmc_samples)
        return pmap_numpyro.last_state, ((mcmc_samples, likelihoods), sample_stats)

    output = _set_tree(
        jax.tree.map(jax.vmap(partial(_gen_arr, nchunk=num_chunks)), (samples, stats)),
        jax.device_put((samples, stats), jax.devices("cpu")[0]),
        0,
    )
    del samples, stats

    _, (all_samples, all_stats) = _do_chunked_sampling(
        pmap_numpyro.last_state,
        output,
        num_chunks,
        nsteps,
        sample_chunk,
        progressbar,
    )

    return all_samples[0], all_stats, all_samples[1], numpyro


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
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,  # Note unused
    postprocessing_vectorize: Literal["vmap", "scan"] | None = None,  # Note unused
    postprocessing_chunks=None,  # Note unused
    idata_kwargs: dict | None = None,
    compute_convergence_checks: bool = True,
    nuts_sampler: Literal["numpyro", "blackjax"],
    num_chunks: int = 1,
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
    num_chunks : int
        Splits sampling into multiple chunks and collects them on the cpu.  Reduces gpu memory
        usage when sampling.  There is no benefit when sampling on the cpu.

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

    if postprocessing_backend is not None:
        import warnings

        warnings.warn(
            "postprocessing_backend={'cpu', 'gpu'} will be removed in a future release, "
            "postprocessing will be done on sampling device in the future.  If device memory "
            "consumption is an issue please use num_chunks to reduce consumption.",
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

    nuts_kwargs = {} if nuts_kwargs is None else nuts_kwargs.copy()
    idata_kwargs = {} if idata_kwargs is None else idata_kwargs.copy()

    vars_to_sample = list(
        get_default_varnames(filtered_var_names, include_transformed=keep_untransformed)
    )

    log_likelihood_fn = partial(_get_log_likelihood, model)
    transform_jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)

    def postprocess_fn(samples):
        mcmc_samples, likelihoods = None, None
        if idata_kwargs.pop("log_likelihood", False):
            likelihoods = log_likelihood_fn(samples)
        result = jax.vmap(transform_jax_fn)(*samples)
        mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}
        return mcmc_samples, likelihoods

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

    current_backend = jax.default_backend()
    if postprocessing_backend is not None and current_backend != postprocessing_backend:

        def process_fn(x):
            return x, None
    else:
        process_fn = postprocess_fn

    tic1 = datetime.now()
    mcmc_samples, sample_stats, log_likelihood, library = sampler_fn(
        model=model,
        target_accept=target_accept,
        tune=tune,
        draws=draws,
        chains=chains,
        chain_method=chain_method,
        progressbar=progressbar,
        random_seed=random_seed,
        initial_points=initial_points,
        postprocess_fn=process_fn,
        nuts_kwargs=nuts_kwargs,
        num_chunks=num_chunks,
    )

    if postprocessing_backend is not None and current_backend != postprocessing_backend:
        mcmc_samples = jax.device_put(mcmc_samples, jax.devices(postprocessing_backend)[0])
        sample_stats = jax.device_put(sample_stats, jax.devices(postprocessing_backend)[0])
        mcmc_samples, log_likelihood = jax.jit(jax.vmap(postprocess_fn), donate_argnums=0)(
            mcmc_samples
        )

    tic2 = datetime.now()

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
