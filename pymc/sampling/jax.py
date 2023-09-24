#   Copyright 2023 The PyMC Developers
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
import os
import re
import sys

from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

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
from pymc.util import (
    RandomSeed,
    RandomState,
    _get_seeds_per_chain,
    get_default_varnames,
)

xla_flags_env = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(r"--xla_force_host_platform_device_count=.+\s", "", xla_flags_env).split()
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={100}"] + xla_flags)

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


def _replace_shared_variables(graph: List[TensorVariable]) -> List[TensorVariable]:
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
    inputs: Optional[List[TensorVariable]] = None,
    outputs: Optional[List[TensorVariable]] = None,
) -> List[TensorVariable]:
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


# Adopted from arviz numpyro extractor
def _sample_stats_to_xarray(posterior):
    """Extract sample_stats from NumPyro posterior."""
    rename_key = {
        "potential_energy": "lp",
        "adapt_state.step_size": "step_size",
        "num_steps": "n_steps",
        "accept_prob": "acceptance_rate",
    }
    data = {}
    for stat, value in posterior.get_extra_fields(group_by_chain=True).items():
        if isinstance(value, (dict, tuple)):
            continue
        name = rename_key.get(stat, stat)
        value = value.copy()
        data[name] = value
        if stat == "num_steps":
            data["tree_depth"] = np.log2(value).astype(int) + 1
    return data


def _device_put(input, device: str):
    return jax.device_put(input, jax.devices(device)[0])


def _postprocess_samples(
    jax_fn: Callable,
    raw_mcmc_samples: List[TensorVariable],
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "scan",
) -> List[TensorVariable]:
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
        return jax.vmap(jax.vmap(jax_fn))(*_device_put(raw_mcmc_samples, postprocessing_backend))
    else:
        raise ValueError(f"Unrecognized postprocessing_vectorize: {postprocessing_vectorize}")


def _blackjax_stats_to_dict(sample_stats, potential_energy) -> Dict:
    """Extract compatible stats from blackjax NUTS sampler
    with PyMC/Arviz naming conventions.

    Parameters
    ----------
    sample_stats: NUTSInfo
        Blackjax NUTSInfo object containing sampler statistics
    potential_energy: ArrayLike
        Potential energy values of sampled positions.

    Returns
    -------
    Dict[str, ArrayLike]
        Dictionary of sampler statistics.
    """
    rename_key = {
        "is_divergent": "diverging",
        "energy": "energy",
        "num_trajectory_expansions": "tree_depth",
        "num_integration_steps": "n_steps",
        "acceptance_rate": "acceptance_rate",  # naming here is
        "acceptance_probability": "acceptance_rate",  # depending on blackjax version
    }
    converted_stats = {}
    converted_stats["lp"] = potential_energy
    for old_name, new_name in rename_key.items():
        value = getattr(sample_stats, old_name, None)
        if value is None:
            continue
        converted_stats[new_name] = value
    return converted_stats


def _get_log_likelihood(
    model: Model,
    samples,
    backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "scan",
) -> Dict:
    """Compute log-likelihood for all observations"""
    elemwise_logp = model.logp(model.observed_RVs, sum=False)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=elemwise_logp)
    result = _postprocess_samples(
        jax_fn, samples, backend, postprocessing_vectorize=postprocessing_vectorize
    )
    return {v.name: r for v, r in zip(model.observed_RVs, result)}


def _get_batched_jittered_initial_points(
    model: Model,
    chains: int,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]],
    random_seed: RandomSeed,
    jitter: bool = True,
    jitter_max_retries: int = 10,
) -> Union[np.ndarray, List[np.ndarray]]:
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


def _update_coords_and_dims(
    coords: Dict[str, Any], dims: Dict[str, Any], idata_kwargs: Dict[str, Any]
) -> None:
    """Update 'coords' and 'dims' dicts with values in 'idata_kwargs'."""
    if "coords" in idata_kwargs:
        coords.update(idata_kwargs.pop("coords"))
    if "dims" in idata_kwargs:
        dims.update(idata_kwargs.pop("dims"))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _blackjax_inference_loop(
    seed,
    init_position,
    logprob_fn,
    draws,
    tune,
    target_accept,
    algorithm=None,
):
    import blackjax

    if algorithm is None:
        algorithm = blackjax.nuts

    adapt = blackjax.window_adaptation(
        algorithm=algorithm,
        logdensity_fn=logprob_fn,
        target_acceptance_rate=target_accept,
    )
    (last_state, tuned_params), _ = adapt.run(seed, init_position, num_steps=tune)
    kernel = algorithm(logprob_fn, **tuned_params).step

    def inference_loop(rng_key, initial_state):
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, draws)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

        return states, infos

    return inference_loop(seed, last_state)


def sample_blackjax_nuts(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: Optional[RandomState] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "scan",
    idata_kwargs: Optional[Dict[str, Any]] = None,
    postprocessing_chunks=None,  # deprecated
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
    if postprocessing_chunks is not None:
        import warnings

        warnings.warn(
            "postprocessing_chunks is deprecated due to being unstable, "
            "using postprocessing_vectorize='scan' instead",
            DeprecationWarning,
        )
    import blackjax

    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    tic1 = datetime.now()
    print("Compiling...", file=sys.stdout)

    init_params = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
    )

    if chains == 1:
        init_params = [np.stack(init_state) for init_state in zip(init_params)]

    logprob_fn = get_jaxified_logp(model)

    seed = jax.random.PRNGKey(random_seed)
    keys = jax.random.split(seed, chains)

    get_posterior_samples = partial(
        _blackjax_inference_loop,
        logprob_fn=logprob_fn,
        tune=tune,
        draws=draws,
        target_accept=target_accept,
    )

    tic2 = datetime.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    # Adapted from numpyro
    if chain_method == "parallel":
        map_fn = jax.pmap
    elif chain_method == "vectorized":
        map_fn = jax.vmap
    else:
        raise ValueError(
            "Only supporting the following methods to draw chains:" ' "parallel" or "vectorized"'
        )

    states, stats = map_fn(get_posterior_samples)(keys, init_params)
    raw_mcmc_samples = states.position
    potential_energy = states.logdensity
    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn,
        raw_mcmc_samples,
        postprocessing_backend=postprocessing_backend,
        postprocessing_vectorize=postprocessing_vectorize,
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}
    mcmc_stats = _blackjax_stats_to_dict(stats, potential_energy)
    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            postprocessing_vectorize=postprocessing_vectorize,
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=mcmc_stats,
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=blackjax),
    )
    az_trace = to_trace(posterior=mcmc_samples, **idata_kwargs)

    return az_trace


def _numpyro_nuts_defaults() -> Dict[str, Any]:
    """Defaults parameters for Numpyro NUTS."""
    return {
        "adapt_step_size": True,
        "adapt_mass_matrix": True,
        "dense_mass": False,
    }


def _update_numpyro_nuts_kwargs(nuts_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Update default Numpyro NUTS parameters with new values."""
    nuts_kwargs_defaults = _numpyro_nuts_defaults()
    if nuts_kwargs is not None:
        nuts_kwargs_defaults.update(nuts_kwargs)
    return nuts_kwargs_defaults


def sample_numpyro_nuts(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: Optional[RandomState] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "scan",
    idata_kwargs: Optional[Dict] = None,
    nuts_kwargs: Optional[Dict] = None,
    postprocessing_chunks=None,
) -> az.InferenceData:
    """
    Draw samples from the posterior using the NUTS method from the ``numpyro`` library.

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
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    progressbar : bool, default True
        Whether or not to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "sequential",
        "parallel", and "vectorized".
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
    nuts_kwargs: dict, optional
        Keyword arguments for :func:`numpyro.infer.NUTS`.

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
    import numpyro

    from numpyro.infer import MCMC, NUTS

    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    tic1 = datetime.now()
    print("Compiling...", file=sys.stdout)

    init_params = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
    )

    logp_fn = get_jaxified_logp(model, negative_logp=False)

    nuts_kwargs = _update_numpyro_nuts_kwargs(nuts_kwargs)
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

    tic2 = datetime.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    map_seed = jax.random.PRNGKey(random_seed)
    if chains > 1:
        map_seed = jax.random.split(map_seed, chains)

    pmap_numpyro.run(
        map_seed,
        init_params=init_params,
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

    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn,
        raw_mcmc_samples,
        postprocessing_backend=postprocessing_backend,
        postprocessing_vectorize=postprocessing_vectorize,
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            postprocessing_vectorize=postprocessing_vectorize,
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=_sample_stats_to_xarray(pmap_numpyro),
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=numpyro),
    )
    az_trace = to_trace(posterior=mcmc_samples, **idata_kwargs)
    return az_trace
