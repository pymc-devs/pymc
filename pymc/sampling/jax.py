import os
import re
import sys
import warnings

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from pymc.initial_point import StartDict
from pymc.sampling.mcmc import _init_jitter

xla_flags = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(r"--xla_force_host_platform_device_count=.+\s", "", xla_flags).split()
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={100}"] + xla_flags)

from datetime import datetime

import aesara.tensor as at
import arviz as az
import jax
import numpy as np

from aesara.compile import SharedVariable, Supervisor, mode
from aesara.graph.basic import clone_replace, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.link.jax.dispatch import jax_funcify
from aesara.raise_op import Assert
from aesara.tensor import TensorVariable
from aesara.tensor.shape import SpecifyShape
from arviz.data.base import make_attrs

from pymc import Model, modelcontext
from pymc.backends.arviz import find_constants, find_observations
from pymc.logprob.utils import CheckParameterValue
from pymc.util import RandomSeed, _get_seeds_per_chain, get_default_varnames

warnings.warn("This module is experimental.")


__all__ = (
    "get_jaxified_graph",
    "get_jaxified_logp",
    "sample_blackjax_nuts",
    "sample_numpyro_nuts",
)


@jax_funcify.register(Assert)
@jax_funcify.register(CheckParameterValue)
@jax_funcify.register(SpecifyShape)
def jax_funcify_Assert(op, **kwargs):
    # Jax does not allow assert whose values aren't known during JIT compilation
    # within it's JIT-ed code. Hence we need to make a simple pass through
    # version of the Assert Op.
    # https://github.com/google/jax/issues/2273#issuecomment-589098722
    def assert_fn(value, *inps):
        return value

    return assert_fn


def _replace_shared_variables(graph: List[TensorVariable]) -> List[TensorVariable]:
    """Replace shared variables in graph by their constant values

    Raises
    ------
    ValueError
        If any shared variable contains default_updates
    """

    shared_variables = [var for var in graph_inputs(graph) if isinstance(var, SharedVariable)]

    if any(hasattr(var, "default_update") for var in shared_variables):
        raise ValueError(
            "Graph contains shared variables with default_update which cannot "
            "be safely replaced."
        )

    replacements = {var: at.constant(var.get_value(borrow=True)) for var in shared_variables}

    new_graph = clone_replace(graph, replace=replacements)
    return new_graph


def get_jaxified_graph(
    inputs: Optional[List[TensorVariable]] = None,
    outputs: Optional[List[TensorVariable]] = None,
) -> List[TensorVariable]:
    """Compile an Aesara graph into an optimized JAX function"""

    graph = _replace_shared_variables(outputs)

    fgraph = FunctionGraph(inputs=inputs, outputs=graph, clone=True)
    # We need to add a Supervisor to the fgraph to be able to run the
    # JAX sequential optimizer without warnings. We made sure there
    # are no mutable input variables, so we only need to check for
    # "destroyers". This should be automatically handled by Aesara
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


def _get_log_likelihood(model: Model, samples, backend=None) -> Dict:
    """Compute log-likelihood for all observations"""
    elemwise_logp = model.logp(model.observed_RVs, sum=False)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=elemwise_logp)
    result = jax.vmap(jax.vmap(jax_fn))(*jax.device_put(samples, jax.devices(backend)[0]))
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
    initial_points = [list(initial_point.values()) for initial_point in initial_points]
    if chains == 1:
        initial_points = initial_points[0]
    else:
        initial_points = [np.stack(init_state) for init_state in zip(*initial_points)]
    return initial_points


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
        logprob_fn=logprob_fn,
        num_steps=tune,
        target_acceptance_rate=target_accept,
    )
    last_state, kernel, _ = adapt.run(seed, init_position)

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
    random_seed: Optional[RandomSeed] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Optional[str] = None,
    idata_kwargs: Optional[Dict[str, Any]] = None,
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
    postprocessing_backend : str, optional
        Specify how postprocessing should be computed. gpu or cpu
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

    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.named_vars_to_dims.items()
    }

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
    potential_energy = states.potential_energy
    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = jax.vmap(jax.vmap(jax_fn))(
        *jax.device_put(raw_mcmc_samples, jax.devices(postprocessing_backend)[0])
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}
    mcmc_stats = _blackjax_stats_to_dict(stats, potential_energy)
    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", bool(model.observed_RVs)):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model, raw_mcmc_samples, backend=postprocessing_backend
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    posterior = mcmc_samples
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
    az_trace = to_trace(posterior=posterior, **idata_kwargs)

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
    random_seed: Optional[RandomSeed] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    progress_bar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Optional[str] = None,
    idata_kwargs: Optional[Dict] = None,
    nuts_kwargs: Optional[Dict] = None,
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
    progress_bar : bool, default True
        Whether or not to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "sequential",
        "parallel", and "vectorized".
    postprocessing_backend : Optional[str]
        Specify how postprocessing should be computed. gpu or cpu
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

    import numpyro

    from numpyro.infer import MCMC, NUTS

    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.named_vars_to_dims.items()
    }

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
        progress_bar=progress_bar,
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
    result = jax.vmap(jax.vmap(jax_fn))(
        *jax.device_put(raw_mcmc_samples, jax.devices(postprocessing_backend)[0])
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", bool(model.observed_RVs)):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model, raw_mcmc_samples, backend=postprocessing_backend
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    posterior = mcmc_samples
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
    az_trace = to_trace(posterior=posterior, **idata_kwargs)
    return az_trace
