# pylint: skip-file
import os
import re
import sys
import warnings

from typing import Callable, List, Optional

xla_flags = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(r"--xla_force_host_platform_device_count=.+\s", "", xla_flags).split()
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={100}"] + xla_flags)

import aesara.tensor as at
import arviz as az
import jax
import numpy as np
import pandas as pd

from aeppl.logprob import CheckParameterValue
from aesara.compile import SharedVariable, Supervisor, mode
from aesara.graph.basic import clone_replace, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.link.jax.dispatch import jax_funcify
from aesara.raise_op import Assert
from aesara.tensor import TensorVariable

from pymc import Model, modelcontext
from pymc.backends.arviz import find_observations
from pymc.util import get_default_varnames

warnings.warn("This module is experimental.")


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
    mode.JAX.optimizer.optimize(fgraph)

    # We now jaxify the optimized fgraph
    return jax_funcify(fgraph)


def get_jaxified_logp(model: Model) -> Callable:

    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[model.logpt()])

    def logp_fn_wrap(x):
        # NumPyro expects a scalar potential with the opposite sign of model.logpt
        res = logp_fn(*x)[0]
        return -res

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


def _get_log_likelihood(model, samples):
    """Compute log-likelihood for all observations"""
    data = {}
    for v in model.observed_RVs:
        v_elemwise_logpt = model.logpt(v, sum=False)
        jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=v_elemwise_logpt)
        result = jax.jit(jax.vmap(jax.vmap(jax_fn)))(*samples)[0]
        data[v.name] = result
    return data


def sample_numpyro_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    var_names=None,
    progress_bar=True,
    keep_untransformed=False,
    chain_method="parallel",
    idata_kwargs=None,
):
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

    if hasattr(model, "RV_dims"):
        dims = {
            var_name: [dim for dim in dims if dim is not None]
            for var_name, dims in model.RV_dims.items()
        }
    else:
        dims = {}

    tic1 = pd.Timestamp.now()
    print("Compiling...", file=sys.stdout)

    rv_names = [rv.name for rv in model.value_vars]
    initial_point = model.compute_initial_point()
    init_state = [initial_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    logp_fn = get_jaxified_logp(model)

    nuts_kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=target_accept,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
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

    tic2 = pd.Timestamp.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    seed = jax.random.PRNGKey(random_seed)
    map_seed = jax.random.split(seed, chains)

    if chains == 1:
        init_params = init_state
        map_seed = seed
    else:
        init_params = init_state_batched

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

    tic3 = pd.Timestamp.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    mcmc_samples = {}
    for v in vars_to_sample:
        jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[v])
        result = jax.vmap(jax.vmap(jax_fn))(*raw_mcmc_samples)[0]
        mcmc_samples[v.name] = result

    tic4 = pd.Timestamp.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", True):
        log_likelihood = _get_log_likelihood(model, raw_mcmc_samples)
    else:
        log_likelihood = None

    posterior = mcmc_samples
    az_trace = az.from_dict(
        posterior=posterior,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        sample_stats=_sample_stats_to_xarray(pmap_numpyro),
        coords=coords,
        dims=dims,
        attrs={"sampling_time": (tic3 - tic2).total_seconds()},
        **idata_kwargs,
    )

    return az_trace
