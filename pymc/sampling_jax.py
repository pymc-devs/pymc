# pylint: skip-file
import os
import re
import sys
import warnings

from typing import Callable, List

from aesara.graph import optimize_graph
from aesara.tensor import TensorVariable

xla_flags = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(r"--xla_force_host_platform_device_count=.+\s", "", xla_flags).split()
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={100}"] + xla_flags)

import aesara.tensor as at
import arviz as az
import jax
import numpy as np
import pandas as pd

from aesara.assert_op import Assert
from aesara.compile import SharedVariable
from aesara.graph.basic import clone_replace, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.link.jax.dispatch import jax_funcify

from pymc import Model, modelcontext
from pymc.aesaraf import compile_rv_inplace

warnings.warn("This module is experimental.")


@jax_funcify.register(Assert)
def jax_funcify_Assert(op, **kwargs):
    # Jax does not allow assert whose values aren't known during JIT compilation
    # within it's JIT-ed code. Hence we need to make a simple pass through
    # version of the Assert Op.
    # https://github.com/google/jax/issues/2273#issuecomment-589098722
    def assert_fn(value, *inps):
        return value

    return assert_fn


def replace_shared_variables(graph: List[TensorVariable]) -> List[TensorVariable]:
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


def get_jaxified_logp(model: Model) -> Callable:
    """Compile model.logpt into an optimized jax function"""

    logpt = replace_shared_variables([model.logpt])[0]

    logpt_fgraph = FunctionGraph(outputs=[logpt], clone=False)
    optimize_graph(logpt_fgraph, include=["fast_run"], exclude=["cxx_only", "BlasOpt"])

    # We now jaxify the optimized fgraph
    logp_fn = jax_funcify(logpt_fgraph)

    if isinstance(logp_fn, (list, tuple)):
        # This handles the new JAX backend, which always returns a tuple
        logp_fn = logp_fn[0]

    def logp_fn_wrap(x):
        res = logp_fn(*x)

        if isinstance(res, (list, tuple)):
            # This handles the new JAX backend, which always returns a tuple
            res = res[0]

        # Jax expects a potential with the opposite sign of model.logpt
        return -res

    return logp_fn_wrap


def sample_numpyro_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    progress_bar=True,
    keep_untransformed=False,
):
    from numpyro.infer import MCMC, NUTS

    model = modelcontext(model)

    tic1 = pd.Timestamp.now()
    print("Compiling...", file=sys.stdout)

    rv_names = [rv.name for rv in model.value_vars]
    init_state = [model.initial_point[rv_name] for rv_name in rv_names]
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
        chain_method="parallel",
        progress_bar=progress_bar,
    )

    tic2 = pd.Timestamp.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    seed = jax.random.PRNGKey(random_seed)
    map_seed = jax.random.split(seed, chains)

    pmap_numpyro.run(map_seed, init_params=init_state_batched, extra_fields=("num_steps",))
    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)

    tic3 = pd.Timestamp.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    mcmc_samples = []
    for i, (value_var, raw_samples) in enumerate(zip(model.value_vars, raw_mcmc_samples)):
        raw_samples = at.constant(np.asarray(raw_samples))

        rv = model.values_to_rvs[value_var]
        transform = getattr(value_var.tag, "transform", None)

        if transform is not None:
            # TODO: This will fail when the transformation depends on another variable
            #  such as in interval transform with RVs as edges
            trans_samples = transform.backward(raw_samples, *rv.owner.inputs)
            trans_samples.name = rv.name
            mcmc_samples.append(trans_samples)

            if keep_untransformed:
                raw_samples.name = value_var.name
                mcmc_samples.append(raw_samples)
        else:
            raw_samples.name = rv.name
            mcmc_samples.append(raw_samples)

    mcmc_varnames = [var.name for var in mcmc_samples]
    mcmc_samples = compile_rv_inplace(
        [],
        mcmc_samples,
        mode="JAX",
    )()

    tic4 = pd.Timestamp.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    posterior = {k: v for k, v in zip(mcmc_varnames, mcmc_samples)}
    az_trace = az.from_dict(posterior=posterior)

    return az_trace
