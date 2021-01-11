# pylint: skip-file
import os
import re
import warnings

xla_flags = os.getenv("XLA_FLAGS", "").lstrip("--")
xla_flags = re.sub(r"xla_force_host_platform_device_count=.+\s", "", xla_flags).split()
os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(100)])

import arviz as az
import jax
import numpy as np
import pandas as pd
import theano.graph.fg

from theano.link.jax.jax_dispatch import jax_funcify

import pymc3 as pm

from pymc3 import modelcontext

warnings.warn("This module is experimental.")

# Disable C compilation by default
# theano.config.cxx = ""
# This will make the JAX Linker the default
# theano.config.mode = "JAX"


def sample_tfp_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    num_tuning_epoch=2,
    num_compute_step_size=500,
):
    import jax

    from tensorflow_probability.substrates import jax as tfp

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    @jax.pmap
    def _sample(init_state, seed):
        def gen_kernel(step_size):
            hmc = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=logp_fn_jax, step_size=step_size)
            return tfp.mcmc.DualAveragingStepSizeAdaptation(
                hmc, tune // num_tuning_epoch, target_accept_prob=target_accept
            )

        def trace_fn(_, pkr):
            return pkr.new_step_size

        def get_tuned_stepsize(samples, step_size):
            return step_size[-1] * jax.numpy.std(samples[-num_compute_step_size:])

        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        for i in range(num_tuning_epoch - 1):
            tuning_hmc = gen_kernel(step_size)
            init_samples, tuning_result, kernel_results = tfp.mcmc.sample_chain(
                num_results=tune // num_tuning_epoch,
                current_state=init_state,
                kernel=tuning_hmc,
                trace_fn=trace_fn,
                return_final_kernel_results=True,
                seed=seed,
            )

            step_size = jax.tree_multimap(get_tuned_stepsize, list(init_samples), tuning_result)
            init_state = [x[-1] for x in init_samples]

        # Run inference
        sample_kernel = gen_kernel(step_size)
        mcmc_samples, leapfrog_num = tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune // num_tuning_epoch,
            current_state=init_state,
            kernel=sample_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.leapfrogs_taken,
            seed=seed,
        )

        return mcmc_samples, leapfrog_num

    print("Compiling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrog_num = _sample(init_state_batched, map_seed)

    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace  # , leapfrog_num, tic3 - tic2


def sample_numpyro_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    progress_bar=True,
):
    from numpyro.infer import MCMC, NUTS

    from pymc3 import modelcontext

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    @jax.jit
    def _sample(current_state, seed):
        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        nuts_kernel = NUTS(
            potential_fn=lambda x: -logp_fn_jax(*x),
            # model=model,
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

        pmap_numpyro.run(seed, init_params=current_state, extra_fields=("num_steps",))
        samples = pmap_numpyro.get_samples(group_by_chain=True)
        leapfrogs_taken = pmap_numpyro.get_extra_fields(group_by_chain=True)["num_steps"]
        return samples, leapfrogs_taken

    print("Compiling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrogs_taken = _sample(init_state_batched, map_seed)
    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace  # , leapfrogs_taken, tic3 - tic2
