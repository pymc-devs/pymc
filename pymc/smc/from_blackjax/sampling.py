#!/usr/bin/env python
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
#!/usr/bin/env python
# coding: utf-8
import logging
import time

from typing import Callable, NamedTuple, Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from pymc import modelcontext, to_inference_data
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pymc.sampling.jax import get_jaxified_graph
from pymc.smc.from_blackjax.kernels import (
    build_smc_with_hmc_kernel,
    build_smc_with_nuts_kernel,
)
from pymc.smc.kernels import initialize_population, var_map_from_model
from pymc.util import RandomState, _get_seeds_per_chain

log = logging.getLogger(__name__)


def sample_with_blackjax_smc(
    n_particles,
    random_seed: RandomState = None,
    kernel: str = "HMC",
    target_ess: float = 0.8,
    num_mcmc_steps: int = 10,
    inner_kernel_params: Optional[dict] = None,
    model=None,
    iterations_to_diagnose: int = 100,
):
    """Samples using BlackJax's implementation of Sequential Monte Carlo.
       A summary of the algorithm is:

     1. Initialize :math:`\beta` at zero and stage at zero.
     2. Generate N samples :math:`S_{\beta}` from the prior (because when :math `\beta = 0` the
        tempered posterior is the prior).
     3. Increase :math:`\beta` in order to make the effective sample size equal some predefined
        value (target_ess)
     4. Compute a set of N importance weights W. The weights are computed as the ratio of the
        likelihoods of a sample at stage i+1 and stage i.
     5. Obtain :math:`S_{w}` by re-sampling according to W.
     6. Run N independent MCMC chains, starting each one from a different sample
        in :math:`S_{w}`. For that, set the kernel and inner_kernel_params.
     7. The N chains are run for num_mcmc_steps each.
     8. Repeat from step 3 until :math:`\beta \\ge 1`.
     9. The final result is a collection of N samples from the posterior

    Parameters
    ----------
    n_particles
    random_seed
    kernel
    target_ess
    num_mcmc_steps
    inner_kernel_params:
    model: PyMC model to sample from
    iterations_to_diagnose: Number of iterations to generate diagnosis for.

    Returns
    -------
    An Arviz Inference data.

    """

    model = modelcontext(model)
    random_seed = np.random.default_rng(seed=random_seed)

    if inner_kernel_params is None:
        inner_kernel_params = {}
    if kernel == "HMC":
        build_sampler = build_smc_with_hmc_kernel
    elif kernel == "NUTS":
        build_sampler = build_smc_with_nuts_kernel
    else:
        raise ValueError(f"Valid kernels are 'HMC' and 'NUTS'")

    log.info(
        f"Will only diagnose the first {iterations_to_diagnose} SMC iterations,"
        f"this number can be increased by setting iterations_to_diagnose parameter"
        f" in sample_with_blackjax_smc"
    )

    key = jax.random.PRNGKey(_get_seeds_per_chain(random_seed, 1)[0])

    key, initial_particles_key, iterations_key = jax.random.split(key, 3)

    initial_particles = blackjax_particles_from_pymc_population(
        model, initialize_population(model, n_particles, random_seed)
    )

    var_map = var_map_from_model(
        model, model.initial_point(random_seed=random_seed.integers(2**30))
    )

    sampler = build_sampler(
        prior_log_prob=get_jaxified_logprior(model),
        loglikelihood=get_jaxified_loglikelihood(model),
        posterior_dimensions=sum([var_map[k][1] for k in var_map]),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
        kernel_parameters=inner_kernel_params,
    )

    start = time.time()
    total_iterations, particles, diagnosis = inference_loop(
        iterations_key,
        sampler.init(initial_particles),
        sampler,
        iterations_to_diagnose,
        n_particles,
    )
    end = time.time()
    running_time = end - start

    inference_data = arviz_from_particles(model, particles)

    add_to_inference_data(
        inference_data,
        n_particles,
        target_ess,
        num_mcmc_steps,
        kernel,
        diagnosis,
        total_iterations,
        iterations_to_diagnose,
        inner_kernel_params,
        running_time,
    )

    if total_iterations < iterations_to_diagnose:
        log.warning(
            f"Only the first {iterations_to_diagnose} were included in diagnosed quantities out of {total_iterations}."
        )

    return inference_data


def arviz_from_particles(model, particles):
    """
    Given Particles in Blackjax format,
    builds an Arviz Inference Data object.
    In order to do so in a consistent way,
    particles are assumed to be encoded in
    model.value_vars order.

    Parameters
    ----------
    model: Pymc Model
    particles: output of Blackjax SMC.


    Returns an Arviz Inference Data Object
    -------
    """
    n_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    by_varname = {k.name: v.squeeze()[np.newaxis, :] for k, v in zip(model.value_vars, particles)}
    varnames = [v.name for v in model.value_vars]
    with model:
        strace = NDArray(name=model.name)
        strace.setup(n_particles, 0)
    for particle_index in range(0, n_particles):
        strace.record(point={k: by_varname[k][0][particle_index] for k in varnames})
        multitrace = MultiTrace((strace,))
    return to_inference_data(multitrace, log_likelihood=False)


class SMCDiagnostics(NamedTuple):
    """
    A Jax-compilable object to track
    quantities of interest of an SMC run.
    Note that initial_diagnosis and update_diagnosis
    must return copies and not modify in place for the class
    to be Jax Compilable, reason why they are static methods.
    """

    lmbda_evolution: jax.Array
    log_likelihood_increment_evolution: jax.Array
    ancestors_evolution: jax.Array
    weights_evolution: jax.Array

    @staticmethod
    def update_diagnosis(i, history, info, state):
        le, lli, ancestors, weights_evolution = history
        return SMCDiagnostics(
            le.at[i].set(state.lmbda),
            lli.at[i].set(info.log_likelihood_increment),
            ancestors.at[i].set(info.ancestors),
            weights_evolution.at[i].set(state.weights),
        )

    @staticmethod
    def initial_diagnosis(iterations_to_diagnose, n_particles):
        return SMCDiagnostics(
            jnp.zeros(iterations_to_diagnose),
            jnp.zeros(iterations_to_diagnose),
            jnp.zeros((iterations_to_diagnose, n_particles)),
            jnp.zeros((iterations_to_diagnose, n_particles)),
        )


def flatten_single_particle(particle):
    return jnp.hstack([v.squeeze() for v in particle])


def inference_loop(rng_key, initial_state, kernel, iterations_to_diagnose, n_particles):
    """
    SMC inference loop that keeps tracks of diagnosis quantities.
    """

    def cond(carry):
        i, state, _, _ = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k, previous_info = carry
        k, subk = jax.random.split(k, 2)
        state, info = kernel.step(subk, state)
        full_info = SMCDiagnostics.update_diagnosis(i, previous_info, info, state)

        return i + 1, state, k, full_info

    n_iter, final_state, _, diagnosis = jax.lax.while_loop(
        cond,
        one_step,
        (
            0,
            initial_state,
            rng_key,
            SMCDiagnostics.initial_diagnosis(iterations_to_diagnose, n_particles),
        ),
    )

    return n_iter, final_state.particles, diagnosis


def blackjax_particles_from_pymc_population(model, pymc_population):
    """
    Transforms a pymc population of particles into the format
    accepted by BlackJax. Particles must be a PyTree, each leave represents
    a variable from the posterior, being an array of size n_particles
    * the variable's dimensionality.
    Note that the order in which variables are stored in the Pytree
    must be the same order used to calculate the logprior and loglikelihood.

    Parameters
    ----------
    pymc_population : A dictionary with variables as keys, and arrays
    with samples as values.
    """

    order_of_vars = model.value_vars

    def _format(var):
        variable = pymc_population[var.name]
        if len(variable.shape) == 1:
            return variable[:, np.newaxis]
        else:
            return variable

    return [_format(var) for var in order_of_vars]


def add_to_inference_data(
    inference_data: az.InferenceData,
    n_particles: int,
    target_ess: float,
    num_mcmc_steps: int,
    kernel: str,
    diagnosis: SMCDiagnostics,
    total_iterations: int,
    iterations_to_diagnose: int,
    kernel_parameters: dict,
    running_time_seconds: float,
):
    """
    Adds several SMC parameters into the az.InferenceData result
    Parameters
    ----------
    inference_data: arviz object to add attributes to.
    n_particles: number of particles present in the result
    target_ess: target effective sampling size between SMC iterations, used
    to calculate the tempering exponent
    num_mcmc_steps: number of steps of the inner kernel when mutating particles
    kernel: string representing the kernel used to mutate particles
    diagnosis: SMCDiagnostics, containing quantities of interest for the full
    SMC run
    total_iterations: the total number of iterations executed by the sampler
    iterations_to_diagnose: the number of iterations represented in the diagnosed
    quantities
    kernel_parameters: dict parameters from the inner kernel used to mutate particles
    running_time_seconds: float sampling time
    """
    experiment_parameters = {
        "particles": n_particles,
        "target_ess": target_ess,
        "num_mcmc_steps": num_mcmc_steps,
        "iterations": total_iterations,
        "iterations_to_diagnose": iterations_to_diagnose,
        "sampler": f"Blackjax SMC with {kernel} kernel",
    }

    inference_data.posterior.attrs["lambda_evolution"] = np.array(diagnosis.lmbda_evolution)[
        :iterations_to_diagnose
    ]
    inference_data.posterior.attrs["log_likelihood_increments"] = np.array(
        diagnosis.log_likelihood_increment_evolution
    )[:iterations_to_diagnose]
    inference_data.posterior.attrs["ancestors_evolution"] = np.array(diagnosis.ancestors_evolution)[
        :iterations_to_diagnose
    ]
    inference_data.posterior.attrs["weights_evolution"] = np.array(diagnosis.weights_evolution)[
        :iterations_to_diagnose
    ]

    for k in experiment_parameters:
        inference_data.posterior.attrs[k] = experiment_parameters[k]

    for k in kernel_parameters:
        inference_data.posterior.attrs[k] = kernel_parameters[k]

    inference_data.posterior.attrs["running_time_seconds"] = running_time_seconds

    return inference_data


def get_jaxified_logprior(model) -> Callable:
    return get_jaxified_particles_fn(model, model.varlogp)


def get_jaxified_loglikelihood(model) -> Callable:
    return get_jaxified_particles_fn(model, model.datalogp)


def get_jaxified_particles_fn(model, graph_outputs):
    """
    Builds a Jaxified version of a value_vars function,
    that is applyable to Blackjax particles format.
    """
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[graph_outputs])

    def logp_fn_wrap(particles):
        return logp_fn(*[p.squeeze() for p in particles])[0]

    return logp_fn_wrap
