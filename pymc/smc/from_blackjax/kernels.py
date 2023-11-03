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
import blackjax

from blackjax.smc.resampling import systematic
from jax import numpy as jnp


def build_smc_with_hmc_kernel(
    prior_log_prob,
    loglikelihood,
    posterior_dimensions,
    target_ess,
    num_mcmc_steps,
    kernel_parameters,
):
    return build_blackjax_smc(
        prior_log_prob,
        loglikelihood,
        blackjax.mcmc.hmc,
        mcmc_parameters=dict(
            step_size=kernel_parameters["step_size"],
            inverse_mass_matrix=jnp.eye(posterior_dimensions),
            num_integration_steps=kernel_parameters["integration_steps"],
        ),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )


def build_smc_with_nuts_kernel(
    prior_log_prob,
    loglikelihood,
    posterior_dimensions,
    target_ess,
    num_mcmc_steps,
    kernel_parameters,
):
    return build_blackjax_smc(
        prior_log_prob,
        loglikelihood,
        blackjax.mcmc.nuts,
        mcmc_parameters=dict(
            step_size=kernel_parameters["step_size"],
            inverse_mass_matrix=jnp.eye(posterior_dimensions),
        ),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )


def build_blackjax_smc(
    prior_log_prob, loglikelihood, sampler_module, mcmc_parameters, target_ess, num_mcmc_steps
):
    return blackjax.adaptive_tempered_smc(
        prior_log_prob,
        loglikelihood,
        sampler_module.build_kernel(),
        sampler_module.init,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=systematic,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )
