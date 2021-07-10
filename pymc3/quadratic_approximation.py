#   Copyright 2020 The PyMC Developers
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

"""Functions for Quadratic Approximation."""

import numpy as np
import scipy
import arviz as az

__all__ = [
    "quadratic_approximation"
]

from pymc3.tuning import find_MAP, find_hessian


def quadratic_approximation(vars, n_chains=2, n_samples=10_000):
    """ Finds the quadratic approximation to the posterior, also known as the Laplace approximation.

    NOTE: The quadratic approximation only works well for unimodal and roughly symmetrical posteriors of continuous variables.
    The usual MCMC convergence and mixing statistics (e.g. R-hat, ESS) will NOT tell you anything about how well this approximation fits your actual (unknown) posterior, indeed they'll always be extremely nice since all "chains" are sampling from exactly the same distribution, the posterior quadratic approximation.
    Use at your own risk.

    See Chapter 4 of "Bayesian Data Analysis" 3rd edition for background.

    Returns an arviz.InferenceData object for compatibility by sampling from the approximated quadratic posterior. Note these are NOT MCMC samples, so the notion of chains is meaningless, and is only included for downstream compatibility with Arviz.

    Also returns the exact posterior approximation as a scipy.stats.multivariate_normal distribution.

    Parameters
    ----------
    vars: list
        List of variables to approximate the posterior for.
    n_chains: int
        How many chains to simulate.
    n_samples: int
        How many samples to sample from the approximate posterior for each chain.

    Returns
    -------
    (arviz.InferenceData, scipy.stats.multivariate_normal):
        InferenceData with samples from the approximate posterior, multivariate normal posterior approximation

    """
    map = find_MAP(vars=vars)
    H = find_hessian(map, vars=vars)
    cov = np.linalg.inv(H)
    mean = np.concatenate([np.atleast_1d(map[v.name]) for v in vars])
    posterior = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    draws = posterior.rvs((n_chains, n_samples))
    samples = {}
    i = 0
    for v in vars:
        var_size = map[v.name].size
        samples[v.name] = draws[:, :, i:i + var_size].squeeze()
        i += var_size
    return az.convert_to_inference_data(samples), posterior
