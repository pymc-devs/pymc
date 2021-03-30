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

import logging
import multiprocessing as mp
import time
import warnings

from collections.abc import Iterable

import numpy as np

from pymc3.backends.base import MultiTrace
from pymc3.model import modelcontext
from pymc3.parallel_sampling import _cpu_count
from pymc3.smc.smc import SMC


def sample_smc(
    draws=2000,
    kernel="metropolis",
    n_steps=25,
    start=None,
    tune_steps=True,
    p_acc_rate=0.85,
    threshold=0.5,
    save_sim_data=False,
    save_log_pseudolikelihood=True,
    model=None,
    random_seed=-1,
    parallel=False,
    chains=None,
    cores=None,
):
    r"""
    Sequential Monte Carlo based sampling.

    Parameters
    ----------
    draws: int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 2000.
    kernel: str
        Kernel method for the SMC sampler. Available option are ``metropolis`` (default) and `ABC`.
        Use `ABC` for likelihood free inference together with a ``pm.Simulator``.
    n_steps: int
        The number of steps of each Markov Chain. If ``tune_steps == True`` ``n_steps`` will be used
        for the first stage and for the others it will be determined automatically based on the
        acceptance rate and `p_acc_rate`, the max number of steps is ``n_steps``.
    start: dict, or array of dict
        Starting point in parameter space. It should be a list of dict with length `chains`.
        When None (default) the starting point is sampled from the prior distribution.
    tune_steps: bool
        Whether to compute the number of steps automatically or not. Defaults to True
    p_acc_rate: float
        Used to compute ``n_steps`` when ``tune_steps == True``. The higher the value of
        ``p_acc_rate`` the higher the number of steps computed automatically. Defaults to 0.85.
        It should be between 0 and 1.
    threshold: float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
        It should be between 0 and 1.
    save_sim_data : bool
        Whether or not to save the simulated data. This parameter only works with the ABC kernel.
        The stored data corresponds to a samples from the posterior predictive distribution.
    save_log_pseudolikelihood : bool
        Whether or not to save the log pseudolikelihood values. This parameter only works with the
        ABC kernel. The stored data can be used to compute LOO or WAIC values. Computing LOO/WAIC
        values from log pseudolikelihood values is experimental.
    model: Model (optional if in ``with`` context)).
    random_seed: int
        random seed
    parallel: bool
        Distribute computations across cores if the number of cores is larger than 1.
        Defaults to False.
    cores : int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system, but at most 4.
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics. If ``None`` (default), then set to either ``cores`` or 2, whichever
        is larger.

    Notes
    -----
    SMC works by moving through successive stages. At each stage the inverse temperature
    :math:`\beta` is increased a little bit (starting from 0 up to 1). When :math:`\beta` = 0
    we have the prior distribution and when :math:`\beta` =1 we have the posterior distribution.
    So in more general terms we are always computing samples from a tempered posterior that we can
    write as:

    .. math::

        p(\theta \mid y)_{\beta} = p(y \mid \theta)^{\beta} p(\theta)

    A summary of the algorithm is:

     1. Initialize :math:`\beta` at zero and stage at zero.
     2. Generate N samples :math:`S_{\beta}` from the prior (because when :math `\beta = 0` the
        tempered posterior is the prior).
     3. Increase :math:`\beta` in order to make the effective sample size equals some predefined
        value (we use :math:`Nt`, where :math:`t` is 0.5 by default).
     4. Compute a set of N importance weights W. The weights are computed as the ratio of the
        likelihoods of a sample at stage i+1 and stage i.
     5. Obtain :math:`S_{w}` by re-sampling according to W.
     6. Use W to compute the mean and covariance for the proposal distribution, a MVNormal.
     7. For stages other than 0 use the acceptance rate from the previous stage to estimate
        `n_steps`.
     8. Run N independent Metropolis-Hastings (IMH) chains (each one of length `n_steps`),
        starting each one from a different sample in :math:`S_{w}`. Samples are IMH as the proposal
        mean is the of the previous posterior stage and not the current point in parameter space.
     9. Repeat from step 3 until :math:`\beta \ge 1`.
     10. The final result is a collection of N samples from the posterior.


    References
    ----------
    .. [Minson2013] Minson, S. E. and Simons, M. and Beck, J. L., (2013),
        Bayesian inversion for finite fault earthquake source models I- Theory and algorithm.
        Geophysical Journal International, 2013, 194(3), pp.1701-1726,
        `link <https://gji.oxfordjournals.org/content/194/3/1701.full>`__

    .. [Ching2007] Ching, J. and Chen, Y. (2007).
        Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class
        Selection, and Model Averaging. J. Eng. Mech., 10.1061/(ASCE)0733-9399(2007)133:7(816),
        816-832. `link <http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399
        %282007%29133:7%28816%29>`__
    """
    _log = logging.getLogger("pymc3")
    _log.info("Initializing SMC sampler...")

    model = modelcontext(model)
    if model.name:
        raise NotImplementedError(
            "The SMC implementation currently does not support named models. "
            "See https://github.com/pymc-devs/pymc3/pull/4365."
        )
    if cores is None:
        cores = _cpu_count()

    if chains is None:
        chains = max(2, cores)
    elif chains == 1:
        cores = 1

    _log.info(
        f"Sampling {chains} chain{'s' if chains > 1 else ''} "
        f"in {cores} job{'s' if cores > 1 else ''}"
    )

    if random_seed == -1:
        random_seed = None
    if chains == 1 and isinstance(random_seed, int):
        random_seed = [random_seed]
    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]
    if not isinstance(random_seed, Iterable):
        raise TypeError("Invalid value for `random_seed`. Must be tuple, list or int")

    if kernel.lower() == "abc":
        if len(model.observed_RVs) != 1:
            warnings.warn("SMC-ABC only works properly with models with one observed variable")
        if model.potentials:
            _log.info("Potentials will be added to the prior term")

    params = (
        draws,
        kernel,
        n_steps,
        start,
        tune_steps,
        p_acc_rate,
        threshold,
        save_sim_data,
        save_log_pseudolikelihood,
        model,
    )

    t1 = time.time()
    if parallel and chains > 1:
        loggers = [_log] + [None] * (chains - 1)
        pool = mp.Pool(cores)
        results = pool.starmap(
            sample_smc_int, [(*params, random_seed[i], i, loggers[i]) for i in range(chains)]
        )

        pool.close()
        pool.join()
    else:
        results = []
        for i in range(chains):
            results.append(sample_smc_int(*params, random_seed[i], i, _log))

    (
        traces,
        sim_data,
        log_marginal_likelihoods,
        log_pseudolikelihood,
        betas,
        accept_ratios,
        nsteps,
    ) = zip(*results)
    trace = MultiTrace(traces)
    trace.report._n_draws = draws
    trace.report._n_tune = 0
    trace.report.log_marginal_likelihood = np.array(log_marginal_likelihoods)
    trace.report.log_pseudolikelihood = log_pseudolikelihood
    trace.report.betas = betas
    trace.report.accept_ratios = accept_ratios
    trace.report.nsteps = nsteps
    trace.report._t_sampling = time.time() - t1

    if save_sim_data:
        return trace, {modelcontext(model).observed_RVs[0].name: np.array(sim_data)}
    else:
        return trace


def sample_smc_int(
    draws,
    kernel,
    n_steps,
    start,
    tune_steps,
    p_acc_rate,
    threshold,
    save_sim_data,
    save_log_pseudolikelihood,
    model,
    random_seed,
    chain,
    _log,
):
    """Run one SMC instance."""
    smc = SMC(
        draws=draws,
        kernel=kernel,
        n_steps=n_steps,
        start=start,
        tune_steps=tune_steps,
        p_acc_rate=p_acc_rate,
        threshold=threshold,
        save_sim_data=save_sim_data,
        save_log_pseudolikelihood=save_log_pseudolikelihood,
        model=model,
        random_seed=random_seed,
        chain=chain,
    )
    stage = 0
    betas = []
    accept_ratios = []
    nsteps = []
    smc.initialize_population()
    smc.setup_kernel()
    smc.initialize_logp()

    while smc.beta < 1:
        smc.update_weights_beta()
        if _log is not None:
            _log.info(f"Stage: {stage:3d} Beta: {smc.beta:.3f}")
        smc.update_proposal()
        smc.resample()
        smc.mutate()
        smc.tune()
        stage += 1
        betas.append(smc.beta)
        accept_ratios.append(smc.acc_rate)
        nsteps.append(smc.n_steps)

    return (
        smc.posterior_to_trace(),
        smc.sim_data,
        smc.log_marginal_likelihood,
        smc.log_pseudolikelihood,
        betas,
        accept_ratios,
        nsteps,
    )
