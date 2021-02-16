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
from scipy.special import logsumexp

from pymc3.backends.base import MultiTrace
from pymc3.model import modelcontext
from pymc3.parallel_sampling import _cpu_count
from pymc3.ns_nfmc.ns_nfmc import NS_NFMC


def sample_ns_nfmc(
    draws=2000,
    start=None,
    rho=0.01,
    epsilon=0.01,
    model=None,
    frac_validate=0.8,
    alpha=(0,0),
    random_seed=-1,
    parallel=False,
    chains=None,
    cores=None,
):
    r"""
    Normalizing flow based nested sampling.

    Parameters
    ----------
    draws: int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 2000.
    start: dict, or array of dict
        Starting point in parameter space. It should be a list of dict with length `chains`.
        When None (default) the starting point is sampled from the prior distribution.
    rho: float
        Sets fraction of points we want to be above the likelihood threshold at each iteration.
        Used to adaptively set the likelihood threshold during sampling.
    epsilon: float
        Stopping factor for the algorithm. At each iteration we compare the ratio of the evidences
        from the current and previous iterations. If it is less than 1-epsilon we stop.
    model: Model (optional if in ``with`` context)).
    frac_validate: float
        Fraction of the live points at each NS iteration that we use for validation of the NF fit.
    alpha: tuple of floats
        Regularization parameters used for the NF fit. 
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

    """
    _log = logging.getLogger("pymc3")
    _log.info("Initializing normalizing flow based nested sampling...")

    model = modelcontext(model)
    if model.name:
        raise NotImplementedError(
            "The NS_NFMC implementation currently does not support named models. "
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

    params = (
        draws,
        start,
        rho,
        epsilon,
        model,
        frac_validate,
        alpha,
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
            results.append(sample_ns_nfmc_int(*params, random_seed[i], i, _log))

    (
        traces,
        log_evidence,
    ) = zip(*results)
    trace = MultiTrace(traces)
    trace.report._n_draws = draws
    trace.report.log_evidence = np.array(log_evidence)
    trace.report._t_sampling = time.time() - t1

    return trace


def sample_ns_nfmc_int(
    draws,
    start,
    rho,
    epsilon,
    model,
    frac_validate,
    alpha,
    random_seed,
    chain,
    _log,
):
    """Run one NS_NFMC instance."""
    ns_nfmc = NS_NFMC(
        draws=draws,
        model=model,
        random_seed=random_seed,
        chain=chain,
        frac_validate=frac_validate,
        alpha=alpha,
        rho=rho,
    )
    stage = 0
    evidence_ratio = 1
    ns_nfmc.initialize_population()
    ns_nfmc.setup_logp()
    ns_nfmc.get_prior_logp()
    ns_nfmc.get_likelihood_logp()
    
    while evidence_ratio > 1 - epsilon:
        ns_nfmc.update_likelihood_thresh()
        ns_nfmc.update_weights()
        if _log is not None:
            _log.info(f"Stage: {stage:3d} Likelihood logp threshold: {ns_nfmc.likelihood_logp_thresh[-1:]:.3f}")
        ns_nfmc.fit_nf()
        stage += 1
        evidence_ratio = ns_nfmc.cumul_evidences[-1:] / ns_nfmc.cumul_evidences[-2:]
    ns_nfmc.resample()
    log_evidence = logsumexp(ns_nfmc.log_evidences)

    return (
        ns_nf_mc.posterior_to_trace(),
        log_evidence,
        ns_nfmc.log_evidences,
        ns_nfmc.likelihood_logp_thresh,
    )
