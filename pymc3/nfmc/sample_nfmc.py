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
from pymc3.nfmc.nfmc import NFMC


def sample_nfmc(
    draws=500,
    start=None,
    optim_iter=1000,
    nf_iter=3,
    model=None,
    frac_validate=0.1,
    alpha=(0,0),
    verbose=False,
    random_seed=-1,
    parallel=False,
    chains=None,
    cores=None
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
    optim_iter: int
        Maximum number of optimization steps to run during the initialization.
    nf_iter: int
        Number of NF fit iterations to go through after the optimization step.
    model: Model (optional if in ``with`` context)).
    frac_validate: float
        Fraction of the live points at each NS iteration that we use for validation of the NF fit.
    alpha: tuple of floats
        Regularization parameters used for the NF fit. 
    verbose: boolean
        Whether you want verbose output from the NF fit.
    random_seed: int
        random seed
    parallel: bool
        Distribute computations across cores if the number of cores is larger than 1.
        Defaults to False.
    cores : int
        Number of cores available for the optimization step. Defaults to None, in which case the CPU
        count is used.
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics. Default is 2.

    """
    _log = logging.getLogger("pymc3")
    _log.info("Initializing normalizing flow based sampling...")

    model = modelcontext(model)
    if model.name:
        raise NotImplementedError(
            "The NS_NFMC implementation currently does not support named models. "
            "See https://github.com/pymc-devs/pymc3/pull/4365."
        )
    if cores is None:
        cores = _cpu_count()
        
    _log.info(
        f"Sampling {chains} chain{'s' if chains > 1 else ''} "
        f"Number of cores available for optimization is {cores}"
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
        optim_iter,
        nf_iter,
        model,
        frac_validate,
        alpha,
        cores,
        verbose,
        parallel,
    )

    t1 = time.time()
    '''
    if parallel and chains > 1:
        loggers = [_log] + [None] * (chains - 1)
        pool = mp.Pool(cores)
        results = pool.starmap(
            sample_ns_nfmc_int, [(*params, random_seed[i], i, loggers[i]) for i in range(chains)]
        )

        pool.close()
        pool.join()
    else:
        results = []
        for i in range(chains):
            results.append(sample_ns_nfmc_int(*params, random_seed[i], i, _log))
    '''
    results = []
    for i in range(chains):
        results.append(sample_nfmc_int(*params, random_seed[i], i, _log))
    (
        traces
    ) = zip(*results)
    trace = MultiTrace(traces)
    trace.report._n_draws = draws
    trace.report._t_sampling = time.time() - t1

    return trace


def sample_nfmc_int(
    draws,
    start,
    optim_iter,
    nf_iter,
    model,
    frac_validate,
    alpha,
    cores,
    verbose,
    parallel,
    random_seed,
    chain,
    _log,
):
    """Run one NS_NFMC instance."""
    nfmc = NFMC(
        draws=draws,
        model=model,
        random_seed=random_seed,
        chain=chain,
        frac_validate=frac_validate,
        alpha=alpha,
        verbose=verbose,
        optim_iter=optim_iter,
    )
    stage = 1
    nfmc.initialize_population()
    nfmc.setup_logp()

    # Run the optimization ...
    print('Running initial optimization ...')
    if parallel:
        loggers = [_log] + [None] * (cores - 1)
        pool = mp.Pool(cores)
        optim_results = pool.starmap(
            nfmc.optimize, [sample for sample in nfmc.prior_samples]
        )
        pool.close()
        pool.join()
        np.random.shuffle(np.array(optim_results))
    elif not parallel:
        optim_results = np.empty((0, np.shape(nfmc.prior_samples)[1]))
        for sample in nfmc.prior_samples:
            optim_results = np.append(optim_results, nfmc.optimize(sample), axis=0)
        np.random.shuffle(optim_results)
    nfmc.optim_samples = np.copy(optim_results)
    
    print('Fitting the first NF approx to the prior optimized samples ...')
    nfmc.initialize_nf()
    
    for i in range(nf_iter):

        nfmc.fit_nf()
        if _log is not None:
            _log.info(f"Stage: {stage:3d}; Mean weights: {np.mean(nfmc.importance_weights)}; Std weights: {np.std(nfmc.importance_weights)}")
        stage += 1

    ns_nfmc.resample()

    return (
        ns_nfmc.posterior_to_trace(),
    )
