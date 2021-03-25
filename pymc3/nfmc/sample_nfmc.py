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
    init_samples=None,
    init_el2o=None,
    absEL2O=1e-10,
    fracEL2O=1e-2,
    pareto=False,
    k_trunc=0.25,
    norm_tol=0.01,
    optim_iter=1000,
    ftol=2.220446049250313e-9,
    gtol=1.0e-5,
    nf_iter=3,
    model=None,
    frac_validate=0.1,
    alpha=(0,0),
    verbose=False,
    n_component=None,
    interp_nbin=None,
    KDE=True,
    bw_factor=0.5,
    edge_bins=None,
    ndata_wT=None,
    MSWD_max_iter=None,
    NBfirstlayer=True,
    logit=False,
    Whiten=False,
    batchsize=None,
    nocuda=False,
    patch=False,
    shape=[28,28,1],
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
    init_el2o: str
        If specified, tells us to initialize with EL2O algorithm. Currently must be set to
        either None or 'full_rank'. Will add option for 'gauss_mix' later.
    norm_tol: float
        Fractional difference in the evidence estimate between two steps. If it falls below this we
        stop iterating over the NF fits.
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

    assert init_el2o is None or init_el2o == 'full_rank'
    
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
        f"Cores available for optimization: {cores}"
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
        init_samples,
        init_el2o,
        absEL2O,
        fracEL2O,
        pareto,
        k_trunc,
        norm_tol,
        optim_iter,
        ftol,
        gtol,
        nf_iter,
        model,
        frac_validate,
        alpha,
        cores,
        verbose,
        n_component,
        interp_nbin,
        KDE,
        bw_factor,
        edge_bins,
        ndata_wT,
        MSWD_max_iter,
        NBfirstlayer,
        logit,
        Whiten,
        batchsize,
        nocuda,
        patch,
        shape,
        parallel,
    )

    t1 = time.time()

    results = []
    for i in range(chains):
        results.append(sample_nfmc_int(*params, random_seed[i], i, _log))
    (
        traces,
        evidence,
        weighted_samples,
        importance_weights,
    ) = zip(*results)
    trace = MultiTrace(traces)
    trace.report.evidence = evidence
    trace.report.weighted_samples = weighted_samples
    trace.report.importance_weights = importance_weights
    trace.report._n_draws = draws
    trace.report._t_sampling = time.time() - t1
    
    return trace


def sample_nfmc_int(
    draws,
    init_samples,
    init_el2o,
    absEL2O,
    fracEL2O,
    pareto,
    k_trunc,
    norm_tol,
    optim_iter,
    ftol,
    gtol,
    nf_iter,
    model,
    frac_validate,
    alpha,
    cores,
    verbose,
    n_component,
    interp_nbin,
    KDE,
    bw_factor,
    edge_bins,
    ndata_wT,
    MSWD_max_iter,
    NBfirstlayer,
    logit,
    Whiten,
    batchsize,
    nocuda,
    patch,
    shape,
    parallel,
    random_seed,
    chain,
    _log,
):
    """Run one NS_NFMC instance."""
    nfmc = NFMC(
        draws=draws,
        model=model,
        init_samples=init_samples,
        init_el2o=init_el2o,
        absEL2O=absEL2O,
        fracEL2O=fracEL2O,
        pareto=pareto,
        k_trunc=k_trunc,
        random_seed=random_seed,
        chain=chain,
        frac_validate=frac_validate,
        alpha=alpha,
        verbose=verbose,
        optim_iter=optim_iter,
        ftol=ftol,
        gtol=gtol,
        n_component=n_component,
        interp_nbin=interp_nbin,
        KDE=KDE,
        bw_factor=bw_factor,
        edge_bins=edge_bins,
        ndata_wT=ndata_wT,
        MSWD_max_iter=MSWD_max_iter,
        NBfirstlayer=NBfirstlayer,
        logit=logit,
        Whiten=Whiten,
        batchsize=batchsize,
        nocuda=nocuda,
        patch=patch,
        shape=shape,
    )
    stage = 1
    nfmc.initialize_var_info()
    nfmc.setup_logp()
    if init_el2o is not None:
        print(f'Initializing with EL2O approximation family: {init_el2o}')
        nfmc.get_map_laplace()
        nfmc.run_el2o()
    else:
        nfmc.initialize_population()

    '''
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
    '''

    print('Fitting the first NF approx to the initial samples ...')
    nfmc.initialize_nf()
    iter_evidence = 1.0 * nfmc.evidence
    
    for i in range(nf_iter):

        if _log is not None:
            _log.info(f"Stage: {stage:3d}, Normalizing Constant Estimate: {nfmc.evidence}")
        nfmc.fit_nf()
        stage += 1
        if np.abs((iter_evidence - nfmc.evidence) / nfmc.evidence) <= norm_tol:
            print('Normalizing constant estimate has stabilised - ending NF fits.')
            break
        iter_evidence = 1.0 * nfmc.evidence
        
    nfmc.resample()

    return (
        nfmc.posterior_to_trace(),
        nfmc.evidence,
        nfmc.weighted_samples,
        nfmc.importance_weights,
    )
