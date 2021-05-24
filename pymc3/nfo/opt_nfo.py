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

# This file is adapted from sample_nfmc.py

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
from pymc3.nfo.nfo import NFO


def opt_nfo(
	#Optimization parameters
    draws=500,
    init_draws=500,
    init_method='prior',
    init_samples=None,
    start=None,
    k_trunc=0.25,
    norm_tol=0.01,
    ess_tol=0.5,
    nf_iter=3,
	#NF parameters
    model=None,
    frac_validate=0.1,
    iteration=None,
    alpha=(0,0),
    verbose=False,
    n_component=None,
    interp_nbin=None,
    KDE=True,
    bw_factor_min=0.5,
    bw_factor_max=2.5,
    bw_factor_num=11,
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
	#Other
	redraw=True,
    random_seed=-1,
    parallel=False,
    cores=None
):
    r"""
    Normalizing flow-based Bayesian Optimization.

    Parameters
    ----------
    draws: int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 2000.
    start: dict, or array of dict
        Starting point in parameter space. It should be a list of dict with length `chains`.
        When None (default) the starting point is sampled from the prior distribution.
    init_method: str
        Tells us how to initialize the NFMC fits. Default is 'prior'. If this is supplied along with init_samples
        we use those instead. Current options are 'prior', 'full_rank', 'lbfgs'.
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

    """

    _log = logging.getLogger("pymc3")
    _log.info("Initializing normalizing flow-based optimization...")

    model = modelcontext(model)
    if model.name:
        raise NotImplementedError(
            "The NS_NFO implementation currently does not support named models. "
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
        init_draws,
        init_method,
        init_samples,
        start,
        k_trunc,
        norm_tol,
        ess_tol,
        nf_iter,
        model,
        frac_validate,
        iteration,
        alpha,
        cores,
        verbose,
        n_component,
        interp_nbin,
        KDE,
        bw_factor_min,
        bw_factor_max,
        bw_factor_num,
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
		redraw,
        parallel,
    )

    t1 = time.time()

    results = []
    for i in range(chains):
        results.append(opt_nfo_int(*params, random_seed[i], i, _log))
    (
        traces,
        log_evidence,
        q_samples,
        importance_weights,
        logp,
        logq,
        train_logp,
        train_logq,
        logZ,
        q_models,
        q_ess,
        train_ess,
        total_ess,
        min_var_bws,
        min_pq_bws
    ) = zip(*results)
    trace = MultiTrace(traces)
    trace.report.log_evidence = log_evidence
    trace.report.q_samples = q_samples
    trace.report.importance_weights = importance_weights
    trace.report.logp = logp
    trace.report.logq = logq
    trace.report.train_logp = train_logp
    trace.report.train_logq = train_logq
    trace.report.logZ = logZ
    trace.report.q_models = q_models
    trace.report.q_ess = q_ess
    trace.report.train_ess = train_ess
    trace.report.total_ess = total_ess
    trace.report._n_draws = draws
    trace.report.min_var_bws = min_var_bws
    trace.report.min_pq_bws = min_pq_bws
    trace.report._t_sampling = time.time() - t1


    return trace


def opt_nfo_int(
    draws,
    init_draws,
    init_method,
    init_samples,
    start,
    k_trunc,
    norm_tol,
    ess_tol,
    nf_iter,
    model,
    frac_validate,
    iteration,
    alpha,
    cores,
    verbose,
    n_component,
    interp_nbin,
    KDE,
    bw_factor_min,
    bw_factor_max,
    bw_factor_num,
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
	redraw,
    parallel,
    random_seed,
    _log,
):
    """Run one NS_NFO instance."""
    nfmc = NFO(
        draws=draws,
        init_draws=init_draws,
        model=model,
        init_method=init_method,
        init_samples=init_samples,
        start=start,
        k_trunc=k_trunc,
        random_seed=random_seed,
        frac_validate=frac_validate,
        iteration=iteration,
        alpha=alpha,
        verbose=verbose,
        optim_iter=optim_iter,
        ftol=ftol,
        gtol=gtol,
        n_component=n_component,
        interp_nbin=interp_nbin,
        KDE=KDE,
        bw_factor_min=bw_factor_min,
        bw_factor_max=bw_factor_max,
        bw_factor_num=bw_factor_num,
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
		redraw=redraw,
    )

    iter_sample_dict = {}
    iter_weight_dict = {}
    iter_logp_dict = {}
    iter_logq_dict = {}
    iter_train_logp_dict = {}
    iter_train_logq_dict = {}
    iter_logZ_dict = {}
    iter_qmodel_dict = {}
    iter_q_ess_dict = {}
    iter_train_ess_dict = {}
    iter_total_ess_dict = {}
    iter_min_var_bw_dict = {}
    iter_min_pq_bw_dict = {}

    nfo.initialize_var_info()
    nfo.setup_logp()
    if init_method == 'prior':
        nfo.initialize_population()
    nfo.nf_samples_to_trace()
    iter_sample_dict['q_init0'] = nfo.nf_trace
    iter_weight_dict['q_init0'] = nfo.weights
    iter_logZ_dict['q_init0'] = nfo.log_evidence
    iter_q_ess_dict['q_init0'] = nfo.q_ess
    iter_total_ess_dict['q_init0'] = nfo.total_ess

    iter_log_evidence = 1.0 * nfo.log_evidence
    iter_ess = 1.0 * nfo.q_ess

    print(f"Initialization logZ: {nfo.log_evidence:.3f}, ESS/N: {nfo.q_ess:.3f}, logZ_pq: {nfo.log_evidence_pq:.3f}")

    stage = 1

    for i in range(nf_iter):

        nfmc.fit_nf(num_draws=draws)
        nfmc.nf_samples_to_trace()
        iter_sample_dict[f'q{int(stage)}'] = nfmc.nf_trace
        iter_weight_dict[f'q{int(stage)}'] = nfmc.weights
        iter_logp_dict[f'q{int(stage)}'] = nfmc.posterior_logp
        iter_logq_dict[f'q{int(stage)}'] = nfmc.logq
        iter_train_logp_dict[f'q{stage}'] = nfmc.train_logp
        iter_train_logq_dict[f'q{stage}'] = nfmc.train_logq
        iter_logZ_dict[f'q{int(stage)}'] = nfmc.log_evidence
        iter_logZ_dict[f'q{int(stage)}_pq'] = nfmc.log_evidence_pq
        iter_qmodel_dict[f'q{int(stage)}'] = nfmc.nf_model
        iter_q_ess_dict[f'q{int(stage)}'] = nfmc.q_ess
        iter_train_ess_dict[f'q{int(stage)}'] = nfmc.train_ess
        iter_total_ess_dict[f'q{int(stage)}'] = nfmc.total_ess
        iter_min_var_bw_dict[f'q{int(stage)}'] = nfmc.min_var_bw
        iter_min_pq_bw_dict[f'q{int(stage)}'] = nfmc.min_pq_bw
        if _log is not None:
            _log.info(f"Stage: {stage:3d}, logZ Estimate: {nfmc.log_evidence:.3f}, Train ESS/N: {nfmc.train_ess:.3f},logZ_pq Estimate: {nfmc.log_evidence_pq:.3f}")
            _log.info(f"Stage: {stage:3d}, q ESS/N: {nfmc.q_ess:.3f}")
            _log.info(f"Stage: {stage:3d}, Min variance BW factor: {nfmc.min_var_bw}, Var(IW): {nfmc.min_var_weights}, Min Zpq BW factor: {nfmc.min_pq_bw}")
        stage += 1
        if (abs(iter_log_evidence - nfmc.log_evidence) <= norm_tol or
            (nfmc.q_ess / iter_ess) <= ess_tol):
            if abs(iter_log_evidence - nfmc.log_evidence) <= norm_tol:
                print("Normalizing constant estimate has stabilised - ending NF fits.")
            else:
                print(f"Effective sample size has decreased by more than specified tolerance of {ess_tol}")
                nfmc.nf_model = iter_qmodel_dict[f'q{int(stage - 1)}']
                nfmc.log_evidence = iter_logZ_dict[f'q{int(stage - 1)}']
                nfmc.weighted_samples = nfmc.weighted_samples[:-len(nfmc.weights), :]
                nfmc.importance_weights = nfmc.importance_weights[:-len(nfmc.weights)]
            break
        iter_log_evidence = 1.0 * nfmc.log_evidence
        iter_ess = nfmc.q_ess

    return (
        nfmc.posterior_to_trace(),
        nfmc.log_evidence,
        iter_sample_dict,
        iter_weight_dict,
        iter_logp_dict,
        iter_logq_dict,
        iter_train_logp_dict,
        iter_train_logq_dict,
        iter_logZ_dict,
        iter_qmodel_dict,
        iter_q_ess_dict,
        iter_train_ess_dict,
        iter_total_ess_dict,
        iter_min_var_bw_dict,
        iter_min_pq_bw_dict,
    )
