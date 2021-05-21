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
    init_draws=500,
    resampling_draws=500,
    init_method='prior',
    init_samples=None,
    start=None,
    init_EL2O='adam',
    mean_field_EL2O=False,
    use_hess_EL2O=False,
    absEL2O=1e-10,
    fracEL2O=1e-2,
    EL2O_draws=100,
    maxiter_EL2O=500,
    EL2O_optim_method='L-BFGS-B',
    scipy_map_method='L-BFGS-B',
    adam_lr=1e-3,
    adam_b1=0.9,
    adam_b2=0.999,
    adam_eps=1.0e-8,
    adam_steps=1000,
    simulator=None,
    model_data=None,
    sim_data_cov=None,
    sim_size=None,
    sim_params=None,
    sim_start=None,
    sim_optim_method='lbfgs',
    sim_tol=0.01,
    local_thresh=3,
    local_step_size=0.1,
    local_grad=True,
    init_local=True,
    full_local=False,
    nf_local_iter=3,
    max_line_search=100,
    k_trunc=0.25,
    norm_tol=0.01,
    ess_tol=0.5,
    optim_iter=1000,
    ftol=2.220446049250313e-9,
    gtol=1.0e-5,
    nf_iter=3,
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
	redraw=True,
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
        resampling_draws,
        init_method,
        init_samples,
        start,
        init_EL2O,
        mean_field_EL2O,
        use_hess_EL2O,
        absEL2O,
        fracEL2O,
        EL2O_draws,
        maxiter_EL2O,
	EL2O_optim_method,
        scipy_map_method,
        adam_lr,
        adam_b1,
        adam_b2,
        adam_eps,
        adam_steps,
        simulator,
        model_data,
        sim_data_cov,
        sim_size,
        sim_params,
	sim_start,
        sim_optim_method,
        sim_tol,
        local_thresh,
        local_step_size,
        local_grad,
        init_local,
        full_local,
        nf_local_iter,
        max_line_search,
        k_trunc,
        norm_tol,
        ess_tol,
        optim_iter,
        ftol,
        gtol,
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
        results.append(sample_nfmc_int(*params, random_seed[i], i, _log))
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
    trace.report._t_sampling = time.time() - t1

    return trace


def sample_nfmc_int(
    draws,
    init_draws,
    resampling_draws,
    init_method,
    init_samples,
    start,
    init_EL2O,
    mean_field_EL2O,
    use_hess_EL2O,
    absEL2O,
    fracEL2O,
    EL2O_draws,
    maxiter_EL2O,
    EL2O_optim_method,
    scipy_map_method,
    adam_lr,
    adam_b1,
    adam_b2,
    adam_eps,
    adam_steps,
    simulator,
    model_data,
    sim_data_cov,
    sim_size,
    sim_params,
    sim_start,
    sim_optim_method,
    sim_tol,
    local_thresh,
    local_step_size,
    local_grad,
    init_local,
    full_local,
    nf_local_iter,
    max_line_search,
    k_trunc,
    norm_tol,
    ess_tol,
    optim_iter,
    ftol,
    gtol,
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
    chain,
    _log,
):
    """Run one NS_NFMC instance."""
    nfmc = NFMC(
        draws=draws,
        init_draws=init_draws,
        resampling_draws=resampling_draws,
        model=model,
        init_method=init_method,
        init_samples=init_samples,
        start=start,
        init_EL2O=init_EL2O,
        mean_field_EL2O=mean_field_EL2O,
        use_hess_EL2O=use_hess_EL2O,
        absEL2O=absEL2O,
        fracEL2O=fracEL2O,
        EL2O_draws=EL2O_draws,
        maxiter_EL2O=maxiter_EL2O,
	EL2O_optim_method=EL2O_optim_method,
        scipy_map_method=scipy_map_method,
        adam_lr=adam_lr,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        adam_steps=adam_steps,
        simulator=simulator,
        model_data=model_data,
        sim_data_cov=sim_data_cov,
        sim_size=sim_size,
        sim_params=sim_params,
	sim_start=sim_start,
        sim_optim_method=sim_optim_method,
        sim_tol=sim_tol,
        local_thresh=local_thresh,
        local_step_size=local_step_size,
        local_grad=local_grad,
        init_local=init_local,
	nf_local_iter=nf_local_iter,
        max_line_search=max_line_search,
        k_trunc=k_trunc,
        random_seed=random_seed,
        chain=chain,
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

    nfmc.initialize_var_info()
    nfmc.setup_logp()
    if init_method == 'prior':
        nfmc.initialize_population()
    elif init_method == 'EL2O_exact':
        print(f'Initializing with Gaussian EL2O approx family, using exact update steps.')
        print(f'Mean field EL2O: {mean_field_EL2O}')
        nfmc.get_map_laplace()
        nfmc.run_el2o()
    elif init_method == 'EL2O_optim':
        print(f'Initializing with Gaussian EL2O approx family, optimization updates.')
        print(f'Mean field EL2O: {mean_field_EL2O}')
        nfmc.get_map_laplace()
        nfmc.run_el2o_optim()
    elif init_method == 'lbfgs' or init_method == 'map+laplace':
        print(f'Using {init_method} to initialize.')
        nfmc.initialize_map_hess()
    elif init_method == 'adam':
        print(f'Using ADAM optimization to intialize (Jax implementation).')
        nfmc.adam_map_hess()
    elif init_method == 'simulation':
        print('Using simulation based initialization.')
        nfmc.simulation_init()
    else:
        raise ValueError('init_method must be one of: prior, EL2O_exact, EL2O_optim, lbfgs, adam, map+laplace or simulation.')

    nfmc.nf_samples_to_trace()
    iter_sample_dict['q_init0'] = nfmc.nf_trace
    iter_weight_dict['q_init0'] = nfmc.weights
    iter_logZ_dict['q_init0'] = nfmc.log_evidence
    iter_q_ess_dict['q_init0'] = nfmc.q_ess
    iter_total_ess_dict['q_init0'] = nfmc.total_ess

    iter_log_evidence = 1.0 * nfmc.log_evidence
    iter_ess = 1.0 * nfmc.q_ess

    print(f"Initialization logZ: {nfmc.log_evidence:.3f}, ESS/N: {nfmc.q_ess:.3f}")

    if nf_local_iter > 0:
        print(f'Using local exploration to improve the SINF initialization.')
        for j in range(nf_local_iter):
            nfmc.fit_nf(num_draws=draws)
            nfmc.nf_samples_to_trace()
            print(f"Local exploration iteration: {int(j + 1)}, logZ: {nfmc.log_evidence:.3f}, Train ESS/N: {nfmc.train_ess:.3f}")
            print(f"Local exploration iteration: {int(j + 1)}, q_init{int(j+1)} ESS/N: {nfmc.q_ess:.3f}")
            print(f"Local exploration iteration: {int(j + 1)}, Min variance BW factor: {nfmc.min_var_bw}, Var(IW): {nfmc.min_var_weights}")
            iter_sample_dict[f'q_init{int(j + 1)}'] = nfmc.nf_trace
            iter_weight_dict[f'q_init{int(j + 1)}'] = nfmc.weights
            iter_logp_dict[f'q_init{int(j + 1)}'] = nfmc.posterior_logp
            iter_logq_dict[f'q_init{int(j + 1)}'] = nfmc.logq
            iter_train_logp_dict[f'q_init{int(j + 1)}'] = nfmc.train_logp
            iter_train_logq_dict[f'q_init{int(j + 1)}'] = nfmc.train_logq
            iter_logZ_dict[f'q_init{int(j + 1)}'] = nfmc.log_evidence
            iter_qmodel_dict[f'q_init{int(j + 1)}'] = nfmc.nf_model
            iter_q_ess_dict[f'q_init{int(j + 1)}'] = nfmc.q_ess
            iter_train_ess_dict[f'q_init{int(j + 1)}'] = nfmc.train_ess
            iter_total_ess_dict[f'q_init{int(j + 1)}'] = nfmc.total_ess
            if (abs(iter_log_evidence - nfmc.log_evidence) <= norm_tol or (nfmc.q_ess / iter_ess) <= ess_tol):
                if abs(iter_log_evidence - nfmc.log_evidence) <= norm_tol:
                    print("Normalizing constant estimate has stabilised during local exploration initialization - ending NF fits with local exploration.")
                elif j == 0 and abs(iter_log_evidence - nfmc.log_evidence) > norm_tol and (nfmc.q_ess / iter_ess) <= ess_tol:
                    print(f"Effective sample size has decreased by more than specified tolerance of {ess_tol}")
                    print("Only using the initialization samples.")
                    nfmc.nf_model = 'init'
                    nfmc.log_evidence = iter_logZ_dict[f'q_init{int(j)}']
                    nfmc.weighted_samples = nfmc.weighted_samples[:-len(nfmc.weights), :]
                    nfmc.importance_weights = nfmc.importance_weights[:-len(nfmc.weights)]
                    nfmc.sinf_logw = nfmc.sinf_logw[:-len(nfmc.weights)]
                elif j > 0 and abs(iter_log_evidence - nfmc.log_evidence) > norm_tol and (nfmc.q_ess / iter_ess) <= ess_tol:
                    print(f"Effective sample size has decreased by more than specified tolerance of {ess_tol}")
                    print("Discarding most recent samples.")
                    nfmc.nf_model = iter_qmodel_dict[f'q_init{int(j)}']
                    nfmc.log_evidence = iter_logZ_dict[f'q_init{int(j)}']
                    nfmc.weighted_samples = nfmc.weighted_samples[:-len(nfmc.weights), :]
                    nfmc.importance_weights = nfmc.importance_weights[:-len(nfmc.weights)]
                    nfmc.sinf_logw = nfmc.sinf_logw[:-len(nfmc.weights)]
                break
            iter_log_evidence = 1.0 * nfmc.log_evidence
            iter_ess = 1.0 * nfmc.q_ess
        print('Re-initializing SINF fits using samples from latest iteration after local exploration.')
        nfmc.reinitialize_nf()
        print(f'Re-initialization logZ: {nfmc.log_evidence:.3f}, ESS/N: {nfmc.q_ess:.3f}')
        iter_sample_dict[f'q_reinit'] = nfmc.nf_trace
        iter_weight_dict[f'q_reinit'] = nfmc.weights
        iter_logp_dict[f'q_reinit'] = nfmc.posterior_logp
        iter_logq_dict[f'q_reinit'] = nfmc.logq
        iter_logZ_dict[f'q_reinit'] = nfmc.log_evidence
        iter_q_ess_dict[f'q_reinit'] = nfmc.q_ess
        iter_total_ess_dict[f'q_reinit'] = nfmc.total_ess
        iter_log_evidence = 1.0 * nfmc.log_evidence
        iter_ess = 1.0 * nfmc.q_ess

    if full_local:
        print('Using local exploration at every iteration except the final one (where IW exceed the local threshold).')
        nfmc.nf_local_iter = 1
    elif not full_local:
        print('No longer using local exploration after warm-up iterations.')
        nfmc.nf_local_iter = 0

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
        iter_qmodel_dict[f'q{int(stage)}'] = nfmc.nf_model
        iter_q_ess_dict[f'q{int(stage)}'] = nfmc.q_ess
        iter_train_ess_dict[f'q{int(stage)}'] = nfmc.train_ess
        iter_total_ess_dict[f'q{int(stage)}'] = nfmc.total_ess
        if _log is not None:
            _log.info(f"Stage: {stage:3d}, logZ Estimate: {nfmc.log_evidence:.3f}, Train ESS/N: {nfmc.train_ess:.3f}")
            _log.info(f"Stage: {stage:3d}, q ESS/N: {nfmc.q_ess:.3f}")
            _log.info(f"Stage: {stage:3d}, Min variance BW factor: {nfmc.min_var_bw}, Var(IW): {nfmc.min_var_weights}")
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

    if full_local:
        nfmc.final_nf()
        nfmc.nf_samples_to_trace()
        iter_sample_dict[f'q_final'] = nfmc.nf_trace
        iter_weight_dict[f'q_final'] = nfmc.weights
        iter_logp_dict[f'q_final'] = nfmc.posterior_logp
        iter_logq_dict[f'q_final'] = nfmc.logq
        iter_train_logp_dict[f'q_final'] = nfmc.train_logp
        iter_train_logq_dict[f'q_final'] = nfmc.train_logq
        iter_logZ_dict[f'q_final'] = nfmc.log_evidence
        iter_qmodel_dict[f'q_final'] = nfmc.nf_model
        iter_q_ess_dict[f'q_final'] = nfmc.q_ess
        iter_train_ess_dict[f'q_final'] = nfmc.train_ess
        iter_total_ess_dict[f'q_final'] = nfmc.total_ess
    elif not full_local:
        nfmc.resample()

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
    )
