#   Copyright 2024 - present The PyMC Developers
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
import time

from typing import Any

import numpy as np

from arviz import InferenceData
from rich.theme import Theme

import pymc

from pymc.backends.arviz import dict_to_dataset, to_inference_data
from pymc.backends.base import MultiTrace
from pymc.model import Model, modelcontext
from pymc.progress_bar import SMCProgressBarManager, default_progress_theme
from pymc.sampling.mcmc import setup_cores_blas_cores
from pymc.sampling.parallel import _cpu_count, _initialize_multiprocessing_context
from pymc.smc.kernels import IMH
from pymc.smc.parallel import ParallelSMCSampler
from pymc.stats.convergence import log_warnings, run_convergence_checks
from pymc.util import (
    RandomState,
    _get_seeds_per_chain,
)

logger = logging.getLogger(__name__)


def sample_smc(
    draws=2000,
    kernel=IMH,
    *,
    start=None,
    model=None,
    random_seed: RandomState = None,
    chains=None,
    cores=None,
    blas_cores: int | str | None = None,
    compute_convergence_checks=True,
    return_inferencedata=True,
    idata_kwargs=None,
    progressbar=True,
    progressbar_theme: Theme | None = default_progress_theme,
    compile_kwargs: dict | None = None,
    mp_ctx=None,
    **kernel_kwargs,
) -> InferenceData | MultiTrace:
    r"""
    Sequential Monte Carlo based sampling.

    Parameters
    ----------
    draws : int, default 2000
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 2000.
    kernel : SMC_kernel, optional
        SMC kernel used. Defaults to :class:`pymc.smc.smc.IMH` (Independent Metropolis Hastings)
    start : dict or array of dict, optional
        Starting point in parameter space. It should be a list of dict with length `chains`.
        When None (default) the starting point is sampled from the prior distribution.
    model : Model (optional if in ``with`` context).
    random_seed :  int, array_like of int, RandomState or numpy_Generator, optional
        Random seed(s) used by the sampling steps. If a list, tuple or array of ints
        is passed, each entry will be used to seed each chain. A ValueError will be
        raised if the length does not match the number of chains.
    chains : int, optional
        The number of chains to sample. Running independent chains is important for some
        convergence statistics. If ``None`` (default), then set to either ``cores`` or 2, whichever
        is larger.
    cores : int, default None
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system.
    blas_cores: int or "auto" or None, default = "auto"
        The total number of threads blas and openmp functions should use during sampling.
        Setting it to "auto" will ensure that the total number of active blas threads is the
        same as the `cores` argument. If set to an integer, the sampler will try to use that total
        number of blas threads. If `blas_cores` is not divisible by `cores`, it might get rounded
        down. If set to None, this will keep the default behavior of whatever blas implementation
        is used at runtime.
    compute_convergence_checks : bool, default True
        Whether to compute sampler statistics like ``R hat`` and ``effective_n``.
        Defaults to ``True``.
    return_inferencedata : bool, default True
        Whether to return the trace as an InferenceData (True) object or a MultiTrace (False).
        Defaults to ``True``.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`.
    progressbar : bool, optional, default True
        Whether or not to display a progress bar in the command line.
    progressbar_theme : Theme, optional
        Custom theme for progress bar. Defaults to the standard PyMC progress bar theme.
    compile_kwargs: dict, optional
        Keyword arguments to pass to pytensor.function
    mp_ctx : multiprocessing.context or str, optional
        Multiprocessing context for parallel chains. Can be a context object or a string
        ("fork", "spawn", or "forkserver"). If None, defaults to "fork" on macOS ARM and
        "forkserver" on other macOS systems, and the system default elsewhere.

    **kernel_kwargs : dict, optional
        Keyword arguments passed to the SMC_kernel. The default IMH kernel takes the following keywords:

        threshold : float, default 0.5
          Determines the change of beta from stage to stage, i.e. indirectly the number of stages,
          the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
          It should be between 0 and 1.
        correlation_threshold : float, default 0.01
            The lower the value the higher the number of MCMC steps computed automatically.
            Defaults to 0.01. It should be between 0 and 1.

        Additional keyword arguments for other kernels should be checked in the respective docstrings.

    Notes
    -----
    SMC works by moving through successive stages. At each stage the inverse temperature
    :math:`\beta` is increased a little bit (starting from 0 up to 1). When :math:`\beta` = 0
    we have the prior distribution and when :math:`\beta = 1` we have the posterior distribution.
    So in more general terms, we are always computing samples from a tempered posterior that we can
    write as:

    .. math::

        p(\theta \mid y)_{\beta} = p(y \mid \theta)^{\beta} p(\theta)

    A summary of the algorithm is:

     1. Initialize :math:`\beta` at zero and stage at zero.
     2. Generate N samples :math:`S_{\beta}` from the prior (because when :math `\beta = 0` the
        tempered posterior is the prior).
     3. Increase :math:`\beta` in order to make the effective sample size equal some predefined
        value (we use :math:`Nt`, where :math:`t` is 0.5 by default).
     4. Compute a set of N importance weights W. The weights are computed as the ratio of the
        likelihoods of a sample at stage i+1 and stage i.
     5. Obtain :math:`S_{w}` by re-sampling according to W.
     6. Use W to compute the mean and covariance for the proposal distribution, a MvNormal.
     7. Run N independent MCMC chains, starting each one from a different sample
        in :math:`S_{w}`. For the IMH kernel, the mean of the proposal distribution is the
        mean of the previous posterior stage and not the current point in parameter space.
     8. The N chains are run until the autocorrelation with the samples from the previous stage
        stops decreasing given a certain threshold.
     9. Repeat from step 3 until :math:`\beta \ge 1`.
     10. The final result is a collection of N samples from the posterior.


    References
    ----------
    .. [Minson2013] Minson, S. E., Simons, M., and Beck, J. L. (2013).
        "Bayesian inversion for finite fault earthquake source models I- Theory and algorithm."
        Geophysical Journal International, 2013, 194(3), pp.1701-1726.
        `link <https://gji.oxfordjournals.org/content/194/3/1701.full>`__

    .. [Ching2007] Ching, J., and Chen, Y. (2007).
        "Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class
        Selection, and Model Averaging." J. Eng. Mech., 2007, 133(7), pp. 816-832. doi:10.1061/(ASCE)0733-9399(2007)133:7(816).
        `link <http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399
        %282007%29133:7%28816%29>`__
    """
    if cores is None:
        cores = _cpu_count()

    if chains is None:
        chains = max(2, cores)
    else:
        cores = min(chains, cores)

    if compile_kwargs is None:
        compile_kwargs = {}

    kernel_kwargs["compile_kwargs"] = compile_kwargs

    random_seed = _get_seeds_per_chain(random_state=random_seed, chains=chains)

    model = modelcontext(model)

    logger.info("Initializing SMC sampler...")

    mp_ctx = _initialize_multiprocessing_context(mp_ctx)
    joined_blas_limiter, cores, num_blas_cores_per_worker = setup_cores_blas_cores(
        blas_cores, chains, cores, mp_ctx
    )

    t1 = time.time()

    rngs = [np.random.default_rng(seed) for seed in random_seed]

    with model:
        smc_kernel = kernel(
            draws=draws,
            start=None,
            model=model,
            random_seed=rngs[0].integers(2**30),
            **kernel_kwargs,
        )

    # Prepare start points for each chain
    start_points: list[dict | None]
    if start is None:
        start_points = [None] * chains
    elif isinstance(start, dict):
        start_points = [start] * chains
    else:
        if len(start) != chains:
            raise ValueError(f"Number of start dicts must match number of chains ({chains})")
        start_points = start

    parallel = cores > 1 and chains > 1

    traces = []
    sample_stats = []
    sample_settings = []

    if parallel:
        logger.info(
            f"Sampling {chains} chain{'s' if chains > 1 else ''} "
            f"in {cores} job{'s' if cores > 1 else ''}"
        )

        results = []
        with joined_blas_limiter():
            with ParallelSMCSampler(
                kernel=smc_kernel,
                chains=chains,
                cores=cores,
                rngs=rngs,
                start_points=start_points,
                progressbar=progressbar,
                progressbar_theme=progressbar_theme,
                mp_ctx=mp_ctx,
                blas_cores=num_blas_cores_per_worker,
            ) as sampler:
                for result in sampler:
                    results.append(result)

        chain_results: list[list] = [[] for _ in range(chains)]
        for result in results:
            chain_results[result.chain].append(result)

        for chain_idx, chain_samples in enumerate(chain_results):
            if not chain_samples:
                raise RuntimeError(
                    f"Chain {chain_idx} did not produce any results. "
                    "This indicates a failure in parallel sampling."
                )
            final_result = chain_samples[-1]
            trace = _build_trace_from_kernel_state(
                final_result.tempered_posterior,
                final_result.var_info,
                final_result.variables,
                chain_idx,
                model,
            )
            traces.append(trace)

            sample_stats.append(final_result.sample_stats)
            sample_settings.append(final_result.sample_settings)

    else:
        logger.info(
            f"Sampling {chains} chain{'s' if chains > 1 else ''}{' sequentially' if chains > 1 else ''}"
        )
        with joined_blas_limiter():
            _sample_smc_sequentially(
                kernel=smc_kernel,
                chains=chains,
                rngs=rngs,
                start_points=start_points,
                model=model,
                progressbar=progressbar,
                progressbar_theme=progressbar_theme,
                traces=traces,
                sample_stats=sample_stats,
                sample_settings=sample_settings,
            )

    trace = MultiTrace(traces)

    _t_sampling = time.time() - t1
    _, idata = _save_sample_stats(
        sample_settings,
        sample_stats,
        chains,
        trace,
        return_inferencedata,
        _t_sampling,
        idata_kwargs,
        model,
    )

    if compute_convergence_checks:
        if idata is None:
            idata = to_inference_data(trace, log_likelihood=False)
        warns = run_convergence_checks(idata, model)
        trace.report._add_warnings(warns)
        log_warnings(warns)

    if return_inferencedata:
        assert idata is not None
        return idata
    return trace


def _save_sample_stats(
    sample_settings,
    sample_stats,
    chains,
    trace: MultiTrace,
    return_inferencedata: bool,
    _t_sampling,
    idata_kwargs,
    model: Model,
) -> tuple[Any | None, InferenceData | None]:
    sample_settings_dict = sample_settings[0]
    sample_settings_dict["_t_sampling"] = _t_sampling
    sample_stats_dict = sample_stats[0]

    if chains > 1:
        # Collect the stat values from each chain in a single list
        for stat in sample_stats[0].keys():
            value_list = []
            for chain_sample_stats in sample_stats:
                value_list.append(chain_sample_stats[stat])
            sample_stats_dict[stat] = value_list

    idata: InferenceData | None = None
    if not return_inferencedata:
        for stat, value in sample_stats_dict.items():
            setattr(trace.report, stat, value)
        for stat, value in sample_settings_dict.items():
            setattr(trace.report, stat, value)
    else:
        for stat, value in sample_stats_dict.items():
            if chains > 1:
                # Different chains might have more iteration steps, leading to a
                # non-square `sample_stats` dataset, we cast as `object` to avoid
                # numpy ragged array deprecation warning
                sample_stats_dict[stat] = np.array(value, dtype=object)
            else:
                sample_stats_dict[stat] = np.array(value)

        sample_stats = dict_to_dataset(
            sample_stats_dict,
            attrs=sample_settings_dict,
            library=pymc,
        )

        ikwargs: dict[str, Any] = {"model": model}
        if idata_kwargs is not None:
            ikwargs.update(idata_kwargs)
        idata = to_inference_data(trace, **ikwargs)
        idata = InferenceData(**idata, sample_stats=sample_stats)  # type: ignore[arg-type]

    return sample_stats, idata


def _build_trace_from_kernel_state(
    tempered_posterior: np.ndarray,
    var_info: dict,
    variables: list,
    chain: int,
    model: Model,
):
    """Build a trace from kernel state.

    This allows trace building to happen in the main process rather than workers.

    Parameters
    ----------
    tempered_posterior : ndarray
        The final particle positions
    var_info : dict
        Dictionary of variable info {var.name: (shape, size)}
    variables : list
        List of model variables
    chain : int
        Chain index
    model : Model
        PyMC model for trace setup

    Returns
    -------
    NDArray trace backend
    """
    from pymc.backends.ndarray import NDArray
    from pymc.vartypes import discrete_types

    length_pos = len(tempered_posterior)
    varnames = [v.name for v in variables]

    strace = NDArray(name=model.name, model=model)
    strace.setup(length_pos, chain)

    for i in range(length_pos):
        value = []
        size = 0
        for var in variables:
            shape, new_size = var_info[var.name]
            var_samples = tempered_posterior[i][size : size + new_size]
            # Round discrete variable samples
            if var.dtype in discrete_types:
                var_samples = np.round(var_samples).astype(var.dtype)
            value.append(var_samples.reshape(shape))
            size += new_size
        strace.record(point=dict(zip(varnames, value)))

    return strace


def _sample_smc_sequentially(
    *,
    kernel,
    chains: int,
    rngs: list[np.random.Generator],
    start_points: list[dict | None],
    model: Model,
    progressbar: bool,
    progressbar_theme: Theme | None,
    traces: list,
    sample_stats: list,
    sample_settings: list,
):
    """Sample all SMC chains sequentially.

    Parameters
    ----------
    kernel: SMC_KERNEL instance
        An initialized SMC kernel (with compiled functions)
    chains: int
        Total number of chains to sample
    rngs: list of random Generators
        A list of random number generators, one for each chain
    start_points: list
        Starting points for each chain
    model: Model
        The PyMC model
    progressbar: bool
        Whether to show progress bar
    progressbar_theme: Theme
        Progress bar theme
    traces: list
        List to append trace results to
    sample_stats: list
        List to append sample_stats to
    sample_settings: list
        List to append sample_settings to
    """
    with SMCProgressBarManager(
        kernel=kernel,
        chains=chains,
        progressbar=progressbar,
        progressbar_theme=progressbar_theme,
    ) as progress_manager:
        for i in range(chains):
            kernel.initialize(start_points[i], rngs[i])

            stage = 0
            chain_sample_stats: dict[str, list] = {stat: [] for stat in kernel.stats_dtypes_shapes}

            while kernel.beta < 1:
                old_beta = kernel.beta
                kernel.update_beta_and_weights()

                progress_manager.update(
                    chain_idx=i, stage=stage, beta=kernel.beta, old_beta=old_beta, is_last=False
                )

                for stat, value in kernel.step().items():
                    chain_sample_stats[stat].append(value)

                stage += 1

            progress_manager.update(chain_idx=i, stage=stage, beta=kernel.beta, is_last=True)

            trace = _build_trace_from_kernel_state(
                tempered_posterior=kernel.tempered_posterior,
                var_info=kernel.var_info,
                variables=kernel.variables,
                chain=i,
                model=model,
            )

            traces.append(trace)
            sample_stats.append(chain_sample_stats)
            sample_settings.append(kernel.sample_settings())
