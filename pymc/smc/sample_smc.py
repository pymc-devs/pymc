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

from collections import defaultdict
from collections.abc import Iterable
from itertools import repeat

import cloudpickle
import numpy as np

from arviz import InferenceData
from fastprogress.fastprogress import progress_bar

import pymc

from pymc.backends.arviz import dict_to_dataset, to_inference_data
from pymc.backends.base import MultiTrace
from pymc.model import modelcontext
from pymc.parallel_sampling import _cpu_count
from pymc.smc.smc import IMH


def sample_smc(
    draws=2000,
    kernel=IMH,
    *,
    start=None,
    model=None,
    random_seed=None,
    chains=None,
    cores=None,
    compute_convergence_checks=True,
    return_inferencedata=True,
    idata_kwargs=None,
    progressbar=True,
    **kernel_kwargs,
):
    r"""
    Sequential Monte Carlo based sampling.

    Parameters
    ----------
    draws: int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 2000.
    kernel: SMC Kernel used. Defaults to pm.smc.IMH (Independent Metropolis Hastings)
    start: dict, or array of dict
        Starting point in parameter space. It should be a list of dict with length `chains`.
        When None (default) the starting point is sampled from the prior distribution.
    model: Model (optional if in ``with`` context)).
    random_seed: int
        random seed
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics. If ``None`` (default), then set to either ``cores`` or 2, whichever
        is larger.
    cores : int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system.
    compute_convergence_checks : bool
        Whether to compute sampler statistics like Gelman-Rubin and ``effective_n``.
        Defaults to ``True``.
    return_inferencedata : bool, default=True
        Whether to return the trace as an :class:`arviz:arviz.InferenceData` (True) object or a `MultiTrace` (False)
        Defaults to ``True``.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    progressbar : bool, optional default=True
        Whether or not to display a progress bar in the command line.
    **kernel_kwargs: keyword arguments passed to the SMC kernel.
        The default IMH kernel takes the following keywords:
            threshold: float
                Determines the change of beta from stage to stage, i.e. indirectly the number of stages,
                the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
                It should be between 0 and 1.
            n_steps: int
                The number of steps of each Markov Chain. If ``tune_steps == True`` ``n_steps`` will be used
                for the first stage and for the others it will be determined automatically based on the
                acceptance rate and `p_acc_rate`, the max number of steps is ``n_steps``.
            tune_steps: bool
                Whether to compute the number of steps automatically or not. Defaults to True
            p_acc_rate: float
                Used to compute ``n_steps`` when ``tune_steps == True``. The higher the value of
                ``p_acc_rate`` the higher the number of steps computed automatically. Defaults to 0.85.
                It should be between 0 and 1.
        Keyword arguments for other kernels should be checked in the respective docstrings

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

    if isinstance(kernel, str) and kernel.lower() in ("abc", "metropolis"):
        warnings.warn(
            f'The kernel string argument "{kernel}" in sample_smc has been deprecated. '
            f"It is no longer needed to distinguish between `abc` and `metropolis`",
            FutureWarning,
            stacklevel=2,
        )
        kernel = IMH

    if kernel_kwargs.pop("save_sim_data", None) is not None:
        warnings.warn(
            "save_sim_data has been deprecated. Use pm.sample_posterior_predictive "
            "to obtain the same type of samples.",
            FutureWarning,
            stacklevel=2,
        )

    if kernel_kwargs.pop("save_log_pseudolikelihood", None) is not None:
        warnings.warn(
            "save_log_pseudolikelihood has been deprecated. This information is "
            "now saved as log_likelihood in models with Simulator distributions.",
            FutureWarning,
            stacklevel=2,
        )

    parallel = kernel_kwargs.pop("parallel", None)
    if parallel is not None:
        warnings.warn(
            "The argument parallel is deprecated, use the argument cores instead.",
            FutureWarning,
            stacklevel=2,
        )
        if parallel is False:
            cores = 1

    if cores is None:
        cores = _cpu_count()

    if chains is None:
        chains = max(2, cores)
    else:
        cores = min(chains, cores)

    if random_seed == -1:
        raise FutureWarning(
            f"random_seed should be a non-negative integer or None, got: {random_seed}"
            "This will raise a ValueError in the Future"
        )
        random_seed = None
    if isinstance(random_seed, int) or random_seed is None:
        rng = np.random.default_rng(seed=random_seed)
        random_seed = list(rng.integers(2**30, size=chains))
    elif isinstance(random_seed, Iterable):
        if len(random_seed) != chains:
            raise ValueError(f"Length of seeds ({len(seeds)}) must match number of chains {chains}")
    else:
        raise TypeError("Invalid value for `random_seed`. Must be tuple, list, int or None")

    model = modelcontext(model)

    _log = logging.getLogger("pymc")
    _log.info("Initializing SMC sampler...")
    _log.info(
        f"Sampling {chains} chain{'s' if chains > 1 else ''} "
        f"in {cores} job{'s' if cores > 1 else ''}"
    )

    params = (
        draws,
        kernel,
        start,
        model,
    )

    t1 = time.time()

    if cores > 1:
        results = run_chains_parallel(
            chains, progressbar, _sample_smc_int, params, random_seed, kernel_kwargs, cores
        )
    else:
        results = run_chains_sequential(
            chains, progressbar, _sample_smc_int, params, random_seed, kernel_kwargs
        )
    (
        traces,
        sample_stats,
        sample_settings,
    ) = zip(*results)

    trace = MultiTrace(traces)

    _t_sampling = time.time() - t1
    sample_stats, idata = _save_sample_stats(
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
        _compute_convergence_checks(idata, draws, model, trace)
    return idata if return_inferencedata else trace


def _save_sample_stats(
    sample_settings,
    sample_stats,
    chains,
    trace,
    return_inferencedata,
    _t_sampling,
    idata_kwargs,
    model,
):
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

    if not return_inferencedata:
        for stat, value in sample_stats_dict.items():
            setattr(trace.report, stat, value)
        for stat, value in sample_settings_dict.items():
            setattr(trace.report, stat, value)
        idata = None
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

        ikwargs = dict(model=model)
        if idata_kwargs is not None:
            ikwargs.update(idata_kwargs)
        idata = to_inference_data(trace, **ikwargs)
        idata = InferenceData(**idata, sample_stats=sample_stats)

    return sample_stats, idata


def _compute_convergence_checks(idata, draws, model, trace):
    if draws < 100:
        warnings.warn(
            "The number of samples is too small to check convergence reliably.",
            stacklevel=2,
        )
    else:
        if idata is None:
            idata = to_inference_data(trace, log_likelihood=False)
        trace.report._run_convergence_checks(idata, model)
    trace.report._log_summary()


def _sample_smc_int(
    draws,
    kernel,
    start,
    model,
    random_seed,
    chain,
    progressbar=None,
    **kernel_kwargs,
):
    """Run one SMC instance."""
    in_out_pickled = type(model) == bytes
    if in_out_pickled:
        # function was called in multiprocessing context, deserialize first
        (draws, kernel, start, model) = map(
            cloudpickle.loads,
            (
                draws,
                kernel,
                start,
                model,
            ),
        )

        kernel_kwargs = {key: cloudpickle.loads(value) for key, value in kernel_kwargs.items()}

    smc = kernel(
        draws=draws,
        start=start,
        model=model,
        random_seed=random_seed,
        **kernel_kwargs,
    )

    if progressbar:
        progressbar.comment = f"{getattr(progressbar, 'base_comment', '')} Stage: 0 Beta: 0"
        progressbar.update_bar(getattr(progressbar, "offset", 0) + 0)

    smc._initialize_kernel()
    smc.setup_kernel()

    stage = 0
    sample_stats = defaultdict(list)
    while smc.beta < 1:

        smc.update_beta_and_weights()

        if progressbar:
            progressbar.comment = (
                f"{getattr(progressbar, 'base_comment', '')} Stage: {stage} Beta: {smc.beta:.3f}"
            )
            progressbar.update_bar(getattr(progressbar, "offset", 0) + int(smc.beta * 100))

        smc.resample()
        smc.tune()
        smc.mutate()
        for stat, value in smc.sample_stats().items():
            sample_stats[stat].append(value)

        stage += 1

    results = (
        smc._posterior_to_trace(chain),
        sample_stats,
        smc.sample_settings(),
    )

    if in_out_pickled:
        results = cloudpickle.dumps(results)

    return results


def run_chains_parallel(chains, progressbar, to_run, params, random_seed, kernel_kwargs, cores):
    pbar = progress_bar((), total=100, display=progressbar)
    pbar.update(0)
    pbars = [pbar] + [None] * (chains - 1)

    pool = mp.Pool(cores)

    # "manually" (de)serialize params before/after multiprocessing
    params = tuple(cloudpickle.dumps(p) for p in params)
    kernel_kwargs = {key: cloudpickle.dumps(value) for key, value in kernel_kwargs.items()}
    results = _starmap_with_kwargs(
        pool,
        to_run,
        [(*params, random_seed[chain], chain, pbars[chain]) for chain in range(chains)],
        repeat(kernel_kwargs),
    )
    results = tuple(cloudpickle.loads(r) for r in results)
    pool.close()
    pool.join()
    return results


def run_chains_sequential(chains, progressbar, to_run, params, random_seed, kernel_kwargs):
    results = []
    pbar = progress_bar((), total=100 * chains, display=progressbar)
    pbar.update(0)
    for chain in range(chains):
        pbar.offset = 100 * chain
        pbar.base_comment = f"Chain: {chain + 1}/{chains}"
        results.append(to_run(*params, random_seed[chain], chain, pbar, **kernel_kwargs))
    return results


def _starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    # Helper function to allow kwargs with Pool.starmap
    # Copied from https://stackoverflow.com/a/53173433/13311693
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
