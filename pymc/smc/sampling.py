#   Copyright 2024 The PyMC Developers
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
import multiprocessing
import time
import warnings

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any

import cloudpickle
import numpy as np

from arviz import InferenceData
from rich.progress import (
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import pymc

from pymc.backends.arviz import dict_to_dataset, to_inference_data
from pymc.backends.base import MultiTrace
from pymc.model import Model, modelcontext
from pymc.sampling.parallel import _cpu_count
from pymc.smc.kernels import IMH
from pymc.stats.convergence import log_warnings, run_convergence_checks
from pymc.util import CustomProgress, RandomState, _get_seeds_per_chain


def sample_smc(
    draws=2000,
    kernel=IMH,
    *,
    start=None,
    model=None,
    random_seed: RandomState = None,
    chains=None,
    cores=None,
    compute_convergence_checks=True,
    return_inferencedata=True,
    idata_kwargs=None,
    progressbar=True,
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
    **kernel_kwargs : dict, optional
        Keyword arguments passed to the SMC_kernel. The default IMH kernel takes the following keywords:

        threshold : float, default 0.5
          Determines the change of beta from stage to stage, i.e. indirectly the number of stages,
          the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
          It should be between 0 and 1.
            correlation_threshold : float, default 0.01
                The lower the value the higher the number of MCMC steps computed automatically.
                Defaults to 0.01. It should be between 0 and 1.
        Keyword arguments for other kernels should be checked in the respective docstrings.

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

    random_seed = _get_seeds_per_chain(random_state=random_seed, chains=chains)

    model = modelcontext(model)

    _log = logging.getLogger(__name__)
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

    results = run_chains(chains, progressbar, params, random_seed, kernel_kwargs, cores)

    (
        traces,
        sample_stats,
        sample_settings,
    ) = zip(*results)

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

        ikwargs: dict[str, Any] = dict(model=model)
        if idata_kwargs is not None:
            ikwargs.update(idata_kwargs)
        idata = to_inference_data(trace, **ikwargs)
        idata = InferenceData(**idata, sample_stats=sample_stats)

    return sample_stats, idata


def _sample_smc_int(
    draws,
    kernel,
    start,
    model,
    random_seed,
    chain,
    progress_dict,
    task_id,
    **kernel_kwargs,
):
    """Run one SMC instance."""
    in_out_pickled = isinstance(model, bytes)
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

    smc._initialize_kernel()
    smc.setup_kernel()

    stage = 0
    sample_stats = defaultdict(list)
    while smc.beta < 1:
        smc.update_beta_and_weights()

        progress_dict[task_id] = {"stage": stage, "beta": smc.beta}

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


def run_chains(chains, progressbar, params, random_seed, kernel_kwargs, cores):
    with CustomProgress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        TimeRemainingColumn(),
        TextColumn("/"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
        disable=not progressbar,
    ) as progress:
        futures = []  # keep track of the jobs
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our
            # main process and our worker functions
            _progress = manager.dict()

            # "manually" (de)serialize params before/after multiprocessing
            params = tuple(cloudpickle.dumps(p) for p in params)
            kernel_kwargs = {key: cloudpickle.dumps(value) for key, value in kernel_kwargs.items()}

            with ProcessPoolExecutor(max_workers=cores) as executor:
                for c in range(chains):  # iterate over the jobs we need to run
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(f"Chain {c}", status="Stage: 0 Beta: 0")
                    futures.append(
                        executor.submit(
                            _sample_smc_int,
                            *params,
                            random_seed[c],
                            c,
                            _progress,
                            task_id,
                            **kernel_kwargs,
                        )
                    )

                # monitor the progress:
                done = []
                remaining = futures
                while len(remaining) > 0:
                    finished, remaining = wait(remaining, timeout=0.1)
                    done.extend(finished)
                    for task_id, update_data in _progress.items():
                        stage = update_data["stage"]
                        beta = update_data["beta"]
                        # update the progress bar for this task:
                        progress.update(
                            status=f"Stage: {stage} Beta: {beta:.3f}",
                            task_id=task_id,
                            refresh=True,
                        )

        return tuple(cloudpickle.loads(r.result()) for r in done)
