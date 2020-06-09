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

import time
import logging
from .smc import SMC


def sample_smc(
    draws=1000,
    kernel="metropolis",
    n_steps=25,
    parallel=False,
    start=None,
    cores=None,
    tune_steps=True,
    p_acc_rate=0.99,
    threshold=0.5,
    epsilon=1.0,
    dist_func="gaussian_kernel",
    sum_stat="identity",
    progressbar=False,
    model=None,
    random_seed=-1,
):
    r"""
    Sequential Monte Carlo based sampling

    Parameters
    ----------
    draws: int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 1000.
    kernel: str
        Kernel method for the SMC sampler. Available option are ``metropolis`` (default) and `ABC`.
        Use `ABC` for likelihood free inference togheter with a ``pm.Simulator``.
    n_steps: int
        The number of steps of each Markov Chain. If ``tune_steps == True`` ``n_steps`` will be used
        for the first stage and for the others it will be determined automatically based on the
        acceptance rate and `p_acc_rate`, the max number of steps is ``n_steps``.
    parallel: bool
        Distribute computations across cores if the number of cores is larger than 1.
        Defaults to False.
    start: dict, or array of dict
        Starting point in parameter space. It should be a list of dict with length `chains`.
        When None (default) the starting point is sampled from the prior distribution. 
    cores: int
        The number of chains to run in parallel. If ``None`` (default), it will be automatically
        set to the number of CPUs in the system.
    tune_steps: bool
        Whether to compute the number of steps automatically or not. Defaults to True
    p_acc_rate: float
        Used to compute ``n_steps`` when ``tune_steps == True``. The higher the value of
        ``p_acc_rate`` the higher the number of steps computed automatically. Defaults to 0.99.
        It should be between 0 and 1.
    threshold: float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
        It should be between 0 and 1.
    epsilon: float
        Standard deviation of the gaussian pseudo likelihood. Only works with `kernel = ABC`
    dist_func: str
        Distance function. The only available option is ``gaussian_kernel``
    sum_stat: str or callable
        Summary statistics. Available options are ``indentity``, ``sorted``, ``mean``, ``median``.
        If a callable is based it should return a number or a 1d numpy array.
    progressbar: bool
        Flag for displaying a progress bar. Defaults to False.
    model: Model (optional if in ``with`` context)).
    random_seed: int
        random seed

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
     6. Use W to compute the covariance for the proposal distribution.
     7. For stages other than 0 use the acceptance rate from the previous stage to estimate the
        scaling of the proposal distribution and `n_steps`.
     8. Run N Metropolis chains (each one of length `n_steps`), starting each one from a different
        sample in :math:`S_{w}`.
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

    smc = SMC(
        draws=draws,
        kernel=kernel,
        n_steps=n_steps,
        parallel=parallel,
        start=start,
        cores=cores,
        tune_steps=tune_steps,
        p_acc_rate=p_acc_rate,
        threshold=threshold,
        epsilon=epsilon,
        dist_func=dist_func,
        sum_stat=sum_stat,
        progressbar=progressbar,
        model=model,
        random_seed=random_seed,
    )

    t1 = time.time()
    _log = logging.getLogger("pymc3")
    _log.info("Sample initial stage: ...")
    stage = 0
    smc.initialize_population()
    smc.setup_kernel()
    smc.initialize_logp()

    while smc.beta < 1:
        smc.update_weights_beta()
        _log.info(
            "Stage: {:3d} Beta: {:.3f} Steps: {:3d} Acce: {:.3f}".format(
                stage, smc.beta, smc.n_steps, smc.acc_rate
            )
        )
        smc.resample()
        smc.update_proposal()
        if stage > 0:
            smc.tune()
        smc.mutate()
        stage += 1

    if smc.parallel and smc.cores > 1:
        smc.pool.close()
        smc.pool.join()

    trace = smc.posterior_to_trace()
    trace.report._n_draws = smc.draws
    trace.report._n_tune = 0
    trace.report._t_sampling = time.time() - t1
    return trace
