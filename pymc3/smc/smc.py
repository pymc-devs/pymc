"""
Sequential Monte Carlo sampler
"""
import numpy as np
import pymc3 as pm
from tqdm import tqdm
import multiprocessing as mp
import warnings

from ..step_methods.metropolis import MultivariateNormalProposal
from .smc_utils import (
    _initial_population,
    _calc_covariance,
    _tune,
    _posterior_to_trace,
    logp_forw,
    calc_beta,
    metrop_kernel,
    PseudoLikelihood,
)
from ..theanof import inputvars, make_shared_replacements
from ..model import modelcontext
from ..parallel_sampling import _cpu_count


EXPERIMENTAL_WARNING = (
    "Warning: SMC-ABC methods are experimental step methods and not yet"
    " recommended for use in PyMC3!"
)

__all__ = ["sample_smc"]


def sample_smc(
    draws=1000,
    start=None,
    cores=None,
    kernel="metropolis",
    n_steps=25,
    scaling=1.0,
    p_acc_rate=0.99,
    tune_scaling=True,
    tune_steps=True,
    threshold=0.5,
    parallel=False,
    epsilon=1.0,
    dist_func="absolute_error",
    sum_stat=False,
    progressbar=False,
    model=None,
    random_seed=-1,
):
    """
    Sequential Monte Carlo based sampling

    Parameters
    ----------
    draws : int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent chains. Defaults to 1000.
    start : dict, or array of dict
        Starting point in parameter space. It should be a list of dict with length `chains`.
        If you use this the first stage will not be sampled from the prior disstribution.
    cores : int
        The number of chains to run in parallel. If ``None`` (default), set to the number of CPUs
        in the system.
    kernel : str
        Kernel method of the SMC sampler. Available option are ``metropolis`` (default) and `ABC`.
        Use `ABC` for likelihood free inference togheter with a ``pm.Simulator``
    n_steps : int
        The number of steps of a Markov Chain. If ``tune_steps == True`` ``n_steps`` will be used
        for the first stage and for the other stages it will be determined automatically based on
        the acceptance rate and `p_acc_rate`, the number of steps will never be larger than
        ``n_steps``.
    scaling : float
        Factor applied to the proposal distribution i.e. the step size of the Markov Chain. Only
        works if ``tune_scaling == False`` otherwise is determined automatically.
    p_acc_rate : float
        Used to compute ``n_steps`` when ``tune_steps == True``. The higher the value of
        ``p_acc_rate`` the higher the number of steps computed automatically. Defaults to 0.99.
        It should be between 0 and 1.
    tune_scaling : bool
        Whether to compute the scaling automatically or not. Defaults to True
    tune_steps : bool
        Whether to compute the number of steps automatically or not. Defaults to True
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
        It should be between 0 and 1.
    parallel : bool
        Distribute computations across cores if the number of cores is larger than 1.
        Defaults to False.
    epsilon : float
        Standard deviation of the gaussian pseudo likelihood. Only works with `kernel = ABC`
    dist_func : str
        Distance function. Available options are ``absolute_error`` (default) and
        ``sum_of_squared_distance``. Only works with ``kernel = ABC``
    sum_stat : bool
        Whether to use or not a summary statistics. Defaults to False. Only works with
        ``kernel = ABC``
    progressbar : bool
        Flag for displaying a progress bar
    model : Model (optional if in ``with`` context)).
    random_seed : int
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

    model = modelcontext(model)

    if random_seed != -1:
        np.random.seed(random_seed)

    if cores is None:
        cores = _cpu_count()

    beta = 0
    stage = 0
    accepted = 0
    acc_rate = 1.0
    max_steps = n_steps
    proposed = draws * n_steps
    model.marginal_likelihood = 1
    variables = inputvars(model.vars)
    discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in variables])
    any_discrete = discrete.any()
    all_discrete = discrete.all()
    shared = make_shared_replacements(variables, model)
    prior_logp = logp_forw([model.varlogpt], variables, shared)

    pm._log.info("Sample initial stage: ...")

    posterior, var_info = _initial_population(draws, model, variables, start)

    if kernel.lower() == "abc":
        warnings.warn(EXPERIMENTAL_WARNING)
        simulator = model.observed_RVs[0]
        likelihood_logp = PseudoLikelihood(
            epsilon,
            simulator.observations,
            simulator.distribution.function,
            model,
            var_info,
            dist_func,
            sum_stat,
        )
    elif kernel.lower() == "metropolis":
        likelihood_logp = logp_forw([model.datalogpt], variables, shared)

    while beta < 1:
        if parallel and cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(likelihood_logp, [(sample,) for sample in posterior])
        else:
            results = [likelihood_logp(sample) for sample in posterior]
        likelihoods = np.array(results).squeeze()
        beta, old_beta, weights, sj = calc_beta(beta, likelihoods, threshold)
        model.marginal_likelihood *= sj
        # resample based on plausibility weights (selection)
        resampling_indexes = np.random.choice(np.arange(draws), size=draws, p=weights)
        posterior = posterior[resampling_indexes]
        likelihoods = likelihoods[resampling_indexes]

        # compute proposal distribution based on weights
        covariance = _calc_covariance(posterior, weights)
        proposal = MultivariateNormalProposal(covariance)

        # compute scaling (optional) and number of Markov chains steps (optional), based on the
        # acceptance rate of the previous stage
        if (tune_scaling or tune_steps) and stage > 0:
            scaling, n_steps = _tune(
                acc_rate,
                proposed,
                tune_scaling,
                tune_steps,
                scaling,
                n_steps,
                max_steps,
                p_acc_rate,
            )

        pm._log.info("Stage: {:d} Beta: {:.3f} Steps: {:d}".format(stage, beta, n_steps))
        # Apply Metropolis kernel (mutation)
        proposed = draws * n_steps
        priors = np.array([prior_logp(sample) for sample in posterior]).squeeze()
        tempered_logp = priors + likelihoods * beta
        deltas = np.squeeze(proposal(n_steps) * scaling)

        parameters = (
            proposal,
            scaling,
            accepted,
            any_discrete,
            all_discrete,
            discrete,
            n_steps,
            prior_logp,
            likelihood_logp,
            beta,
        )

        if parallel and cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(
                metrop_kernel,
                [(posterior[draw], tempered_logp[draw], *parameters) for draw in range(draws)],
            )
        else:
            results = [
                metrop_kernel(posterior[draw], tempered_logp[draw], *parameters)
                for draw in tqdm(range(draws), disable=not progressbar)
            ]

        posterior, acc_list = zip(*results)
        posterior = np.array(posterior)
        acc_rate = sum(acc_list) / proposed
        stage += 1

    trace = _posterior_to_trace(posterior, variables, model, var_info)

    return trace
