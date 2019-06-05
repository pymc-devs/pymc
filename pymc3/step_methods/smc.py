"""
Sequential Monte Carlo sampler
"""
import numpy as np
import theano
import pymc3 as pm
from tqdm import tqdm
import multiprocessing as mp

from .arraystep import metrop_select
from .metropolis import MultivariateNormalProposal
from .smc_utils import _initial_population, _calc_covariance, _tune, _posterior_to_trace
from ..theanof import floatX, make_shared_replacements, join_nonshared_inputs, inputvars
from ..model import modelcontext


__all__ = ["SMC", "sample_smc"]


class SMC:
    R"""
    Sequential Monte Carlo step

    Parameters
    ----------
    n_steps : int
        The number of steps of a Markov Chain. If `tune_steps == True` `n_steps` will be used for
        the first stage and the number of steps of the other stages will be determined
        automatically based on the acceptance rate and `p_acc_rate`.
        The number of steps will never be larger than `n_steps`.
    scaling : float
        Factor applied to the proposal distribution i.e. the step size of the Markov Chain. Only
        works if `tune_scaling == False` otherwise is determined automatically.
    p_acc_rate : float
        Used to compute `n_steps` when `tune_steps == True`. The higher the value of `p_acc_rate`
        the higher the number of steps computed automatically. Defaults to 0.99. It should be
        between 0 and 1.
    tune_scaling : bool
        Whether to compute the scaling automatically or not. Defaults to True
    tune_steps : bool
        Whether to compute the number of steps automatically or not. Defaults to True
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
        It should be between 0 and 1.
    parallel : bool
        Distribute computations across cores if the number of cores is larger than 1
        (see pm.sample() for details). Defaults to True.
    model : :class:`pymc3.Model`
        Optional model for sampling step. Defaults to None (taken from context).

    Notes
    -----
    SMC works by moving from successive stages. At each stage the inverse temperature \beta is
    increased a little bit (starting from 0 up to 1). When \beta = 0 we have the prior distribution
    and when \beta =1 we have the posterior distribution. So in more general terms we are always
    computing samples from a tempered posterior that we can write as:

    p(\theta \mid y)_{\beta} = p(y \mid \theta)^{\beta} p(\theta)

    A summary of the algorithm is:

     1. Initialize \beta at zero and stage at zero.
     2. Generate N samples S_{\beta} from the prior (because when \beta = 0 the tempered posterior
        is the prior).
     3. Increase \beta in order to make the effective sample size equals some predefined value
        (we use N*t, where t is 0.5 by default).
     4. Compute a set of N importance weights W. The weights are computed as the ratio of the
        likelihoods of a sample at stage i+1 and stage i.
     5. Obtain S_{w} by re-sampling according to W.
     6. Use W to compute the covariance for the proposal distribution.
     7. For stages other than 0 use the acceptance rate from the previous stage to estimate the
        scaling of the proposal distribution and n_steps.
     8. Run N Metropolis chains (each one of length n_steps), starting each one from a different
        sample in S_{w}.
     9. Repeat from step 3 until \beta \ge 1.
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

    def __init__(
        self,
        n_steps=25,
        scaling=1.0,
        p_acc_rate=0.99,
        tune_scaling=True,
        tune_steps=True,
        threshold=0.5,
        parallel=True,
    ):

        self.n_steps = n_steps
        self.max_steps = n_steps
        self.scaling = scaling
        self.p_acc_rate = 1 - p_acc_rate
        self.tune_scaling = tune_scaling
        self.tune_steps = tune_steps
        self.threshold = threshold
        self.parallel = parallel


def sample_smc(draws=5000, step=None, cores=None, progressbar=False, model=None, random_seed=-1):
    """
    Sequential Monte Carlo sampling

    Parameters
    ----------
    draws : int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent Markov Chains. Defaults to 5000.
    step : :class:`SMC`
        SMC initialization object
    cores : int
        The number of chains to run in parallel.
    progressbar : bool
        Flag for displaying a progress bar
    model : pymc3 Model
        optional if in `with` context
    random_seed : int
        random seed
    """
    model = modelcontext(model)

    if random_seed != -1:
        np.random.seed(random_seed)

    beta = 0.0
    stage = 0
    accepted = 0
    acc_rate = 1.0
    proposed = draws * step.n_steps
    model.marginal_likelihood = 1
    variables = inputvars(model.vars)
    discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in variables])
    any_discrete = discrete.any()
    all_discrete = discrete.all()
    shared = make_shared_replacements(variables, model)
    prior_logp = logp_forw([model.varlogpt], variables, shared)
    likelihood_logp = logp_forw([model.datalogpt], variables, shared)

    pm._log.info("Sample initial stage: ...")
    posterior, var_info = _initial_population(draws, model, variables)

    while beta < 1:
        # compute plausibility weights (measure fitness)
        likelihoods = np.array([likelihood_logp(sample) for sample in posterior]).squeeze()
        beta, old_beta, weights, sj = _calc_beta(beta, likelihoods, step.threshold)
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
        if (step.tune_scaling or step.tune_steps) and stage > 0:
            _tune(acc_rate, proposed, step)

        pm._log.info("Stage: {:d} Beta: {:.3f} Steps: {:d}".format(stage, beta, step.n_steps))
        # Apply Metropolis kernel (mutation)
        proposed = draws * step.n_steps
        priors = np.array([prior_logp(sample) for sample in posterior]).squeeze()
        tempered_logp = priors + likelihoods * beta
        deltas = np.squeeze(proposal(step.n_steps) * step.scaling)

        parameters = (
            proposal,
            step.scaling,
            accepted,
            any_discrete,
            all_discrete,
            discrete,
            step.n_steps,
            prior_logp,
            likelihood_logp,
            beta,
        )

        if step.parallel and cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(
                _metrop_kernel,
                [(posterior[draw], tempered_logp[draw], *parameters) for draw in range(draws)],
            )
        else:
            results = [
                _metrop_kernel(posterior[draw], tempered_logp[draw], *parameters)
                for draw in tqdm(range(draws), disable=not progressbar)
            ]

        posterior, acc_list = zip(*results)
        posterior = np.array(posterior)
        acc_rate = sum(acc_list) / proposed
        stage += 1

    trace = _posterior_to_trace(posterior, variables, model, var_info)

    return trace


def _metrop_kernel(
    q_old,
    old_tempered_logp,
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
):
    """
    Metropolis kernel
    """
    deltas = np.squeeze(proposal(n_steps) * scaling)
    for n_step in range(n_steps):
        delta = deltas[n_step]

        if any_discrete:
            if all_discrete:
                delta = np.round(delta, 0).astype("int64")
                q_old = q_old.astype("int64")
                q_new = (q_old + delta).astype("int64")
            else:
                delta[discrete] = np.round(delta[discrete], 0)
                q_new = floatX(q_old + delta)
        else:
            q_new = floatX(q_old + delta)

        new_tempered_logp = prior_logp(q_new) + likelihood_logp(q_new)[0] * beta

        q_old, accept = metrop_select(new_tempered_logp - old_tempered_logp, q_new, q_old)
        if accept:
            accepted += 1
            old_tempered_logp = new_tempered_logp

    return q_old, accepted


def _calc_beta(beta, likelihoods, threshold=0.5):
    """
    Calculate next inverse temperature (beta) and importance weights based on current beta
    and tempered likelihood.

    Parameters
    ----------
    beta : float
        tempering parameter of current stage
    likelihoods : numpy array
        likelihoods computed from the model
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of threshold the higher the number of stage. Defaults to 0.5.
        It should be between 0 and 1.

    Returns
    -------
    new_beta : float
        tempering parameter of the next stage
    old_beta : float
        tempering parameter of the current stage
    weights : numpy array
        Importance weights (floats)
    sj : float
        Partial marginal likelihood
    """
    low_beta = old_beta = beta
    up_beta = 2.0
    rN = int(len(likelihoods) * threshold)

    while up_beta - low_beta > 1e-6:
        new_beta = (low_beta + up_beta) / 2.0
        weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
        weights = weights_un / np.sum(weights_un)
        ESS = int(1 / np.sum(weights ** 2))
        if ESS == rN:
            break
        elif ESS < rN:
            up_beta = new_beta
        else:
            low_beta = new_beta
    if new_beta >= 1:
        new_beta = 1
    sj = np.exp((new_beta - old_beta) * likelihoods)
    weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
    weights = weights_un / np.sum(weights_un)
    return new_beta, old_beta, weights, np.mean(sj)


def logp_forw(out_vars, vars, shared):
    """Compile Theano function of the model and the input and output variables.

    Parameters
    ----------
    out_vars : List
        containing :class:`pymc3.Distribution` for the output variables
    vars : List
        containing :class:`pymc3.Distribution` for the input variables
    shared : List
        containing :class:`theano.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(out_vars, vars, shared)
    f = theano.function([inarray0], out_list)
    f.trust_input = True
    return f
