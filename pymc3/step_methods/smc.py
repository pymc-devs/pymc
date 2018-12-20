"""
Sequential Monte Carlo sampler
"""
import numpy as np
import theano
import pymc3 as pm
from tqdm import tqdm

from .arraystep import metrop_select
from .metropolis import MultivariateNormalProposal
from ..theanof import floatX, make_shared_replacements, join_nonshared_inputs, inputvars
from ..model import modelcontext
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace


__all__ = ["SMC", "sample_smc"]


class SMC:
    """
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
    model : :class:`pymc3.Model`
        Optional model for sampling step. Defaults to None (taken from context).

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
    ):

        self.n_steps = n_steps
        self.max_steps = n_steps
        self.scaling = scaling
        self.p_acc_rate = 1 - p_acc_rate
        self.tune_scaling = tune_scaling
        self.tune_steps = tune_steps
        self.threshold = threshold


def sample_smc(draws=5000, step=None, progressbar=False, model=None, random_seed=-1):
    """
    Sequential Monte Carlo sampling

    Parameters
    ----------
    draws : int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent Markov Chains. Defaults to 5000.
    step : :class:`SMC`
        SMC initialization object
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

    beta = 0.
    stage = 0
    acc_rate = 1.
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
            if step.tune_scaling:
                step.scaling = _tune(acc_rate)
            if step.tune_steps:
                acc_rate = max(1. / proposed, acc_rate)
                step.n_steps = min(
                    step.max_steps, 1 + int(np.log(step.p_acc_rate) / np.log(1 - acc_rate))
                )

        pm._log.info(
            "Stage: {:d} Beta: {:f} Steps: {:d}".format(stage, beta, step.n_steps, acc_rate)
        )
        # Apply Metropolis kernel (mutation)
        proposed = draws * step.n_steps
        accepted = 0.
        priors = np.array([prior_logp(sample) for sample in posterior]).squeeze()
        tempered_post = priors + likelihoods * beta
        for draw in tqdm(range(draws), disable=not progressbar):
            old_tempered_post = tempered_post[draw]
            q_old = posterior[draw]
            deltas = np.squeeze(proposal(step.n_steps) * step.scaling)
            for n_step in range(step.n_steps):
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

                new_tempered_post = prior_logp(q_new) + likelihood_logp(q_new)[0] * beta

                q_old, accept = metrop_select(new_tempered_post - old_tempered_post, q_new, q_old)
                if accept:
                    accepted += 1
                    posterior[draw] = q_old
                    old_tempered_post = new_tempered_post

        acc_rate = accepted / proposed
        stage += 1

    trace = _posterior_to_trace(posterior, variables, model, var_info)

    return trace


def _initial_population(draws, model, variables):
    """
    Create an initial population from the prior
    """

    population = []
    var_info = {}
    start = model.test_point
    init_rnd = pm.sample_prior_predictive(draws, model=model)
    for v in variables:
        var_info[v.name] = (start[v.name].shape, start[v.name].size)

    for i in range(draws):
        point = pm.Point({v.name: init_rnd[v.name][i] for v in variables}, model=model)
        population.append(model.dict_to_array(point))

    return np.array(floatX(population)), var_info


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


def _calc_covariance(posterior, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(posterior, aweights=weights.ravel(), bias=False, rowvar=0)
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
    return np.atleast_2d(cov)


def _tune(acc_rate):
    """
    Tune adaptively based on the acceptance rate.

    Parameters
    ----------
    acc_rate: float
        Acceptance rate of the Metropolis sampling

    Returns
    -------
    scaling: float
    """
    # a and b after Muto & Beck 2008.
    a = 1.0 / 9
    b = 8.0 / 9
    return (a + b * acc_rate) ** 2


def _posterior_to_trace(posterior, variables, model, var_info):
    """
    Save results into a PyMC3 trace
    """
    lenght_pos = len(posterior)
    varnames = [v.name for v in variables]

    with model:
        strace = NDArray(model)
        strace.setup(lenght_pos, 0)
    for i in range(lenght_pos):
        value = []
        size = 0
        for var in varnames:
            shape, new_size = var_info[var]
            value.append(posterior[i][size : size + new_size].reshape(shape))
            size += new_size
        strace.record({k: v for k, v in zip(varnames, value)})
    return MultiTrace([strace])


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
