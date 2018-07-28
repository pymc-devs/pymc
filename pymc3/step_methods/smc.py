"""
Sequential Monte Carlo sampler
"""
import numpy as np
import theano
import pymc3 as pm
from tqdm import tqdm

from .arraystep import metrop_select
from .metropolis import MultivariateNormalProposal
from ..theanof import floatX
from ..model import modelcontext
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace


__all__ = ['SMC', 'sample_smc']

proposal_dists = {'MultivariateNormal': MultivariateNormalProposal}


class SMC():
    """
    Sequential Monte Carlo step

    Parameters
    ----------
    n_steps : int
        The number of steps of a Markov Chain. If `tune == True` `n_steps` will be used for
        the first stage, and the number of steps of the other states will be determined
        automatically based on the acceptance rate and `p_acc_rate`.
    scaling : float
        Factor applied to the proposal distribution i.e. the step size of the Markov Chain. Only
        works if `tune == False` otherwise is determined automatically
    p_acc_rate : float
        Probability of not accepting a Markov Chain proposal. Used to compute `n_steps` when
        `tune == True`. It should be between 0 and 1.
    proposal_name :
        Type of proposal distribution. Currently the only valid option is `MultivariateNormal`.
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of threshold the higher the number of stages. Defaults to 0.5.
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
    default_blocked = True

    def __init__(self, n_steps=5, scaling=1., p_acc_rate=0.01, tune=True,
                 proposal_name='MultivariateNormal', threshold=0.5):

        self.n_steps = n_steps
        self.scaling = scaling
        self.p_acc_rate = p_acc_rate
        self.tune = tune
        self.proposal = proposal_dists[proposal_name]
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

    beta = 0
    stage = 0
    model.marginal_likelihood = 1
    variables = model.vars
    discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in variables])
    any_discrete = discrete.any()
    all_discrete = discrete.all()
    prior_logp = theano.function(model.vars, model.varlogpt)
    likelihood_logp = theano.function(model.vars, model.datalogpt)
    pm._log.info('Sample initial stage: ...')
    posterior = _initial_population(draws, model, variables)

    while beta < 1:
        # compute plausibility weights (measure fitness)
        likelihoods = np.array([likelihood_logp(*sample) for sample in posterior])
        beta, old_beta, weights, sj = _calc_beta(beta, likelihoods, step.threshold)
        model.marginal_likelihood *= sj
        pm._log.info('Beta: {:f} Stage: {:d}'.format(beta, stage))

        # resample based on plausibility weights (selection)
        resampling_indexes = np.random.choice(np.arange(draws), size=draws, p=weights)
        posterior = posterior[resampling_indexes]
        likelihoods = likelihoods[resampling_indexes]

        # compute proposal distribution based on weights
        covariance = _calc_covariance(posterior, weights)
        proposal = step.proposal(covariance)

        # compute scaling and number of Markov chains steps (optional), based on previous
        # acceptance rate
        if step.tune and stage > 0:
            if acc_rate == 0:
                acc_rate = 1. / step.n_steps
            step.scaling = _tune(acc_rate)
            step.n_steps = 1 + (np.ceil(np.log(step.p_acc_rate) / np.log(1 - acc_rate)).astype(int))

        # Apply Metropolis kernel (mutation)
        proposed = 0.
        accepted = 0.
        priors = np.array([prior_logp(*sample) for sample in posterior])
        tempered_post = priors + likelihoods * beta
        for draw in tqdm(range(draws), disable=not progressbar):
            old_tempered_post = tempered_post[draw]
            q_old = posterior[draw]
            deltas = np.squeeze(proposal(step.n_steps) * step.scaling)
            for n_step in range(0, step.n_steps):
                delta = deltas[n_step]

                if any_discrete:
                    if all_discrete:
                        delta = np.round(delta, 0).astype('int64')
                        q_old = q_old.astype('int64')
                        q_new = (q_old + delta).astype('int64')
                    else:
                        delta[discrete] = np.round(delta[discrete], 0)
                        q_new = (q_old + delta)
                else:
                    q_new = floatX(q_old + delta)

                new_tempered_post = prior_logp(*q_new) + likelihood_logp(*q_new) * beta

                q_old, accept = metrop_select(new_tempered_post - old_tempered_post, q_new, q_old)
                if accept:
                    accepted += accept
                    posterior[draw] = q_old
                    old_tempered_post = new_tempered_post
                proposed += 1.

        acc_rate = accepted / proposed
        stage += 1

    trace = _posterior_to_trace(posterior, model)

    return trace

# FIXME!!!!
def _initial_population(samples, model, variables):
    """
    Create an initial population from the prior
    """
    population = np.zeros((samples, len(variables)))
    init_rnd = {}
    start = model.test_point
    for idx, v in enumerate(variables):
        if pm.util.is_transformed_name(v.name):
            trans = v.distribution.transform_used.forward_val
            population[:,idx] = trans(v.distribution.dist.random(size=samples, point=start))
        else:
            population[:,idx] = v.random(size=samples, point=start)

    return population


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
    beta : float
        tempering parameter of the next stage
    beta : float
        tempering parameter of the current stage
    weights : numpy array
        Importance weights (floats)
    """
    low_beta = old_beta = beta
    up_beta = 2.
    rN = int(len(likelihoods) * threshold)

    while up_beta - low_beta > 1e-6:
        new_beta = (low_beta + up_beta) / 2.
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
    lala = np.exp((new_beta - old_beta) * likelihoods)
    weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
    weights = weights_un / np.sum(weights_un)
    return new_beta, old_beta, weights, np.mean(lala)


def _calc_covariance(posterior_array, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(np.squeeze(posterior_array), aweights=weights.ravel(), bias=False, rowvar=0)
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "chains" is too small!')
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
    # a and b after Muto & Beck 2008 .
    a = 1. / 9
    b = 8. / 9
    return (a + b * acc_rate) ** 2

def _posterior_to_trace(posterior, model):
    """
    Save results into a PyMC3 trace
    """
    length_pos = len(posterior)
    varnames = [v.name for v in model.vars]
    with model:
        strace = NDArray(model)
        strace.setup(length_pos, 0)
    for i in range(length_pos):
        strace.record({k:v for k, v in zip(varnames, posterior[i])})
    return MultiTrace([strace])
