"""
Sequential Monte Carlo-ABC sampler
"""
import numpy as np
import theano
import pymc3 as pm
from scipy.stats.mstats import mquantiles
from tqdm import tqdm

from .arraystep import metrop_select
from .metropolis import MultivariateNormalProposal
from ..theanof import floatX
from ..model import modelcontext, treelist, FreeRV
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace


__all__ = ['SMC-ABC', 'sample_smc_abc']

proposal_dists = {'MultivariateNormal': MultivariateNormalProposal}


class SMC_ABC():
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
    sum_stat: list
        List of summary statistics to compute.
    min_epsilon : float
        Minimum epsilons threshold that the sampler can reach.
    iqr_scale : int
        Scale factor for the inter-quantile range used in epsilon computation.
    distance_metric : string
        Distance metric to be computed between summary statistics.
    epsilons : list
        Pre-specified list of epsilon thresholds.
    proposal_name :
        Type of proposal distribution. Currently the only valid option is `MultivariateNormal`.

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
    def __init__(self, n_steps=5, scaling=1., p_acc_rate=0.01, tune=True, sum_stat=['mean'],
        min_epsilon=0.5, iqr_scale=1, distance_metric='absolute difference', epsilons=None, 
        proposal_name='MultivariateNormal', routine='iqr'):

        self.n_steps = n_steps
        self.scaling = scaling
        self.p_acc_rate = p_acc_rate
        self.tune = tune
        self.proposal = proposal_dists[proposal_name]
        self.sum_stat = sum_stat
        self.min_epsilon = min_epsilon
        self.iqr_scale = iqr_scale
        self.distance_metric = distance_metric
        self.epsilons = epsilons
        self.routine = routine

def sample_smc_abc(draws=5000, step=None, progressbar=False, model=None, random_seed=-1):
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

    stage = 0
    #variables = pm.model.treelist([])
    #for k, v in model.named_vars.items():
    #    if not isinstance(v, pm.model.ObservedRV):
    #        if not pm.util.is_transformed_name(k):
    #            print(v)
    #            variables.append(v)
    variables = model.vars
    discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in variables])
    any_discrete = discrete.any()
    all_discrete = discrete.all()
    prior_logp = theano.function(variables, model.varlogpt)
    simulator = model.observed_RVs[0]
    function = simulator.distribution.function
    observed_sum_stat = get_sum_stats(simulator.observations, sum_stat=step.sum_stat)
    epsilon = np.inf
    distance_list = []
    pm._log.info('Using {} as distance metric'.format(step.distance_metric))
    pm._log.info('Using {} as summary statistic'.format(step.sum_stat))

    if step.epsilons is None:
        epsilon_list = []

    else:
        step.epsilons.append(step.min_epsilon)
        epsilon_list = step.epsilons

    while epsilon > step.min_epsilon:
        if stage == 0:
            pm._log.info('Sample initial stage: ...')
            posterior = _initial_population(draws, model, variables)
            weights = np.ones(draws) / draws
        else:
            weights = calc_weights(un_weights)
            
        # resample based on plausibility weights (selection)
        resampling_indexes = np.random.choice(np.arange(draws), size=draws, p=weights)
        posterior = posterior[resampling_indexes]
        # compute proposal distribution based on weights
        try:
            covariance = _calc_covariance(posterior, weights)
        except ValueError:
            pm._log.info('Could not compute covariance matrix')
            break
        proposal = step.proposal(covariance)
        # get distance function
        distance_function = get_distance(step.distance_metric)
        
        # compute scaling and number of Markov chains steps (optional), based on previous
        # acceptance rate
        #if step.tune and stage > 0:
        #    if acc_rate == 0:
        #        acc_rate = 1. / step.n_steps
        #    step.scaling = _tune(acc_rate)
        #    step.n_steps = 1 + (np.ceil(np.log(step.p_acc_rate) / np.log(1 - acc_rate)).astype(int))

        # Apply Rejection kernel (mutation)
        proposed = 0.
        accepted = 0.
        new_posterior_list = []
        proposal_list = []

        #if step.epsilons is None:
        if stage == 0:
            simulated_sample = [function(*sample) for sample in posterior][::10]
            epsilon_list.append(calc_epsilon(simulated_sample[0], step.iqr_scale, step, step.epsilons, stage, step.routine))
        else:
            epsilon_list.append(calc_epsilon(distance_list, step.iqr_scale, step, step.epsilons, stage, step.routine))
            distance_list = []
        
        epsilon = epsilon_list[stage]
        pm._log.info('Sampling stage {} with Epsilon {:f}'.format(stage, epsilon))

        for draw in tqdm(range(draws), disable=not progressbar):
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
                    q_new = q_old + delta
                
                if np.isfinite(prior_logp(*q_new)):

                    simulated_data = function(*q_new)
                    
                    simulated_stat = get_sum_stats(simulated_data, sum_stat=step.sum_stat)
                    distance = distance_function(simulated_stat, observed_sum_stat)

                    if distance < epsilon:
                        accepted += 1.
                        new_posterior_list.append(q_new)
                        proposal_list.append(proposal.logp(covariance * step.scaling, q_new))
                        distance_list.append(distance)
                        break
                    
                proposed += 1.

        if new_posterior_list:
            new_posterior = np.array(new_posterior_list)
            resampling_indexes = np.random.choice(np.arange(len(new_posterior)), size=draws)

            posterior = new_posterior[resampling_indexes]
            proposal_array = np.array(proposal_list)
            proposal_array = proposal_array[resampling_indexes]
            priors = np.array([prior_logp(*sample) for sample in posterior])
            un_weights = priors - proposal_array
            acc_rate = accepted / proposed
            stage += 1
        else:
            pm._log.info('Acceptance rate is 0')
            break

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
            population[:,idx] = v.distribution.dist.random(size=samples, point=start)
        else:
            population[:,idx] = v.random(size=samples, point=start)

    return population


#def _initial_population(draws, model, variables):
#    """
#    Create an initial population from the prior
#    """
#    population = []
#    var_info = {}
#    start = model.test_point
#    init_rnd = pm.sample_prior_predictive(draws, model=model)
#    for v in variables:
#        var_info[v.name] = (start[v.name].shape, start[v.name].size)
#
#    for i in range(draws):
#        point = pm.Point({v.name: init_rnd[v.name][i] for v in variables}, model=model)
#        population.append(model.dict_to_array(point))
#
#    return np.array(population), var_info



def calc_epsilon(population, iqr_scale, step, epsilons, stage, routine):
    """Calculate next epsilon threshold based on the current population.

    Returns
    -------
    epsilon : float
    """
    if step.epsilons is None:
        if step.routine == 'iqr':
            range_iq = mquantiles(population, prob=[0.25, 0.75])
            epsilon = np.abs(range_iq[1] - range_iq[0]) * iqr_scale
        if step.routine == 'median':
            epsilon = np.abs(np.median(population)) * iqr_scale
    else:
        epsilon = step.epsilons[stage]

    return epsilon 

def calc_weights(un_weights):
    """Compute importance weights"""
    weights_un = np.exp(un_weights - un_weights.max())
    weights = weights_un / np.sum(weights_un)
    return weights

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

def get_sum_stats(data, sum_stat=SMC_ABC().sum_stat):
    """
    Parameters:
    -----------
    data : array
        Observed or simulated data
    sum_stat : list
        List of summary statistics to be computed. Accepted strings are mean, std, var. 
        Python functions can be passed in this argument.

    Returns:
    --------
    sum_stat_vector : array
        Array contaning the summary statistics.
    """
    if data.ndim == 1:
        data = data[:,np.newaxis]
    sum_stat_vector = np.zeros((len(sum_stat), data.shape[1]))

    for i, stat in enumerate(sum_stat):
        for j in range(sum_stat_vector.shape[1]):
            if stat == 'mean':
                sum_stat_vector[i, j] =  data[:,j].mean()
            elif stat == 'std':
                sum_stat_vector[i, j] =  data[:,j].std()
            elif stat == 'var':
                sum_stat_vector[i, j] =  data[:,j].var()
            else:
                sum_stat_vector[i, j] =  stat(data[:,j])

    return np.atleast_1d(np.squeeze(sum_stat_vector))

def absolute_difference(a, b):
    return np.sum(np.abs(a - b))

def sum_of_squared_distance(a, b):
    return np.sum((a - b)**2)

def mean_absolute_error(a, b):
    return np.sum(np.abs(a - b))/len(a)

def mean_squared_error(a, b):
    return np.sum((a - b)**2)/len(a)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def get_distance(func_name):
    d = {'absolute difference': absolute_difference,
         'sum of squared_distance' : sum_of_squared_distance,
         'mean absolute_error' : mean_absolute_error,
         'mean squared_error' : mean_squared_error,
         'euclidean' : euclidean_distance}
    for key, value in d.items():
        if func_name == key:
            return value