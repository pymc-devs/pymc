<<<<<<< HEAD
"""Sequential Monte Carlo sampler also known as
Adaptive Transitional Markov Chain Monte Carlo sampler.
Runs on any pymc3 model.
Created on March, 2016
Various significant updates July, August 2016
Made pymc3 compatible November 2016
Renamed to SMC and further improvements March 2017
@author: Hannes Vasyura-Bathke
=======
"""
Sequential Monte Carlo sampler
>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
"""
import numpy as np
import theano
import pymc3 as pm
from tqdm import tqdm
import warnings

from .arraystep import metrop_select
from .metropolis import MultivariateNormalProposal
from ..theanof import floatX, make_shared_replacements, join_nonshared_inputs
from ..model import modelcontext
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace


__all__ = ['SMC', 'sample_smc']

proposal_dists = {'MultivariateNormal': MultivariateNormalProposal}


<<<<<<< HEAD
def choose_proposal(proposal_name, scale=1.):
    """Initialize and select proposal distribution.
    Parameters
    ----------
    proposal_name : string
        Name of the proposal distribution to initialize
    scale : float or :class:`numpy.ndarray`
    Returns
    -------
    class:`pymc3.Proposal` Object
    """
    return proposal_dists[proposal_name](scale)


class SMC(atext.ArrayStepSharedLLK):
    """Adaptive Transitional Markov-Chain Monte-Carlo sampler class.
    Creates initial samples and framework around the (C)ATMIP parameters
=======
class SMC():
    """
    Sequential Monte Carlo step

>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
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
<<<<<<< HEAD
    random_seed : int
        Optional to set the random seed.  Necessary for initial population.
=======

>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
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

    def __init__(self, n_steps=5, scaling=1., p_acc_rate=0.01, tune=True,
                 proposal_name='MultivariateNormal', threshold=0.5):

        self.n_steps = n_steps
        self.scaling = scaling
        self.p_acc_rate = p_acc_rate
        self.tune = tune
        self.proposal = proposal_dists[proposal_name]
        self.threshold = threshold


<<<<<<< HEAD
    def astep(self, q0):
        if self.stage == 0:
            l_new = self.logp_forw(q0)
            q_new = q0

        else:
            if not self.steps_until_tune and self.tune_interval:

                # Tune scaling parameter
                acc_rate = self.accepted / float(self.tune_interval)
                self.scaling = tune(acc_rate)
                # compute n_steps
                if self.accepted == 0:
                    acc_rate = 1 / float(self.tune_interval)
                self.n_steps = 1 + (np.ceil(np.log(self.p_acc_rate) /
                                            np.log(1 - acc_rate)).astype(int))
                # Reset counter
                self.steps_until_tune = self.tune_interval
                self.accepted = 0
                self.stage_sample = 0

            if not self.stage_sample:
                self.proposal_samples_array = self.proposal_dist(self.n_steps)

            delta = self.proposal_samples_array[self.stage_sample, :] * self.scaling

            if self.any_discrete:
                if self.all_discrete:
                    delta = np.round(delta, 0)
                    q0 = q0.astype(int)
                    q = (q0 + delta).astype(int)
                else:
                    delta[self.discrete] = np.round(delta[self.discrete], 0).astype(int)
                    q = q0 + delta
                    q = q[self.discrete].astype(int)
            else:
                q = q0 + delta

            l0 = self.chain_previous_lpoint[self.chain_index]

            if self.check_bnd:
                varlogp = self.check_bnd(q)

                if np.isfinite(varlogp):
                    logp = self.logp_forw(q)
                    q_new, accepted = metrop_select(
                        self.beta * (logp[self._llk_index] - l0[self._llk_index]), q, q0)

                    if accepted:
                        self.accepted += 1
                        l_new = logp
                        self.chain_previous_lpoint[self.chain_index] = l_new
                    else:
                        l_new = l0
                else:
                    q_new = q0
                    l_new = l0

            else:
                logp = self.logp_forw(q)
                q_new, accepted = metrop_select(
                    self.beta * (logp[self._llk_index] - l0[self._llk_index]), q, q0)

                if accepted:
                    self.accepted += 1
                    l_new = logp
                    self.chain_previous_lpoint[self.chain_index] = l_new
                else:
                    l_new = l0

            self.steps_until_tune -= 1
            self.stage_sample += 1

            # reset sample counter
            if self.stage_sample == self.n_steps:
                self.stage_sample = 0

        return q_new, l_new

    def calc_beta(self):
        """Calculate next tempering beta and importance weights based on current beta and sample
        likelihoods.
        Returns
        -------
        beta(m+1) : scalar, float
            tempering parameter of the next stage
        beta(m) : scalar, float
            tempering parameter of the current stage
        weights : :class:`numpy.ndarray`
            Importance weights (floats)
        """
        low_beta = old_beta = self.beta
        up_beta = 2.
        rN = int(len(self.likelihoods) * self.threshold)

        while up_beta - low_beta > 1e-6:
            new_beta = (low_beta + up_beta) / 2.
            weights_un = np.exp((new_beta - old_beta) * (self.likelihoods - self.likelihoods.max()))

            weights = weights_un / np.sum(weights_un)
            ESS = int(1 / np.sum(weights ** 2))
            #ESS = int(1 / np.max(weights))
            if ESS == rN:
                break
            elif ESS < rN:
                up_beta = new_beta
            else:
                low_beta = new_beta

        return new_beta, old_beta, weights#, np.mean(weights_un)

    def calc_covariance(self):
        """Calculate trace covariance matrix based on importance weights.
        Returns
        -------
        cov : :class:`numpy.ndarray`
            weighted covariances (NumPy > 1.10. required)
        """
        cov = np.cov(self.array_population, aweights=self.weights.ravel(), bias=False, rowvar=0)
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "chains" is too small!')
        return np.atleast_2d(cov)

    def select_end_points(self, mtrace, chains):
        """Read trace results (variables and model likelihood) and take end points for each chain
        and set as start population for the next stage.
        Parameters
        ----------
        mtrace : :class:`.base.MultiTrace`
        Returns
        -------
        population : list
            of :func:`pymc3.Point` dictionaries
        array_population : :class:`numpy.ndarray`
            Array of trace end-points
        likelihoods : :class:`numpy.ndarray`
            Array of likelihoods of the trace end-points
        """
        array_population = np.zeros((chains, self.ordering.size))
        n_steps = len(mtrace)

        # collect end points of each chain and put into array
        for var, slc, shp, _ in self.ordering.vmap:
            slc_population = mtrace.get_values(varname=var, burn=n_steps - 1, combine=True)
            if len(shp) == 0:
                array_population[:, slc] = np.atleast_2d(slc_population).T
            else:
                array_population[:, slc] = slc_population
        # get likelihoods
        likelihoods = mtrace.get_values(varname=self.likelihood_name,
                                        burn=n_steps - 1, combine=True)

        # map end array_endpoints to dict points
        population = [self.bij.rmap(row) for row in array_population]

        return population, array_population, likelihoods

    def get_chain_previous_lpoint(self, mtrace, chains):
        """Read trace results and take end points for each chain and set as previous chain result
        for comparison of metropolis select.
        Parameters
        ----------
        mtrace : :class:`.base.MultiTrace`
        Returns
        -------
        chain_previous_lpoint : list
            all unobservedRV values, including dataset likelihoods
        """
        array_population = np.zeros((chains, self.lordering.size))
        n_steps = len(mtrace)
        for _, slc, shp, _, var in self.lordering.vmap:
            slc_population = mtrace.get_values(varname=var, burn=n_steps - 1, combine=True)
            if len(shp) == 0:
                array_population[:, slc] = np.atleast_2d(slc_population).T
            else:
                array_population[:, slc] = slc_population

        return [self.lij.rmap(row) for row in array_population[self.resampling_indexes, :]]

    def mean_end_points(self):
        """Calculate mean of the end-points and return point.
        Returns
        -------
        Dictionary of trace variables
        """
        return self.bij.rmap(self.array_population.mean(axis=0))

    def resample(self, chains):
        """Resample pdf based on importance weights. based on Kitagawas deterministic resampling
        algorithm.
        Returns
        -------
        outindex : :class:`numpy.ndarray`
            Array of resampled trace indexes
        """
        parents = np.arange(chains)
        N_childs = np.zeros(chains, dtype=int)

        cum_dist = np.cumsum(self.weights)
        u = (parents + np.random.rand()) / chains
        j = 0
        for i in parents:
            while u[i] > cum_dist[j]:
                j += 1

            N_childs[j] += 1

        indx = 0
        outindx = np.zeros(chains, dtype=int)
        for i in parents:
            if N_childs[i] > 0:
                for j in range(indx, (indx + N_childs[i])):
                    outindx[j] = parents[i]

            indx += N_childs[i]

        return outindx


def sample_smc(samples=1000, chains=100, step=None, start=None, homepath=None, stage=0, cores=1,
               progressbar=False, model=None, random_seed=-1, rm_flag=True, **kwargs):
    """Sequential Monte Carlo sampling
<<<<<<< HEAD
    Samples the solution space with `chains` of Metropolis chains, where each chain has `n_steps`=`samples`/`chains`
    iterations. Once finished, the sampled traces are evaluated:
=======

    Samples the parameter space using a `chains` number of parallel Metropolis chains.
    Once finished, the sampled traces are evaluated:

>>>>>>> d7374f5b0cf130f0b71ec95644d6a2c9d555ad8b
    (1) Based on the likelihoods of the final samples, chains are weighted
    (2) the weighted covariance of the ensemble is calculated and set as new proposal distribution
    (3) the variation in the ensemble is calculated and also the next tempering parameter (`beta`)
    (4) New `chains` Markov chains are seeded on the traces with high weight for a given number of
        iterations, the iterations can be computed automatically.
    (5) Repeat until `beta` > 1.
=======
def sample_smc(draws=5000, step=None, progressbar=False, model=None, random_seed=-1):
    """
    Sequential Monte Carlo sampling

>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
    Parameters
    ----------
    draws : int
        The number of samples to draw from the posterior (i.e. last stage). And also the number of
        independent Markov Chains. Defaults to 5000.
    step : :class:`SMC`
        SMC initialization object
    progressbar : bool
        Flag for displaying a progress bar
<<<<<<< HEAD
    model : :class:`pymc3.Model`
        (optional if in `with` context) has to contain deterministic variable name defined under
        `step.likelihood_name` that contains the model likelihood
    random_seed : int or list of ints
        A list is accepted, more if `cores` is greater than one.
    rm_flag : bool
        If True existing stage result folders are being deleted prior to sampling.
    References
    ----------
    .. [Minson2013] Minson, S. E. and Simons, M. and Beck, J. L., (2013),
        Bayesian inversion for finite fault earthquake source models I- Theory and algorithm.
        Geophysical Journal International, 2013, 194(3), pp.1701-1726,
        `link <https://gji.oxfordjournals.org/content/194/3/1701.full>`__
=======
    model : pymc3 Model
        optional if in `with` context
    random_seed : int
        random seed
>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
    """
    warnings.warn("Warning: SMC is experimental, hopefully it will be ready for PyMC 3.6")
    model = modelcontext(model)

    if random_seed != -1:
        np.random.seed(random_seed)

    beta = 0
    stage = 0
    acc_rate = 1
    model.marginal_likelihood = 1
    variables = model.vars
    discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in variables])
    any_discrete = discrete.any()
    all_discrete = discrete.all()
    shared = make_shared_replacements(variables, model)
    prior_logp = logp_forw([model.varlogpt], variables, shared)
    likelihood_logp = logp_forw([model.datalogpt], variables, shared)

    pm._log.info('Sample initial stage: ...')
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
        proposal = step.proposal(covariance)

        # compute scaling and number of Markov chains steps (optional), based on previous
        # acceptance rate
        if step.tune and stage > 0:
            if acc_rate == 0:
                acc_rate = 1. / step.n_steps
            step.scaling = _tune(acc_rate)
            step.n_steps = 1 + int(np.log(step.p_acc_rate) / np.log(1 - acc_rate))

        pm._log.info('Stage: {:d} Beta: {:f} Steps: {:d} Acc: {:f}'.format(stage, beta,
                                                                           step.n_steps, acc_rate))
        # Apply Metropolis kernel (mutation)
        proposed = 0.
        accepted = 0.
        priors = np.array([prior_logp(sample) for sample in posterior]).squeeze()
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

                new_tempered_post = prior_logp(q_new) + likelihood_logp(q_new)[0] * beta

                q_old, accept = metrop_select(new_tempered_post - old_tempered_post, q_new, q_old)
                if accept:
                    accepted += accept
                    posterior[draw] = q_old
                    old_tempered_post = new_tempered_post
                proposed += 1.

        acc_rate = accepted / proposed
        stage += 1

    trace = _posterior_to_trace(posterior, model, var_info)

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

    return np.array(population), var_info


def _calc_beta(beta, likelihoods, threshold=0.5):
    """
<<<<<<< HEAD
    Modified from :func:`pymc3.sampling._iter_sample` to be more efficient with SMC algorithm.
    """
    model = modelcontext(model)
    draws = int(draws)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    if random_seed != -1:
        nr.seed(random_seed)

    try:
        step = pm.step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = pm.Point(start, model=model)
    step.chain_index = chain
    trace.setup(draws, chain_idx)
    for i in range(draws):
        point, out_list = step.step(point)
        trace.record(out_list)
        yield trace


def _work_chain(work):
    """Wrapper function for parallel execution of _sample i.e. the Markov Chains.
    Parameters
    ----------
    work : List
        Containing all the information that is unique for each Markov Chain
        i.e. [:class:'SMC', chain_number(int), sampling index(int), start_point(dictionary)]
=======
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

>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
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
    sj = np.exp((new_beta - old_beta) * likelihoods)
    weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
    weights = weights_un / np.sum(weights_un)
    return new_beta, old_beta, weights, np.mean(sj)


def _calc_covariance(posterior_array, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(np.squeeze(posterior_array), aweights=weights.ravel(), bias=False, rowvar=0)
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "chains" is too small!')
    return np.atleast_2d(cov)


<<<<<<< HEAD
def tune(acc_rate):
    """Tune adaptively based on the acceptance rate.
=======
def _tune(acc_rate):
    """
    Tune adaptively based on the acceptance rate.

>>>>>>> 28ab6d9f9d83cd6a3f85e523d15ee65c2c233e76
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


def _posterior_to_trace(posterior, model, var_info):
    """
    Save results into a PyMC3 trace
    """
    lenght_pos = len(posterior)
    varnames = [v.name for v in model.vars]

    with model:
        strace = NDArray(model)
        strace.setup(lenght_pos, 0)
    for i in range(lenght_pos):
        value = []
        size = 0
        for var in varnames:
            shape, new_size = var_info[var]
            value.append(posterior[i][size:size+new_size].reshape(shape))
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
