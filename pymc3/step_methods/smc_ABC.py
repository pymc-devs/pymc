"""Sequential Monte Carlo sampler also known as
Adaptive Transitional Markov Chain Monte Carlo sampler.

Runs on any pymc3 model.

Created on March, 2016

Various significant updates July, August 2016

Made pymc3 compatible November 2016
Renamed to SMC and further improvements March 2017

@author: Hannes Vasyura-Bathke
"""
import numpy as np
import pymc3 as pm
from tqdm import tqdm

import theano
import copy
import warnings

from scipy.linalg import cholesky
from scipy.stats import multivariate_normal
from scipy.stats.mstats import mquantiles

from ..model import modelcontext
from ..vartypes import discrete_types
from ..theanof import inputvars, make_shared_replacements, join_nonshared_inputs
import numpy.random as nr

#from .metropolis import MultivariateNormalProposal
from .metropolis import Proposal

from .arraystep import metrop_select
from ..backends import smc_text as atext

__all__ = ['SMC', 'sample_smc']

EXPERIMENTAL_WARNING = ("Warning: SMC is an experimental step method, and not yet "
                        "recommended for use in PyMC3!")


class MultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = cholesky(s, lower=True)

    def __call__(self, num_draws=None):
        if num_draws is not None:
            b = np.random.randn(self.n, num_draws)
            return np.dot(self.chol, b).T
        else:
            b = np.random.randn(self.n)
            return np.dot(self.chol, b)

    def logp(self, s, value):
        return multivariate_normal(0, cov=s).logpdf(value)

proposal_dists = {'MultivariateNormal': MultivariateNormalProposal}

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

    Parameters
    ----------
    vars : list
        List of variables for sampler
    out_vars : list
        List of output variables for trace recording. If empty unobserved_RVs are taken.
    samples : int
        The number of samples to draw from the last stage, i.e. the posterior. Defaults to 1000.
        The number of samples should be a multiple of `n_chains`, otherwise the returned number of
        draws will be the lowest closest multiple of `n_chains`.
    n_chains : int
        Number of chains per stage has to be a large number of number of n_jobs (processors to be
        used) on the machine.
    n_steps : int
        The number of steps of a Markov Chain. Only works if `tune_interval=0` otherwise it will be
        determined adaptively.
    scaling : float
        Factor applied to the proposal distribution i.e. the step size of the Markov Chain. Only
        works if `tune_interval=0` otherwise it will be determined adaptively.
    covariance : :class:`numpy.ndarray`
        (n_chains x n_chains)
        Initial Covariance matrix for proposal distribution, if None - identity matrix taken
    likelihood_name : string
        name of the :class:`pymc3.deterministic` variable that contains the model likelihood.
        Defaults to 'l_like__'
    proposal_name :
        Type of proposal distribution, see smc.proposal_dists.keys() for options
    tune_interval : int
        Number of steps to tune for. If tune=0 no tunning will be used. Default 10. SMC tunes two
        related quantities, the scaling of the proposal distribution (i.e. the step size of Markov
        Chain) and the number of steps of a Markov Chain (i.e. `n_steps`).
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of threshold the higher the number of stage. Defaults to 0.5. It should
        be between 0 and 1.
    check_bound : boolean
        Check if current sample lies outside of variable definition speeds up computation as the
        forward model wont be executed. Default: True
    model : :class:`pymc3.Model`
        Optional model for sampling step. Defaults to None (taken from context).
    random_seed : int
        Optional to set the random seed.  Necessary for initial population.

    References
    ----------
    .. [Ching2007] Ching, J. and Chen, Y. (2007).
        Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class
        Selection, and Model Averaging. J. Eng. Mech., 10.1061/(ASCE)0733-9399(2007)133:7(816),
        816-832. `link <http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399
        %282007%29133:7%28816%29>`__
    """
    default_blocked = True

    def __init__(self, vars=None, out_vars=None, samples=1000, n_chains=100, n_steps=25, scaling=1.,
                 covariance=None, likelihood_name='l_like__', proposal_name='MultivariateNormal',
                 tune_interval=10, threshold=0.5, check_bound=True, model=None, random_seed=-1, 
                 observed=None, epsilons=None, ladder=None, minimum_eps=None):

        warnings.warn(EXPERIMENTAL_WARNING)

        if random_seed != -1:
            nr.seed(random_seed)

        model = modelcontext(model)

        if vars is None:
            vars = model.vars

        vars = inputvars(vars)

        if out_vars is None:
            if not any(likelihood_name == RV.name for RV in model.unobserved_RVs):
                pm._log.info('Adding model likelihood to RVs!')
                with model:
                    llk = pm.Deterministic(likelihood_name, model.logpt)
            else:
                pm._log.info('Using present model likelihood!')

            out_vars = model.unobserved_RVs

        out_varnames = [out_var.name for out_var in out_vars]

        if covariance is None and proposal_name == 'MultivariateNormal':
            self.covariance = np.eye(sum(v.dsize for v in vars))
            scale = self.covariance
        elif covariance is None:
            scale = np.ones(sum(v.dsize for v in vars))
        else:
            scale = covariance

        self.proposal_name = proposal_name
        self.proposal_dist = choose_proposal(self.proposal_name, scale=scale)

        self.scaling = np.atleast_1d(scaling)
        self.check_bnd = check_bound
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval

        self.proposal_samples_array = self.proposal_dist(n_chains)

        self.samples = samples
        self.n_steps = n_steps
        self.stage_sample = 0
        self.accepted = 0
        self.observed = observed

        self.ladder = ladder
        self.all_sum_stats = []
        self.sum_stat = 0 
        self.minimum_eps = minimum_eps

        self.beta = 0
        self.sjs = 1
        self.stage = 0
        self.chain_index = 0
        self.resampling_indexes = np.arange(n_chains)

        self.threshold = threshold
        self.n_chains = n_chains
        self.likelihoods = np.zeros(n_chains)

        self.likelihood_name = likelihood_name
        self._llk_index = out_varnames.index(likelihood_name)
        self.discrete = np.concatenate([[v.dtype in discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # create initial population
        self.population = []
        self.array_population = np.zeros(n_chains)
        start = model.test_point

        init_rnd = {}
        for v in vars:
            if pm.util.is_transformed_name(v.name):
                trans = v.distribution.transform_used.forward_val
                init_rnd[v.name] = trans(v.distribution.dist.random(
                    size=self.n_chains, point=start))
            else:
                init_rnd[v.name] = v.random(size=self.n_chains, point=start)

        for i in range(self.n_chains):
            self.population.append(pm.Point({v.name: init_rnd[v.name][i] for v in vars},
                                            model=model))
        self.chain_previous_lpoint = copy.deepcopy(self.population)

        shared = make_shared_replacements(vars, model)
        self.logp_forw = logp_forw(out_vars, vars, shared)
        self.check_bnd = logp_forw([model.varlogpt], vars, shared)

        # epsilon computation, drawing samples from the prior, mean of the first population
        #self.epsilon = np.array([d[str(v)] for d in self.population for v in vars]).mean()
        # epsilon computation, drawing samples from the prior, a quantile of the first population
        #self.epsilon = mquantiles([d[str(v)] for d in self.population for v in vars], 
        #              prob=[0.99])[0]
        self.epsilon_max = mquantiles([d[str(v)] for d in self.population for v in vars], 
                           prob=[0.99])[0]
        self.epsilons = np.linspace(self.epsilon_max, self.minimum_eps, self.ladder)

        super(SMC, self).__init__(vars, out_vars, shared, epsilons)

    def astep(self, q0):
        """[summary]
        
        [description]
        
        Arguments:
            q0 {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        observed = self.observed
        size = len(observed) 
        scale = observed.std()
        mean = observed.mean()
        epsilon = self.epsilon
        if self.stage == 0:
            logp_prior = self.logp_forw(q0)
            l_new = [q0, np.exp(logp_prior[1])]
            q_new = q0
            sum_stat = self.sum_stat

        # tuning step
        else:
            #delta = self.proposal_samples_array[self.stage_sample, :] * self.scaling
            #accepted = self.accepted

            #print(accepted, self.n_steps)
            #for _ in range(self.n_steps):
            #while True:
            for _ in range(5000):
                delta = self.proposal_dist(1)[0,0]
                q_prop = q0 + delta
                #print(q0, delta)
                y_q = np.random.normal(loc=q_prop, scale=scale, size=size) # simulator
                sum_stat = y_q.mean()
                if abs(mean - sum_stat) < epsilon: # distance function, summary statistic
                    q_new = q_prop
                    logp_prior = self.logp_forw(q_new)[1]
                    s = self.covariance * self.scaling
                    logp_proposal = self.proposal_dist.logp(s, q_new)
                    l_new = [q_new, np.exp(logp_prior - logp_proposal)]
                    self.chain_previous_lpoint[self.chain_index] = l_new
                    break
                else:
                    q_new = q0
                    l_new = self.chain_previous_lpoint[self.chain_index]
                self.all_sum_stats.append(sum_stat)

        return q_new, l_new, sum_stat

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
        sj : float
            Mean of unnormalized weights
        """
        weights_un = self.likelihoods
        weights = weights_un / np.sum(weights_un)
        new_beta, old_beta = 1,2

        #low_beta = old_beta = self.beta
        #up_beta = 2.
        #rN = int(len(self.likelihoods) * self.threshold)

        #while up_beta - low_beta > 1e-6:
        #    new_beta = (low_beta + up_beta) / 2.
        #    weights_un = np.exp((new_beta - old_beta) * (self.likelihoods - self.likelihoods.max()))
        #    ESS = int(1 / np.sum(weights ** 2))
            #ESS = int(1 / np.max(weights))
        #    if ESS == rN:
        #        break
        #    elif ESS < rN:
        #        up_beta = new_beta
        #    else:
        #        low_beta = new_beta

        return new_beta, old_beta, weights, np.mean(weights_un)

    def calc_covariance(self):
        """Calculate trace covariance matrix based on importance weights.

        Returns
        -------
        cov : :class:`numpy.ndarray`
            weighted covariances (NumPy > 1.10. required)
        """
        cov = np.cov(self.array_population, aweights=self.weights.ravel(), bias=False, rowvar=0)
        print(cov)
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "n_chains" is too small!')
        return np.atleast_2d(cov)

    def select_end_points(self, mtrace):
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
        array_population = np.zeros((self.n_chains, self.ordering.size))
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

    def get_chain_previous_lpoint(self, mtrace):
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
        array_population = np.zeros((self.n_chains, self.lordering.size))
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

    def resample(self):
        """Resample pdf based on importance weights. based on Kitagawas deterministic resampling
        algorithm.

        Returns
        -------
        outindex : :class:`numpy.ndarray`
            Array of resampled trace indexes
        """
        parents = np.arange(self.n_chains)
        N_childs = np.zeros(self.n_chains, dtype=int)

        cum_dist = np.cumsum(self.weights)
        u = (parents + np.random.rand()) / self.n_chains
        j = 0
        for i in parents:
            while u[i] > cum_dist[j]:
                j += 1

            N_childs[j] += 1

        indx = 0
        outindx = np.zeros(self.n_chains, dtype=int)
        for i in parents:
            if N_childs[i] > 0:
                for j in range(indx, (indx + N_childs[i])):
                    outindx[j] = parents[i]

            indx += N_childs[i]

        return outindx


def sample_smc(samples=1000, chains=100, step=None, start=None, homepath=None, stage=0, cores=1,
               tune_interval=10, progressbar=False, model=None, random_seed=-1, rm_flag=True, 
               observed=None, epsilons=None, ladder=None, minimum_eps=None,**kwargs):
    """Sequential Monte Carlo sampling

    Samples the solution space with `chains` of Metropolis chains, where each chain has `n_steps`=`samples`/`chains`
    iterations. Once finished, the sampled traces are evaluated:

    (1) Based on the likelihoods of the final samples, chains are weighted
    (2) the weighted covariance of the ensemble is calculated and set as new proposal distribution
    (3) the variation in the ensemble is calculated and also the next tempering parameter (`beta`)
    (4) New `chains` Markov chains are seeded on the traces with high weight for n_steps iterations
    (5) Repeat until `beta` > 1.

    Parameters
    ----------
    samples : int
        The number of samples to draw from the last stage, i.e. the posterior. Defaults to 1000.
        The number of samples should be a multiple of `chains`, otherwise the returned number of
        draws will be the lowest closest multiple of `chains`.
    chains : int
        Number of chains used to store samples in backend.
    step : :class:`SMC`
        SMC initialization object
    start : List of dictionaries
        with length of (`chains`). Starting points in parameter space (or partial point)
        Defaults to random draws from variables (defaults to empty dict)
    homepath : string
        Result_folder for storing stages, will be created if not existing.
    stage : int
        Stage where to start or continue the calculation. It is possible to continue after completed
        stages (`stage` should be the number of the completed stage + 1). If None the start will be at
        `stage=0`.
    cores : int
        The number of cores to be used in parallel. Be aware that Theano has internal
        parallelization. Sometimes this is more efficient especially for simple models.
        `step.n_chains / cores` has to be an integer number!
    tune_interval : int
        Number of steps to tune for. Defaults to 10.
    progressbar : bool
        Flag for displaying a progress bar
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
    """
    warnings.warn(EXPERIMENTAL_WARNING)

    n_chains = chains
    if 'n_chains' in kwargs:
        n_chains = kwargs['n_chains']
        warnings.warn(
            "The n_chains argument has been deprecated. Use chains instead.",
            DeprecationWarning)
    remainder = samples % n_chains
    if remainder != 0:
        warnings.warn("'samples' {} is not a multiple of 'chains' {}. Hence, you will get {} "
                      "draws from the posterior".format(samples, n_chains, samples - remainder))

    model = modelcontext(model)

    if random_seed != -1:
        nr.seed(random_seed)

    if step is None:
        pm._log.info('Argument `step` is None. Auto-initialising step object '
                     'using given/default parameters.')
        step = SMC(n_chains=n_chains, tune_interval=tune_interval, model=model, 
                   observed=observed, epsilons=epsilons, ladder=ladder, minimum_eps=minimum_eps)
    print(step.epsilons)

    if homepath is None:
        raise TypeError('Argument `homepath` should be path to result_directory.')

    if 'n_jobs' in kwargs:
        cores = kwargs['n_jobs']
        warnings.warn(
            "The n_jobs argument has been deprecated. Use cores instead.",
            DeprecationWarning)
    if cores > 1:
        if not (step.n_chains / float(cores)).is_integer():
            raise TypeError('chains / cores has to be a whole number!')

    if start is not None:
        if len(start) != step.n_chains:
            raise TypeError('Argument `start` should have dicts equal the '
                            'number of chains (`step.n_chains`)')
        else:
            step.population = start

    if not any(step.likelihood_name in var.name for var in model.deterministics):
        raise TypeError('Model (deterministic) variables need to contain a variable {} as defined '
                        'in `step`.'.format(step.likelihood_name))

    stage_handler = atext.TextStage(homepath)

    if progressbar and cores > 1:
        progressbar = False

    if stage == 0:
        # continue or start initial stage
        step.stage = stage
        draws = 1
    else:
        step = stage_handler.load_atmip_params(stage, model=model)
        draws = step.n_steps

    stage_handler.clean_directory(stage, None, rm_flag)

    chains = stage_handler.recover_existing_results(stage, draws, step)

    with model:
        #while step.beta < 1:
        for step.epsilon in step.epsilons:
            print(step.epsilon)
            if step.stage == 0:
                # Initial stage
                pm._log.info('Sample initial stage: ...')
                draws = 1
            else:
                draws = step.n_steps

            pm._log.info('Beta: %f Stage: %i' % (step.beta, step.stage))

            # Metropolis sampling intermediate stages
            chains = stage_handler.clean_directory(step.stage, chains, rm_flag)
            sample_args = {'draws': draws,
                           'step': step,
                           'stage_path': stage_handler.stage_path(step.stage),
                           'progressbar': progressbar,
                           'model': model,
                           'n_jobs': cores,
                           'chains': chains}

            _iter_parallel_chains(**sample_args)

            mtrace = stage_handler.load_multitrace(step.stage)

            step.population, step.array_population, step.likelihoods = step.select_end_points(
                mtrace)
            step.beta, step.old_beta, step.weights, sj = step.calc_beta()
            #step.beta, step.old_beta, step.weights, sj = 1,2,3,4
            step.sjs *= sj

            #if step.beta > 1.:
            if step.epsilon == step.epsilons[-1]:
                pm._log.info('Beta > 1.: %f' % step.beta)
                step.beta = 1.
                stage_handler.dump_atmip_params(step)
                if stage == -1:
                    chains = []
                else:
                    chains = None
            else:
                step.covariance = step.calc_covariance()
                step.proposal_dist = choose_proposal(step.proposal_name, scale=step.covariance)
                step.resampling_indexes = step.resample()
                stage_handler.dump_atmip_params(step)

                step.stage += 1
                del(mtrace)

        # Metropolis sampling final stage
        pm._log.info('Sample final stage')
        step.stage = -1
        chains = stage_handler.clean_directory(step.stage, chains, rm_flag)
        weights_un = step.likelihoods
        step.weights = weights_un / np.sum(weights_un)
        step.covariance = step.calc_covariance()
        step.proposal_dist = choose_proposal(step.proposal_name, scale=step.covariance)
        step.resampling_indexes = step.resample()
        step.chain_previous_lpoint = step.get_chain_previous_lpoint(mtrace)


        if samples < n_chains:
            samples = 1
        else:
            samples = int(samples / n_chains)
        sample_args['draws'] = samples
        sample_args['step'] = step
        sample_args['stage_path'] = stage_handler.stage_path(step.stage)
        sample_args['chains'] = chains
        _iter_parallel_chains(**sample_args)

        stage_handler.dump_atmip_params(step)

        model.marginal_likelihood = step.sjs
        return stage_handler.create_result_trace(step.stage,
                                                 idxs=range(samples),
                                                 step=step,
                                                 model=model)


def _sample(draws, step=None, start=None, trace=None, chain=0, progressbar=True, model=None,
            random_seed=-1):

    sampling = _iter_sample(draws, step, start, trace, chain, model, random_seed)

    if progressbar:
        sampling = tqdm(sampling, total=draws)

    try:
        for strace in sampling:
            pass

    except KeyboardInterrupt:
        pass

    return chain


def _iter_sample(draws, step, start=None, trace=None, chain=0, model=None, random_seed=-1):
    """Modified from :func:`pymc3.sampling._iter_sample` to be more efficient with SMC algorithm."""
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
    trace.setup(draws, chain)
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

    Returns
    -------
    chain : int
        Index of chain that has been sampled
    """
    return _sample(*work)


def _iter_parallel_chains(draws, step, stage_path, progressbar, model, n_jobs, chains=None):
    """Do Metropolis sampling over all the chains with each chain being sampled 'draws' times.
    Parallel execution according to n_jobs.
    """
    if chains is None:
        chains = range(step.n_chains)

    pm._log.info('Initializing chain traces ...')

    max_int = np.iinfo(np.int32).max
    random_seeds = nr.randint(1, max_int, size=len(chains))

    pm._log.info('Sampling ...')
    work = [(draws,
             step,
             step.population[step.resampling_indexes[chain]],
             atext.TextChain(stage_path, model=model),
             chain,
             False,
             model,
             rseed) for chain, rseed in
            zip(chains, random_seeds)]

    if draws < 10:
        chunksize = n_jobs
    else:
        chunksize = 1

    p = atext.paripool(_work_chain, work, chunksize=chunksize, nprocs=n_jobs)

    if n_jobs == 1 and progressbar:
        p = tqdm(p, total=len(chains))

    for _ in p:
        pass


def tune(acc_rate):
    """Tune adaptively based on the acceptance rate.

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
