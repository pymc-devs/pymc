'''
Created on March, 2016

@author: Hannes Vasyura-Bathke
'''

import numpy as np
from .arraystep import ArrayStep, ArrayStepShared, Competence
from ..model import modelcontext, Point
from ..theanof import inputvars
from ..vartypes import discrete_types
from ..step_methods import metropolis
from ..blocking import DictToArrayBijection
from ..backends import Text
from ..backends.base import merge_traces, BaseTrace, MultiTrace
from ..backends.ndarray import NDArray
from .. import backends
from ..progressbar import progress_bar

import os
import theano

from ..theanof import make_shared_replacements, join_nonshared_inputs
from ..step_methods.metropolis import MultivariateNormalProposal as MvNPd
from numpy.random import seed
from joblib import Parallel, delayed


__all__ = ['ATMCMC', 'ATMIP_sample']


class ATMCMC(ArrayStepShared):
    """
    Adaptive Transitional Markov-Chain Monte-Carlo
    following: Ching & Chen 2007: Transitional Markov chain Monte Carlo method
                for Bayesian model updating, model class selection and model
                averaging
                Journal of Engineering Mechanics 2007
                DOI:10.1016/(ASCE)0733-9399(2007)133:7(816)
    http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399%282007%29133:7%28816%29

    Creates initial samples and framework around the (C)ATMIP parameters

    Parameters
    ----------
    vars : list
        List of variables for sampler
    n_chains : (integer) Number of chains per stage has to be a large number
               of number of njobs (processors to be used) on the machine.
    covariance : (n_chains x n_chains) Numpy array
                 Initial Covariance matrix for proposal distribution,
                 if None - identity matrix taken
    likelihood_name : string
                      name of the determinsitic variable that contains the
                      model likelihood - defaults to 'like'
    proposal_dist : pymc3 object
                    Type of proposal distribution, see metropolis.py for
                    options
    coef_variation : scalar, float
                     Coefficient of variation, determines the change of beta
                     from stage to stage, i.e.indirectly the number of stages,
                     low coef_variation --> slow beta change,
                                         results in many stages and vice verca
                     (default: 1.)
    check_bound : boolean
                  check if current sample lies outside of variable definition
                  speeds up computation as the forward model wont be executed
                  default: True
    model : PyMC Model
        Optional model for sampling step.
        Defaults to None (taken from context).
    """
    default_blocked = True

    def __init__(self, vars=None, covariance=None, scaling=1., n_chains=100,
                 tune=True, tune_interval=100, model=None, check_bound=True,
                 likelihood_name='like', proposal_dist=MvNPd,
                 coef_variation=1., **kwargs):

        model = modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = inputvars(vars)

        if covariance is None:
            self.covariance = np.eye(sum(v.dsize for v in vars))
        self.scaling = np.atleast_1d(scaling)
        self.tune = tune
        self.check_bnd = check_bound
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval

        self.proposal_dist = proposal_dist(self.covariance)
        self.proposal_samples_array = self.proposal_dist(n_chains)

        self.stage_sample = 0
        self.accepted = 0

        self.beta = 0
        self.stage = 0
        self.coef_variation = coef_variation
        self.n_chains = n_chains
        self.likelihoods = []
        self.likelihood_name = likelihood_name
        self.discrete = np.concatenate(
            [[v.dtype in discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # create initial population
        self.population = []
        self.array_population = np.zeros(n_chains)
        for i in range(self.n_chains):
            dummy = Point({v.name: v.random() for v in vars},
                          model=model)
            self.population.append(dummy)

        shared = make_shared_replacements(vars, model)
        self.logp_forw = logp_forw(model.logpt, vars, shared)
        self.check_bnd = logp_forw(model.varlogpt, vars, shared)
        self.delta_logp = metropolis.delta_logp(model.logpt, vars, shared)

        super(ATMCMC, self).__init__(vars, shared)

    def astep(self, q0):
        if self.stage == 0:
            self.likelihoods.append(self.logp_forw(q0))
            q_new = q0
        else:
            if not self.stage_sample:
                self.proposal_samples_array = self.proposal_dist(self.n_steps)

            if not self.steps_until_tune and self.tune:
                # Tune scaling parameter
                self.scaling = tune(self.accepted /
                                    float(self.tune_interval))
                # Reset counter
                self.steps_until_tune = self.tune_interval
                self.accepted = 0

            delta = self.proposal_samples_array[self.stage_sample, :] * \
                self.scaling

            if self.any_discrete:
                if self.all_discrete:
                    delta = np.round(delta, 0)
                    q0 = q0.astype(int)
                    q = (q0 + delta).astype(int)
                else:
                    delta[self.discrete] = np.round(
                        delta[self.discrete], 0).astype(int)
                    q = q0 + delta
                    q = q[self.discrete].astype(int)
            else:
                q = q0 + delta

            if self.check_bnd:
                varlogp = self.check_bnd(q)
                if np.isfinite(varlogp):
                    q_new = metropolis.metrop_select(
                        self.beta * self.delta_logp(q, q0), q, q0)
                    if q_new is q:
                        self.accepted += 1
                else:
                    q_new = q0
            else:
                q_new = metropolis.metrop_select(
                    self.beta * self.delta_logp(q, q0), q, q0)
                if q_new is q:
                    self.accepted += 1

            self.steps_until_tune -= 1
            self.stage_sample += 1

            # reset sample counter
            if self.stage_sample == self.n_steps:
                self.stage_sample = 0

        return q_new

    def calc_beta(self):
        """
        Calculate next tempering beta and importance weights based on
        current beta and sample likelihoods.

        Returns
        -------
        beta(m+1) : scalar float tempering parameter of the next stage
        beta(m) : scalar float tempering parameter of the current stage
        weights : NdArray of importance weights (floats)
        """

        low_beta = self.beta
        up_beta = 2.
        old_beta = self.beta

        while up_beta - low_beta > 1e-6:
            current_beta = (low_beta + up_beta) / 2.
            temp = np.exp((current_beta - self.beta) *
                          (self.likelihoods - self.likelihoods.max()))
            cov_temp = np.std(temp) / np.mean(temp)
            if cov_temp > self.coef_variation:
                up_beta = current_beta
            else:
                low_beta = current_beta

        beta = current_beta
        weights = temp / np.sum(temp)
        return beta, old_beta, weights

    def calc_covariance(self):
        """
        Calculate trace covariance matrix based on importance weights.

        Returns
        -------
        Ndarray of weighted covariances (NumPy > 1.10. required)
        """
        return np.cov(self.array_population, aweights=self.weights, rowvar=0)

    def select_end_points(self, mtrace):
        """
        Read trace results and take end points for each chain and set as
        start population for the next stage.
        Parameters
        -------

        mtrace : Multitrace pymc3 object

        Returns
        -------
        population : List of pymc3.Point - objects,
        array_population : Ndarray of trace end-points
        likelihoods : Ndarray of likelihoods of the trace end-points
        """

        array_population = np.zeros((self.n_chains,
                                     self.ordering.dimensions))
        n_steps = len(mtrace)
        bij = DictToArrayBijection(self.ordering, self.population[0])

        if self.stage > 0:
            # collect end points of each chain and put into array
            for var, slc, shp, _ in self.ordering.vmap:
                if len(shp) == 0:
                    array_population[:, slc] = np.atleast_2d(
                        mtrace.get_values(varname=var,
                                          burn=n_steps - 1,
                                          combine=True)).T
                else:
                    array_population[:, slc] = mtrace.get_values(
                        varname=var,
                        burn=n_steps - 1,
                        combine=True)
            # get likelihoods
            likelihoods = mtrace.get_values(varname=self.likelihood_name,
                                            burn=n_steps - 1,
                                            combine=True)
            population = []

            # map end array_endpoints to dict points
            for i in range(self.n_chains):
                population.append(bij.rmap(array_population[i, :]))

        else:
            # for initial stage only one trace that contains all points for
            # each chain
            likelihoods = mtrace.get_values(self.likelihood_name)
            for var, slc, shp, _ in self.ordering.vmap:
                if len(shp) == 0:
                    array_population[:, slc] = np.atleast_2d(
                        mtrace.get_values(varname=var)).T
                else:
                    array_population[:, slc] = mtrace.get_values(varname=var)

            population = []
            for i in range(self.n_chains):
                population.append(bij.rmap(array_population[i, :]))

        return population, array_population, likelihoods

    def resample(self):
        """
        Resample pdf based on importance weights.
        based on Kitagawas deterministic resampling algorithm.

        Returns
        -------
        outindex : Ndarray of resampled trace indexes
        """
        parents = np.array(range(self.n_chains))
        N_childs = np.zeros(self.n_chains, dtype=int)

        cum_dist = np.cumsum(self.weights)
        aux = np.random.rand(1)
        u = parents + aux
        u /= self.n_chains
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


def ATMIP_sample(n_steps, step=None, start=None, trace=None, chain=0,
                 stage=None, njobs=1, tune=None, progressbar=False,
                 model=None, random_seed=None):
    """
    (C)ATMIP sampling algorithm from Minson et al. 2013:
    Bayesian inversion for finite fault earthquake source models I-
        Theory and algorithm
    (without cascading- C)
    https://gji.oxfordjournals.org/content/194/3/1701.full
    Samples the solution space with n_chains of Metropolis chains, where each
    chain has n_steps iterations. Once finished, the sampled traces are
    evaluated:
    (1) Based on the likelihoods of the final samples, chains are weighted
    (2) the weighted covariance of the ensemble is calculated and set as new
        proposal distribution
    (3) the variation in the ensemble is calculated and the next tempering
        parameter (beta) calculated
    (4) New n_chains Metropolis chains are seeded on the traces with high
        weight for n_steps iterations
    (5) Repeat until beta > 1.

    Parameters
    ----------

    n_steps : int
        The number of samples to draw for each Markov-chain per stage
    step : function from TMCMC initialisation
    start : List of dicts with length(n_chains)
        Starting points in parameter space (or partial point)
        Defaults to random draws from variables (defaults to empty dict)
    trace : backend
        This should be a backend instance.
        Passing either "text" or "sqlite" is taken as a shortcut to set
        up the corresponding backend (with "mcmc" used as the base
        name).
    chain : int
        Chain number used to store sample in backend. If `njobs` is
        greater than one, chain numbers will start here.
    stage : int
        Stage where to start or continue the calculation. If None the start
        will be at stage = 0.
    njobs : int
        The number of cores to be used in parallel. Be aware that theano has
        internal parallelisation. Sometimes this is more efficient especially
        for simple models.
        step.n_chains / njobs has to be an integer number!
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    trace : result_folder for storing stages, will be created if not existing
    progressbar : bool
        Flag for progress bar
    model : Model (optional if in `with` context) has to contain deterministic
            variable 'name defined under step.likelihood_name' that contains
            model likelihood
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Returns
    -------
    MultiTrace object with access to sampling values
    """

    model = modelcontext(model)
    step.n_steps = int(n_steps)
    seed(random_seed)

    if n_steps < 1:
        raise ValueError('Argument `n_steps` should be above 0.')

    if step is None:
        raise Exception('Argument `step` has to be a TMCMC step object.')

    if trace is None:
        raise Exception('Argument `trace` should be either sqlite or text '
                        'backend object.')

    if start is not None:
        if len(start) != step.n_chains:
            raise Exception('Argument `start` should have dicts equal the '
                            'number of chains (step.N-chains)')
        else:
            step.population = start

    if stage is not None:
        step.stage = stage

    if not any(
            step.likelihood_name in var.name for var in model.deterministics):
        raise Exception('Model (deterministic) variables need to contain '
                        'a variable `' + step.likelihood_name + '` as '
                        'defined in `step`.')

    if progressbar:
        verbosity = 5
    else:
        verbosity = 0

    homepath = trace

    if not os.path.exists(homepath):
        os.mkdir(homepath)

    with model:
        with Parallel(n_jobs=njobs, verbose=verbosity) as parallel:
            while step.beta < 1.:
                print('Beta: ' + str(step.beta), ' Stage: ' + str(step.stage))
                if step.stage == 0:
                    # Initial stage
                    print('Sample initial stage: ...')
                    stage_path = homepath + '/stage_' + str(step.stage)
                    trace = Text(stage_path, model=model)
                    initial = _iter_initial(step, chain=chain, trace=trace)
                    progress = progress_bar(step.n_chains)
                    try:
                        for i, strace in enumerate(initial):
                            if progressbar:
                                progress.update(i)
                    except KeyboardInterrupt:
                        strace.close()
                    mtrace = MultiTrace([strace])
                    step.population, step.array_population, step.likelihoods = \
                        step.select_end_points(mtrace)
                    step.beta, step.old_beta, step.weights = step.calc_beta()
                    step.covariance = step.calc_covariance()
                    step.res_indx = step.resample()
                    step.stage += 1
                    del(strace, mtrace, trace)
                else:
                    if progressbar and njobs > 1:
                        progressbar = False
                    # Metropolis sampling intermediate stages
                    stage_path = homepath + '/stage_' + str(step.stage)
                    step.proposal_dist = MvNPd(step.covariance)

                    sample_args = {
                        'draws': n_steps,
                        'step': step,
                        'stage_path': stage_path,
                        'progressbar': progressbar,
                        'model': model}
                    mtrace = _iter_parallel_chains(parallel, **sample_args)

                    step.population, step.array_population, step.likelihoods = \
                        step.select_end_points(mtrace)
                    step.beta, step.old_beta, step.weights = step.calc_beta()
                    step.stage += 1

                    if step.beta > 1.:
                        print('Beta > 1.: ' + str(step.beta))
                        step.beta = 1.
                        break

                    step.covariance = step.calc_covariance()
                    step.res_indx = step.resample()

            # Metropolis sampling final stage
            print('Sample final stage')
            stage_path = homepath + '/stage_final'
            temp = np.exp((1 - step.old_beta) *
                          (step.likelihoods - step.likelihoods.max()))
            step.weights = temp / np.sum(temp)
            step.covariance = step.calc_covariance()
            step.proposal_dist = MvNPd(step.covariance)
            step.res_indx = step.resample()

            sample_args['step'] = step
            sample_args['stage_path'] = stage_path
            mtrace = _iter_parallel_chains(parallel, **sample_args)
            return mtrace


def _choose_backend(trace, chain, shortcuts=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]
    if trace is None:
        return NDArray(**kwds)

    if shortcuts is None:
        shortcuts = backends._shortcuts

    try:
        backend = shortcuts[trace]['backend']
        name = shortcuts[trace]['name']
        return backend(name, **kwds)
    except TypeError:
        return NDArray(vars=trace, **kwds)
    except KeyError:
        raise ValueError('Argument `trace` is invalid.')


def _iter_initial(step, chain=0, trace=None, model=None):
    """
    Yields generator for Iteration over initial stage similar to
    _iter_sample, just different input to loop over.
    """

    strace = sampling._choose_backend(trace, chain, model=model)
    # check if trace file already exists before setup
    filename = os.path.join(strace.name, 'chain-{}.csv'.format(chain))
    if os.path.exists(filename):
        strace.setup(step.n_chains, chain=0)
        l_tr = len(strace)
    else:
        strace.setup(step.n_chains, chain=0)
        l_tr = 0

    if l_tr == step.n_chains:
        # only return strace
        for i in range(1):
            print('Using results of previous run!')
            yield strace
    else:
        for i in range(l_tr, step.n_chains):
            point = step.step(step.population[i])
            strace.record(point)
            yield strace
        else:
            strace.close()


def _iter_serial_chains(draws, step=None, stage_path=None,
                        progressbar=True, model=None):
    """
    Do Metropolis sampling over all the chains with each chain being
    sampled 'draws' times. Serial execution one after another.
    """
    mtraces = []
    progress = progress_bar(step.n_chains)
    for chain in range(step.n_chains):
        if progressbar:
            progress.update(chain)
        trace = Text(stage_path, model=model)
        mtraces.append(sampling._sample(
            draws=draws,
            step=step,
            chain=chain,
            trace=trace,
            model=model,
            progressbar=False,
            start=step.population[step.res_indx[chain]]))

    return sampling.merge_traces(mtraces)


def _iter_parallel_chains(parallel, **kwargs):
    """
    Do Metropolis sampling over all the chains with each chain being
    sampled 'draws' times. Parallel execution according to njobs.
    """
    stage_path = kwargs.pop('stage_path')
    step = kwargs['step']
    chains = list(range(step.n_chains))
    trace_list = []

    print('Initialising chain traces ...')
    for chain in chains:
        trace_list.append(Text(stage_path))

    print('Sampling ...')
    traces = parallel(delayed(
        sampling._sample)(
        chain=chain,
        trace=trace_list[chain],
        start=step.population[step.res_indx[chain]],
        **kwargs) for chain in chains)
    return sampling.merge_traces(traces)


def tune(acc_rate):
    """
    Tune adaptively based on the acceptance rate.
    Parameters
    ----------
    acc_rate: scalar float
              Acceptance rate of the Metropolis sampling

    Returns
    -------
    scaling: scalar float
    """

    # a and b after Muto & Beck 2008 .
    a = 1. / 9
    b = 8. / 9
    return np.power((a + (b * acc_rate)), 2)


def logp_forw(logp, vars, shared):
    [logp0], inarray0 = join_nonshared_inputs([logp], vars, shared)
    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f
