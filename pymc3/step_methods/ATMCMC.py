'''
Created on March, 2016

@author: Hannes Vasyura-Bathke
'''

import numpy as num
import pymc3 as pm

import theano

from pymc3.theanof import make_shared_replacements, join_nonshared_inputs
from pymc3.step_methods.metropolis import MultivariateNormalProposal as MvNPd
from numpy.random import seed
from joblib import Parallel, delayed

__all__ = ['ATMCMC', 'ATMIP_sample']


class ATMCMC(pm.arraystep.ArrayStepShared):
    """
    Transitional Marcov-Chain Monte-Carlo
    following: Ching & Chen 2007: Transitional Markov chain Monte Carlo method
                for Bayesian model updating, model class selection and model
                averaging
                Journal of Engineering Mechanics 2007
                DOI:10.1016/(ASCE)0733-9399(2007)133:7(816)
    Creates initial samples and framework around the CATMIP parameters
    Parameters
    ----------
    vars : list
        List of variables for sampler
    N_chains : (integer) Number of chains per stage has to be a large number
               of number of njobs (processors to be used) on the machine.
    Covariance - Initial Covariance matrix for proposal distribution,
        if None - identity matrix taken
    proposal_dist - Type of proposal distribution, see metropolis.py for
                    options
    model : PyMC Model
        Optional model for sampling step.
        Defaults to None (taken from context).
    """
    default_blocked = True

    def __init__(self, vars=None, Covariance=None, scaling=1., N_chains=100,
                 tune=True, tune_interval=100, model=None,
                 proposal_dist=MvNPd,
                 N_steps=1000, **kwargs):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        if Covariance is None:
            self.Covariance = num.eye(sum(v.dsize for v in vars))
        self.scaling = num.atleast_1d(scaling)
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.proposal_dist = proposal_dist(self.Covariance)
        self.accepted = 0
        self.beta = 0
        self.stage = 0
        self.N_chains = N_chains
        self.likelihoods = []
        self.discrete = num.ravel(
            [[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()
        # create initial population
        self.population = []
        self.array_population = num.zeros(N_chains)
        for i in range(self.N_chains):
            dummy = pm.Point({v.name: v.random() for v in vars},
                                                            model=model)
            self.population.append(dummy)

        shared = make_shared_replacements(vars, model)
        self.logp_forw = logp_forw(model.logpt, vars, shared)
        self.check_bnd = logp_forw(model.varlogpt, vars, shared)
        self.delta_logp = pm.metropolis.delta_logp(model.logpt, vars, shared)

        super(ATMCMC, self).__init__(vars, shared)

    def astep(self, q0):
        if self.stage == 0:
            self.likelihoods.append(self.logp_forw(q0))
            q_new = q0
        else:
            if not self.steps_until_tune and self.tune:
                # Tune scaling parameter
                R = self.accepted / float(self.tune_interval)
                # a and b after Muto & Beck 2008 .
                a = 1. / 9
                b = 8. / 9
                self.scaling = num.power((a + (b * R)), 2)

                # Reset counter
                self.steps_until_tune = self.tune_interval
                self.accepted = 0

            delta = self.proposal_dist() * self.scaling
            check_bnd = True
            while check_bnd:
                if self.any_discrete:
                    if self.all_discrete:
                        delta = round(delta, 0)
                        q0 = q0.astype(int)
                        q = (q0 + delta).astype(int)
                        varlogp = self.check_bnd(q)
                    else:
                        delta[self.discrete] = round(
                                        delta[self.discrete], 0).astype(int)
                        q = q0 + delta
                        varlogp = self.check_bnd(q[self.discrete].astype(int))
                else:
                    q = q0 + delta
                    varlogp = self.check_bnd(q)

                if num.isfinite(varlogp):
                    check_bnd = False
                else:
                    delta = self.proposal_dist() * self.scaling

            q = q0 + delta
            q_new = pm.metropolis.metrop_select(self.delta_logp(q, q0), q, q0)

            if q_new is q:
                self.accepted += 1

            self.steps_until_tune -= 1
        return q_new

    def calc_beta(self):
        '''Calculate next tempering beta and importance weights.
            Inputs: beta(m) - tempering parameter (float 0 <= beta <= 1)
                    log_likelihoods of stage samples
                                (Ndarray[N_chains * N_sample])
            returns: beta(m+1), NdArray Importance weights'''

        low_beta = self.beta
        up_beta = 2.
        old_beta = self.beta

        while up_beta - low_beta > 1e-6:
            current_beta = (low_beta + up_beta) / 2.
            temp = num.exp((current_beta - self.beta) * \
                           (self.likelihoods - self.likelihoods.max()))
            cov_temp = num.std(temp) / num.mean(temp)
            if cov_temp > 1:
                up_beta = current_beta
            else:
                low_beta = current_beta

        beta = current_beta
        weights = temp / num.sum(temp)
        return beta, old_beta, weights

    def calc_covariance(self):
        '''Calculate covariance based on importance weights.'''
        return num.cov(self.array_population, aweights=self.weights, rowvar=0)

    def select_end_points(self, mtrace):
        '''Read trace results and take end points for each chain and set as
           start population for the next stage.
            Input: Multitrace object
            Returns: population
                     array_population
                     likelihoods'''
        array_population = num.zeros((self.N_chains,
                                                     self.ordering.dimensions))
        N_steps = len(mtrace)

        if self.stage > 0:
            # collect end points of each chain and put into array
            for var, slc, _, _ in self.ordering.vmap:
                array_population[:, slc] = mtrace.get_values(
                                    varname=var,
                                    burn=N_steps - 1,
                                    combine=True)

            population = []
            likelihoods = []
            # collect end points of each chain
            for i in range(self.N_chains):
                point = mtrace.point(-1, chain=i)
                likelihoods.append(point.pop('llk'))
                population.append(point)

            likelihoods = num.array(likelihoods)
        else:
            # for initial stage only one trace that contains all points for
            # each chain
            population = self.population
            likelihoods = mtrace.get_values('llk')
            for var, slc, _, _ in self.ordering.vmap:
                array_population[:, slc] = mtrace.get_values(var)

        return population, array_population, likelihoods

    def resample(self):
        '''Resample pdf based on importance weights.
           based on Kitagawas deterministic resampling algorithm.
            Input: self.N_chains - Integer Number of samples
                   self.weights - Ndarray of importance weights
            returns: Ndarray of resampled trace indexes'''
        parents = num.array(range(self.N_chains))
        N_childs = num.zeros(self.N_chains, dtype=int)

        cum_dist = num.cumsum(self.weights)
        aux = num.random.rand(1)
        u = parents + aux
        u /= self.N_chains
        j = 0
        for i in parents:
            while u[i] > cum_dist[j]:
                j += 1

            N_childs[j] += 1

        indx = 0
        outindx = num.zeros(self.N_chains, dtype=int)
        for i in parents:
            if N_childs[i] > 0:
                for j in range(indx, (indx + N_childs[i])):
                    outindx[j] = parents[i]

            indx += N_childs[i]

        return outindx


def ATMIP_sample(N_steps, step=None, start=None, trace=None, chain=0,
                  stage=None, njobs=1, tune=None, progressbar=False,
                  model=None, random_seed=None):
    """
    (C)ATMIP sampling algorithm from Minson et al. 2013:
    Bayesian inversion for finite fault earthquake source models I-
        Theory and algorithm
    (without cascading- C)
    Parameters
    ----------

    N_steps : int
        The number of samples to draw for each Markov-chain per stage
    step : function from TMCMC initialisation
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict)
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
    njobs : int recommended njobs=1 Some internal parallelisation in Theano?
                njobs=1  ca. 3 times faster with 12 cores on station than
                njobs=10
        step.N_chains / njobs has to be an integer number!
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    trace : result_folder for storing stages
    progressbar : bool
        Flag for progress bar
    model : Model (optional if in `with` context) has to contain deterministic
            variable 'llk' that contains model likelihood
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Returns
    -------
    MultiTrace object with access to sampling values
    """

    model = pm.modelcontext(model)
    N_steps = int(N_steps)
    seed(random_seed)

    if N_steps < 1:
        raise ValueError('Argument `N_steps` should be above 0.')

    if step is None:
        raise Exception('Argument `step` has to be a TMCMC step object.')

    if trace is None:
        raise Exception('Argument `trace` should be either sqlite or text \
                        backend object.')

    if start is None:
        start = {}

    if stage is not None:
        step.stage = stage

    if progressbar:
        verbosity = 5
        if njobs > 1:
            progressbar = False
    else:
        verbosity = 0

    homepath = trace

    with model:
        with Parallel(n_jobs=njobs, verbose=verbosity) as parallel:
            while step.beta < 1.:
                print 'Beta:', str(step.beta), 'Stage', str(step.stage)
                if step.stage == 0:
                    # Initial stage
                    print 'Sample initial stage: ...'
                    stage_path = homepath + '/stage_' + str(step.stage)
                    trace = pm.backends.Text(stage_path, model=model)
                    initial = _iter_initial(step, chain=chain, trace=trace,
                                            start=None)
                    progress = pm.progressbar.progress_bar(step.N_chains)
                    try:
                        for i, strace in enumerate(initial):
                            if progressbar:
                                progress.update(i)
                    except KeyboardInterrupt:
                        strace.close()
                    mtrace = pm.backends.base.MultiTrace([strace])
                    step.population, step.array_population, step.likelihoods = \
                                            step.select_end_points(mtrace)
                    step.beta, step.old_beta, step.weights = step.calc_beta()
                    step.Covariance = step.calc_covariance()
                    step.res_indx = step.resample()
                    step.stage += 1
                    del(strace, mtrace, trace)
                else:
                    # Metropolis sampling intermediate stages
                    stage_path = homepath + '/stage_' + str(step.stage)
                    step.proposal_dist = MvNPd(step.Covariance)
                    sample_args = {
                            'draws': N_steps,
                            'step': step,
                            'stage_path': stage_path,
                            'progressbar': progressbar,
                            'model': model}
                    mtrace = _iter_parallel_chains(parallel, **sample_args)

                    step.population, step.array_population, step.likelihoods = \
                                            step.select_end_points(mtrace)
                    step.beta, step.old_beta, step.weights = step.calc_beta()
                    if step.beta > 1.:
                        print 'Beta above 1.:', str(step.beta)
                        break

                    step.Covariance = step.calc_covariance()
                    step.res_indx = step.resample()
                    step.stage += 1

            # Metropolis sampling final stage
            print 'Sample final stage'
            stage_path = homepath + '/stage_final'
            temp = num.exp((1 - step.old_beta) * \
                               (step.likelihoods - step.likelihoods.max()))
            step.weights = temp / num.sum(temp)
            step.Covariance = step.calc_covariance()
            step.proposal_dist = MvNPd(step.Covariance)
            step.res_indx = step.resample()

            sample_args['step'] = step
            sample_args['stage_path'] = stage_path
            mtrace = _iter_parallel_chains(parallel, **sample_args)

            return mtrace


def _iter_initial(step, chain=0, trace=None, model=None, start=None):
    '''Yields generator for Iteration over initial stage similar to
       _iter_sample, just different input to loop over.'''

    if start is None:
        start = {}

    strace = pm.sampling._choose_backend(trace, chain, model=model)
    l_tr = len(strace)

    if l_tr > 0:
        pm.sampling._soft_update(start, strace.point(-1))
    else:
        pm.sampling._soft_update(start, step.population[0])

    strace.setup(step.N_chains, chain=0)
    for i in range(l_tr, step.N_chains):
        point = step.step(step.population[i])
        strace.record(point)
        yield strace
    else:
        strace.close()


def _iter_serial_chains(draws, step=None, stage_path=None,
                        progressbar=True, model=None):
    '''Do Metropolis sampling over all the chains with each chain being
       sampled 'draws' times. Serial execution one after another.'''
    mtraces = []
    progress = pm.progressbar.progress_bar(step.N_chains)
    for chain in range(step.N_chains):
        if progressbar:
            progress.update(chain)
        trace = pm.backends.Text(stage_path, model=model)
        mtraces.append(pm.sampling._sample(
                draws=draws,
                step=step,
                chain=chain,
                trace=trace,
                model=model,
                progressbar=False,
                start=step.population[step.res_indx[chain]]))

    return pm.sampling.merge_traces(mtraces)


def _iter_parallel_chains(parallel, **kwargs):
    '''Do Metropolis sampling over all the chains with each chain being
       sampled 'draws' times. Parallel execution according to after another.'''
    stage_path = kwargs.pop('stage_path')
    step = kwargs['step']
    chains = list(range(step.N_chains))
    trace_list = []
    for chain in chains:
        trace_list.append(pm.backends.Text(stage_path))

    traces = parallel(delayed(
                    pm.sampling._sample)(
                        chain=chain,
                        trace=trace_list[chain],
                        start=step.population[step.res_indx[chain]],
                        **kwargs) for chain in chains)
    return pm.sampling.merge_traces(traces)


def logp_forw(logp, vars, shared):
    [logp0], inarray0 = join_nonshared_inputs([logp], vars, shared)
    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f
