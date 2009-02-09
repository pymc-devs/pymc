"""
Class MCMC, which fits probability models using Markov Chain Monte Carlo, is defined here.
"""

__all__ = ['MCMC']

from Model import Sampler
from Node import ZeroProbability
from StepMethods import StepMethodRegistry, assign_method, DrawFromPrior
from distributions import absolute_loss, squared_loss, chi_square_loss
import sys, time, pdb
import numpy as np
from utils import crawl_dataless

GuiInterrupt = 'Computation halt'
Paused = 'Computation paused'

class MCMC(Sampler):
    """
    This class fits probability models using Markov Chain Monte Carlo. Each stochastic variable
    is assigned a StepMethod object, which makes it take a single MCMC step conditional on the
    rest of the model. These step methods are called in turn.

      >>> A = MCMC(input, db, verbose=0)

      :Parameters:
        - input : module, list, tuple, dictionary, set, object or nothing.
            Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
            If nothing, all nodes are collected from the base namespace.
        - db : string
            The name of the database backend that will store the values
            of the stochastics and deterministics sampled during the MCMC loop.
        - verbose : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    Inherits all methods and attributes from Model. Subclasses must define the _loop method:

        - _loop(self, *args, **kwargs): Can be called after a sampling run is interrupted
            (by pausing, halting or a KeyboardInterrupt) to continue the sampling run.
            _loop must be able to handle KeyboardInterrupts gracefully, and should monitor
            the sampler's status periodically. Available status values are:
            - 'ready': Ready to sample.
            - 'paused': A pause has been requested, or the sampler is paused. _loop should return control
                as soon as it is safe to do so.
            - 'halt': A halt has been requested, or the sampler is stopped. _loop should call halt as soon
                as it is safe to do so.
            - 'running': Sampling is in progress.

    :SeeAlso: Model, Sampler, StepMethod.
    """
    def __init__(self, input=None, db='ram', name='MCMC', calc_deviance=True, **kwds):
        """Initialize an MCMC instance.

        :Parameters:
          - input : module, list, tuple, dictionary, set, object or nothing.
              Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
              If nothing, all nodes are collected from the base namespace.
          - db : string
              The name of the database backend that will store the values
              of the stochastics and deterministics sampled during the MCMC loop.
          - verbose : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
          - **kwds :
              Keywords arguments to be passed to the database instantiation method.
        """
        Sampler.__init__(self, input, db, name, calc_deviance=calc_deviance, **kwds)

        self.step_method_dict = {}
        for s in self.stochastics:
            self.step_method_dict[s] = []

        self._state = ['status', '_current_iter', '_iter', '_tune_interval', '_burn', '_thin']

    def use_step_method(self, step_method_class, *args, **kwds):
        """
        M.use_step_method(step_method_class, *args, **kwds)

        Example of usage: To handle stochastic A with a Metropolis instance,

            M.use_step_method(Metropolis, A, sig=.1)

        To subsequently get a reference to the new step method,

            S = M.step_method_dict[A][0]
        """

        new_method = step_method_class(*args, **kwds)
        if self.verbose > 1:
            print 'Using step method %s. Stochastics: ' % step_method_class.__name__
        for s in new_method.stochastics:
            self.step_method_dict[s].append(new_method)
            if self.verbose > 1:
                print '\t'+s.__name__

        setattr(new_method, '_model', self)

    def assign_step_methods(self):
        """
        Make sure every stochastic variable has a step method. If not,
        assign a step method from the registry.
        """

        # Assign dataless stepper first
        last_gen = set([])
        for s in self.stochastics:
            if len(s.extended_children)==0:
                last_gen.add(s)                
        
        dataless, dataless_gens = crawl_dataless(set(last_gen), [last_gen])
        if len(dataless) > 0:
            new_method = DrawFromPrior(dataless, dataless_gens[::-1])
            setattr(new_method, '_model', self)        
            for d in dataless:
                self.step_method_dict[d].append(new_method)
                if self.verbose > 1:
                    print 'Assigning step method %s to stochastic %s' % (new_method.__class__.__name__, d.__name__)

        for s in self.stochastics:
            # If not handled by any step method, make it a new step method using the registry
            if len(self.step_method_dict[s])==0:
                new_method = assign_method(s)
                setattr(new_method, '_model', self)
                self.step_method_dict[s].append(new_method)
                if self.verbose > 1:
                    print 'Assigning step method %s to stochastic %s' % (new_method.__class__.__name__, s.__name__)

        self.step_methods = set()
        for s in self.stochastics:
            self.step_methods |= set(self.step_method_dict[s])

        self.restore_sm_state()

    def sample(self, iter, burn=0, thin=1, tune_interval=1000, save_interval=None, verbose=0):
        """
        sample(iter, burn, thin, tune_interval, save_interval, verbose)

        Initialize traces, run sampling loop, clean up afterward. Calls _loop.
        """
        
        self.assign_step_methods()
        
        if burn >= iter:
            raise ValueError, 'Burn interval must be smaller than specified number of iterations.'
        self._iter = int(iter)
        self._burn = int(burn)
        self._thin = int(thin)
        self._tune_interval = int(tune_interval)
        self._save_interval = save_interval

        length = int(np.ceil((1.0*iter-burn)/thin))
        self.max_trace_length = length

        # Flags for tuning
        self._tuning = True
        self._tuned_count = 0
        
        Sampler.sample(self, iter, length, verbose)

    def _loop(self):
        # Set status flag
        self.status='running'

        # Record start time
        start = time.time()

        try:
            while self._current_iter < self._iter and not self.status == 'halt':
                if self.status == 'paused':
                    break

                i = self._current_iter

                if i == self._burn and self.verbose>0:
                    print 'Burn-in interval complete'

                # Tune at interval
                if i and not (i % self._tune_interval) and self._tuning:
                    self.tune()

                # Tell all the step methods to take a step
                for step_method in self.step_methods:
                    if self.verbose > 2:
                        print 'Step method %s stepping.' % step_method._id
                    # Step the step method
                    step_method.step()

                # Calculate deviance
                self.deviance._value.force_compute()

                if i % self._thin == 0 and i >= self._burn:
                    self.tally()

                if self._save_interval is not None:
                    if i % self._save_interval==0:
                        self.save_state()

                if not i % 10000 and i and self.verbose > 0:
                    per_step = (time.time() - start)/i
                    remaining = self._iter - i
                    time_left = remaining * per_step

                    print "Iteration %i of %i (%i:%02d:%02d remaining)" % (i, self._iter, time_left/3600, (time_left%3600)/60, (time_left%60))

                if not i % 1000:
                    self.commit()

                self._current_iter += 1

        except KeyboardInterrupt:
            self.status='halt'

        if self.status == 'halt':
            self._halt()

    def tune(self):
        """
        Tell all step methods to tune themselves.
        """

        # =======================================
        # = This is what makes price.py puke... =
        # =======================================
        # Only tune during burn-in
        # if self._current_iter > self._burn:
        #     self._tuning = False
        #     return

        if self.verbose > 0:
            print '\tTuning at iteration', self._current_iter

        # Initialize counter for number of tuning stochastics
        tuning_count = 0

        for step_method in self.step_methods:
            # Tune step methods
            tuning_count += step_method.tune(verbose=self.verbose)
            if self.verbose > 1:
                print '\t\tTuning step method %s, returned %i\n' %(step_method._id, tuning_count)
                sys.stdout.flush()

        if not tuning_count:
            # If no step methods needed tuning, increment count
            self._tuned_count += 1
        else:
            # Otherwise re-initialize count
            self._tuned_count = 0

        # 5 consecutive clean intervals removed tuning
        if self._tuned_count == 5:
            if self.verbose > 0: print 'Finished tuning'
            self._tuning = False


    def get_state(self):
        """
        Return the sampler and step methods current state in order to
        restart sampling at a later time.
        """

        self.step_methods = set()
        for s in self.stochastics:
            self.step_methods |= set(self.step_method_dict[s])

        state = Sampler.get_state(self)
        state['step_methods'] = {}

        # The state of each StepMethod.
        for sm in self.step_methods:
            state['step_methods'][sm._id] = sm.current_state().copy()

        return state

    def restore_sm_state(self):

        sm_state = self.db.getstate()

        if sm_state is not None:
            sm_state = sm_state.get('step_methods', {})

            # Restore stepping methods state
            for sm in self.step_methods:
                sm.__dict__.update(sm_state.get(sm._id, {}))

    def dic(self):
        """Calculates deviance information Criterion"""

        # Find mean deviance
        mean_deviance = np.mean(self.deviance.trace(), axis=0)

        # Set values of all parameters to their mean
        for stochastic in self.stochastics:

            # Calculate mean of paramter
            mean_value = np.mean(stochastic.trace(), axis=0)

            # Set current value to mean
            stochastic.value = mean_value

        # Calculate deviance at the means
        try:
            self.deviance._value.force_compute()
            deviance_at_mean = self.deviance.value
        except ZeroProbability:
            deviance_at_mean = -np.inf
        # Return twice deviance minus deviance at means
        return 2*mean_deviance - deviance_at_mean




