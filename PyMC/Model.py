# Changeset history
# 22/03/2007 -DH- Added methods to query the StepMethod's state and pass it to database.
# 20/03/2007 -DH- Separated Model from Sampler. Removed _prepare(). Commented __setattr__ because it breaks properties.

__docformat__='reStructuredText'

""" Summary"""

from numpy import zeros, floor
from StepMethods import StepMethod, assign_method
from Matplot import Plotter, show
import database
from PyMCObjects import Stochastic, Deterministic, Node, Variable, Potential
from Container import ContainerBase, Container
# from utils import extend_children, extend_parents
import gc, sys,os
from copy import copy
from threading import Thread
from thread import interrupt_main
from time import sleep
from Node import ContainerBase
from Container import ObjectContainer

GuiInterrupt = 'Computation halt'
Paused = 'Computation paused'

class Model(ObjectContainer):
    """
    Model is initialized with:

      >>> A = Model(input, db='ram', output_path=None, verbose=0)

      :Parameters:
        - input : module, list, tuple, dictionary, set, object or nothing.
            Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
            If nothing, all nodes are collected from the base namespace.
        - db : string
            The name of the database backend that will store the values
            of the stochs and dtrms sampled during the MCMC loop.
        - output_path : string
            The place where any output files should be put.
        - verbose : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    Attributes:
      - dtrms
      - stochs (with isdata=False)
      - data (stochastic variables with isdata=True)
      - variables
      - potentials
      - containers
      - nodes
      - all_objects
      - status: Not useful for the Model base class, but may be used by subclasses.
      
    The following attributes only exist after the appropriate method is called:
      - moral_neighbors: The edges of the moralized graph. A dictionary, keyed by stochastic variable,
        whose values are sets of stochastic variables. Edges exist between the key variable and all variables
        in the value. Created by method _moralize.
      - extended_children: The extended children of self's stochastic variables. See the docstring of
        extend_children. This is a dictionary keyed by stochastic variable.
      - generations: A list of sets of stochastic variables. The members of each element only have parents in 
        previous elements. Created by method find_generations.

    Methods:
       - tally(index): Write all variables' current values to trace, at location index.
       - sample_model_likelihood(iter): Generate and return iter samples of p(data and potentials|model).
         Can be used to generate Bayes' factors.
       - save_traces(): Pickle and save traces to disk. XXX Do we still need this method?
       - seed() XXX I don't know what this does...
       - plot(): Visualize traces for all variables
       - remember(trace_index) : Return the entire model to the tallied state indexed by trace_index.

    :SeeAlso: Sampler, MAP, NormalApproximation, weight, Container, graph.
    """
    def __init__(self, input=None, db='ram', output_path=None, verbose=0, **kwds):
        """Initialize a Model instance.

        :Parameters:
          - input : module, list, tuple, dictionary, set, object or nothing.
              Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
              If nothing, all nodes are collected from the base namespace.
          - db : string
              The name of the database backend that will store the values
              of the stochs and dtrms sampled during the MCMC loop.
          - output_path : string
              The place where any output files should be put.
          - verbose : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
          - **kwds : 
              Keywords arguments to be passed to the database instantiation method.
        """

        self.generations = []
        self.verbose = verbose
        self._db_args = kwds
        # Flag for model state
        self.status = 'ready'

        # Get stochs, dtrms, etc.
        if input is None:
            import __main__
            input = __main__
        
        ObjectContainer.__init__(self, input)

        # Specify database backend
        self._assign_database_backend(db)
        
        # Instantiate plotter 
        # Hardcoding the matplotlib backend raises error in
        # interactive use. DH
        try:
            self._plotter = Plotter(plotpath=output_path or self.__name__ + '_output/')
        except:
            self._plotter = 'Could not be instantiated.'
        
        for stoch_attr in ['extended_children', 'extended_parents', 'markov_blanket', 'moral_neighbors']:
            setattr(self, stoch_attr, {})
            for stoch in self.stochs | self.data:
                getattr(self, stoch_attr)[stoch] = getattr(stoch, stoch_attr)
        
        self.find_generations
                        
    def find_generations(self):
        """
        Parse up the generations for model averaging. A generation is the
        set of stochastic variables that only has parents in previous generations.
        """
        self.generations = []
        # self.extend_children()

        # Find root generation
        self.generations.append(set())
        all_children = set()
        for stoch in self.stochs:
            all_children.update(self.extended_children[stoch] & self.stochs)
        self.generations[0] = self.stochs - all_children

        # Find subsequent _generations
        children_remaining = True
        gen_num = 0
        while children_remaining:
            gen_num += 1


            # Find children of last generation
            self.generations.append(set())
            for stoch in self.generations[gen_num-1]:
                self.generations[gen_num].update(self.extended_children[stoch] & self.stochs)


            # Take away stochs that have parents in the current generation.
            thisgen_children = set()
            for stoch in self.generations[gen_num]:
                thisgen_children.update(self.extended_children[stoch] & self.stochs)
            self.generations[gen_num] -= thisgen_children


            # Stop when no subsequent _generations remain
            if len(thisgen_children) == 0:
                children_remaining = False

    def sample_likelihood(self, iter):
        """
        Returns iter samples of (log p(data|self.stochs, self) | self).
        
        Exponentiating and averaging gives an estimate of the model likelihood,
        p(data|self).
        """
        loglikes = zeros(iter)

        try:
            for i in xrange(iter):
                if i % 10000 == 0:
                    print 'Sample ', i, ' of ', iter

                for generation in self.generations:
                    for stoch in generation:
                        stoch.random()

                for datum in self.data | self.potentials:
                    loglikes[i] += datum.logp

        except KeyboardInterrupt:
            print 'Sample ', i, ' of ', iter
            raise KeyboardInterrupt

        return loglikes

    def status():
        doc = \
        """Status of model. May be one of running, paused, halt or ready.
          - `running` : The model is currently sampling.
          - `paused` : The model has been interrupted during sampling. It is
            ready to be restarted by `continuesample`.
          - `halt` : The model has been interrupted. It cannot be restarted.
            If sample is called again, a new chain will be initiated.
          - `ready` : The model is ready to sample.
        """
        def fget(self):
            return self.__status
        def fset(self, value):
            if value in ['running', 'paused', 'halt', 'ready']:
                self.__status=value
            else:
                raise AttributeError, value
        return locals()
    status = property(**status())

    def seed(self):
        """
        Seed new initial values for the stochs.
        """
        for stochs in self.stochs:
            try:
                if stochs.rseed is not None:
                    value = stochs.random(**stochs.parent_values)
            except:
                pass
    
    def tally(self, index):
        self.db.tally(index)
    
    def _assign_database_backend(self, db):
        """Assign Trace instance to stochs and dtrms and Database instance
        to self.

        :Parameters:
          - `db` : string, Database instance
            The name of the database module (see below), or a Database instance.

        Available databases:
          - `no_trace` : Traces are not stored at all.
          - `ram` : Traces stored in memory.
          - `txt` : Traces stored in memory and saved in txt files at end of
                sampling.
          - `sqlite` : Traces stored in sqlite database.
          - `mysql` : Traces stored in a mysql database.
          - `hdf5` : Traces stored in an HDF5 file.
        """
        # Objects that are not to be tallied are assigned a no_trace.Trace
        # Tallyable objects are listed in the _nodes_to_tally set. 

        no_trace = getattr(database, 'no_trace')
        self._variables_to_tally = set()
        for object in self.stochs | self.dtrms :
            if object.trace:
                self._variables_to_tally.add(object)
            else:
                object.trace = no_trace.Trace()

        # If not already done, load the trace backend from the database 
        # module, and assign a database instance to Model.
        if type(db) is str:
            module = getattr(database, db)
            self.db = module.Database(**self._db_args)
        elif isinstance(db, database.base.Database):
            self.db = db
            self.restore_state()
        else:
            module = db.__module__
            self.db = db
        
        # Assign Trace instances to tallyable objects. 
        self.db.connect(self)
        
    def plot(self):
        """
        Plots traces and histograms for variables.
        """

        # Loop over nodes
        for variable in self._variables_to_tally:            
            # Plot object
            self._plotter.plot(variable)

        # show()
    
    


class Sampler(Model):
    # TODO: Docstring!!
    def __init__(self, input=None, db='ram', output_path=None, verbose=0, **kwds):
        
        # Instantiate superclass
        Model.__init__(self, input, db, output_path, verbose, **kwds)
        
        # Default StepMethod
        self._assign_samplingmethod()

        self._state = ['status', '_current_iter', '_iter', '_tune_interval', '_burn', '_thin']
        
    def _assign_samplingmethod(self):
        """
        Make sure every stochastic variable has a step method. If not, 
        assign a step method from the registry.
        """

        for stoch in self.stochs:

            # Is it a member of any StepMethod?
            homeless = True
            for step_method in self.step_methods:
                if stoch in step_method.stochs:
                    homeless = False
                    break

            # If not, make it a new StepMethod using the registry
            if homeless:
                new_method = assign_method(stoch)
                setattr(new_method, '_model', self)
                self.step_methods.add(new_method)
    
        
    def sample(self, iter, burn=0, thin=1, tune_interval=1000, verbose=0):
        """
        sample(iter, burn, thin, tune_interval)

        Initialize traces, run MCMC loop.
        """
        
        self._iter = int(iter)
        self._burn = int(burn)
        self._thin = int(thin)
        self._tune_interval = int(tune_interval)
        self._cur_trace_index = 0
        length = (iter-burn)/thin
        self.max_trace_length = length
        
        # Flag for verbose output
        self.verbose = verbose
        
        # Flags for tuning
        self._tuning = True
        self._tuned_count = 0

        self.seed()

        # Initialize database -> initialize traces.
        self.db._initialize(length)
        
        # Loop
        self._current_iter = 0
        self._loop()
        
    def _loop(self):
        # Set status flag
        self.status='running'

        try:
            while self._current_iter < self._iter and not self.status == 'halt':
                if self.status == 'paused':
                    self.pause_sampling()
                    break

                i = self._current_iter

                if i == self._burn and self.verbose>0: 
                    print 'Burn-in interval complete'

                # Tune at interval
                if i and not (i % self._tune_interval) and self._tuning:
                    self.tune()

                # Tell all the step methods to take a step
                for step_method in self.step_methods:

                    # Step the step method
                    step_method.step()

                if not i % self._thin and i >= self._burn:
                    self.tally()

                if not i % 10000 and self.verbose > 0:
                    print 'Iteration ', i, ' of ', self._iter

                self._current_iter += 1

        except KeyboardInterrupt:
            self.status='paused'
            self.pause_sampling()
            
        if self.status == 'halt':
            self.halt_sampling()
            
        # Finalize
        print 'Sampling finished normally.'
        self.status = 'ready'
        self.save_state()
        self.db._finalize()

        # # TODO: This should interrupt the main thread immediately, but it waits until 
        # # TODO: return is pressed before doing its thing. Bug report filed at python.org.
        # # TODO: Doesn't seem fixable without a funky patch...
        try:
            interrupt_main()
        except KeyboardInterrupt:
            pass
            
    def pause_sampling(self):
        print 'Pausing at iteration ', self._current_iter, ' of ', self._iter
        return None

    def halt_sampling(self):
        print 'Halting at iteration ', self._current_iter, ' of ', self._iter
        for variable in self._variables_to_tally:
            variable.trace.truncate(self._cur_trace_index)
           
    def continue_sampling(self):
        self._loop()
    
    #
    # Tally
    #
    def tally(self):
       """
       tally()

       Records the value of all tracing variables.
       """
       if self.verbose > 2:
           print self.__name__ + ' tallying.'
       if self._cur_trace_index < self.max_trace_length:
           self.db.tally(self._cur_trace_index)

       self._cur_trace_index += 1
       if self.verbose > 2:
           print self.__name__ + ' done tallying.'
        
    def tune(self):
        """
        Tell all step methods to tune themselves.
        """

        # Only tune during burn-in
        if self._current_iter > self._burn:
            self._tuning = False
            return

        if self.verbose > 1:
            print '\tTuning at iteration', self._current_iter

        # Initialize counter for number of tuning stochs
        tuning_count = 0

        for step_method in self.step_methods:
            # Tune step methods
            tuning_count += step_method.tune(verbose=self.verbose)

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


    def interactive_sample(self, *args, **kwds):
        """
        Samples in interactive mode. Main thread of control stays in this function.
        """
        self._sampling_thread = Thread(target=self.sample, args=args, kwargs=kwds)
        self._sampling_thread.start()
        self.interactive_prompt()

    def interactive_continue(self):
        """
        Restarts thread in interactive mode
        """
        self._sampling_thread = Thread(target=self._loop)
        self._sampling_thread.start()
        self.interactive_prompt()
        
    def interactive_prompt(self):
        """
        Drive the sampler from the prompt.

        Commands:
          i -- print current iteration index
          p -- pause
          q -- quit
        """
        print self.interactive_prompt.__doc__, '\n'
        try:
            while self.status in ['running', 'paused']:
                    # sys.stdout.write('PyMC> ')
                    cmd = raw_input('PyMC> ')
                    if cmd == 'i':
                        print 'Current iteration: ', self._current_iter
                    elif cmd == 'p':
                        self.status = 'paused'
                        break
                    elif cmd == 'q':
                        self.status = 'halt'
                        break
                    else:
                        print 'Unknown command'
                        print self.interactive_prompt.__doc__
        except KeyboardInterrupt:
            if not self.status == 'ready':
                self.status = 'pause'      
                
        if not self.status == 'ready':        
            print 'Waiting for current iteration to finish...'
            try:
                while self._sampling_thread.isAlive():
                    sleep(.1)
            except:
                pass

            print 'Exiting interactive prompt...'
            if self.status == 'paused':
                print 'Call interactive_continue method to continue, or call halt_step method to truncate traces and stop.'

    # Should get_state, save_state, restore_state and remember be moved up to Model?
    def get_state(self):
        """
        Return the sampler and step methods current state in order to
        restart sampling at a later time.
        """
        state = dict(sampler={}, step_methods={}, stochs={})
        # The state of the sampler itself.
        for s in self._state:
            state['sampler'][s] = getattr(self, s)

        # The state of each StepMethod.
        for sm in self.step_methods:
            state['step_methods'][sm._id] = sm.current_state().copy()

        # The state of each stochastic parameter
        for stoch in self.stochs:
            state['stochs'][stoch.__name__] = stoch.value
        return state

    def save_state(self):
        """
        Tell the database to save the current state of the sampler.
        """
        self.db.savestate(self.get_state())

    def restore_state(self):
        """
        Restore the state of the sampler and of the step methods to
        the state stored in the database.
        """
        state = self.db.getstate()
        
        # Restore sampler's state
        sampler_state = state.get('sampler', {})
        self.__dict__.update(sampler_state)
        
        # Restore stepping methods state
        sm_state = state.get('step_methods', {})
        for sm in self.step_methods:
            sm.__dict__.update(sm_state.get(sm._id, {}))
            
        # Restore stochastic parameters state
        stoch_state = state.get('stochs', {})
        for sm in self.stochs:
            try:
                self.stoch.value = stoch_state[sm.__name__]
            except:
                pass
            
    def remember(self, trace_index = None):
        """
        remember(trace_index = randint(trace length to date))

        Sets the value of all tracing variables to a value recorded in
        their traces.
        """
        if trace_index is None:
            trace_index = randint(self.cur_trace_index)

        for variable in self._variables_to_tally:
            variable.value = variable.trace()[trace_index]
                        
    def goodness(self, iterations, loss='squared', plot=True, color='b', filename='gof'):
        """
        Calculates Goodness-Of-Fit according to Brooks et al. 1998
        
        :Arguments:
            - iterations : integer
                Number of samples to draw from the model for GOF evaluation.
            - loss (optional) : string
                Loss function; valid entries include 'squared', 'absolute' and
                'chi-square'.
            - plot (optional) : booleam
                Flag for printing GOF plots.
            - color (optional) : string
                Color of plot; see matplotlib docs for valid entries (usually
                the first letter of the desired color).
            - filename (optional) : string
                File name for output statistics.
        """
        
        if self.verbose > 0:
            print
            print "Goodness-of-fit"
            print '='*50
            print 'Generating %s goodness-of-fit simulations' % iterations
        
        
        # Specify loss function
        if loss=='squared':
            self.loss = squared_loss
        elif loss=='absolute':
            self.loss = absolute_loss
        elif loss=='chi-square':
            self.loss = chi_square_loss
        else:
            print 'Invalid loss function specified.'
            return
            
        # Open file for GOF output
        outfile = open(filename + '.csv', 'w')
        outfile.write('Goodness of Fit based on %s iterations\n' % iterations)
        
        # Empty list of GOF plot points
        D_points = []
        
        # Set GOF flag
        self._gof = True
        
        # List of names for conditional likelihoods
        self._like_names = []
        
        # Local variable for the same
        like_names = None
        
        # Generate specified number of points
        for i in range(iterations):
            
            valid_gof_points = False
            
            # Sometimes the likelihood bombs out and doesnt produce
            # GOF points
            while not valid_gof_points:
                
                # Initializealize list of GOF error loss values
                self._gof_loss = []
                
                # Loop over stochs
                for name in self.stochs:
                    
                    # Look up stoch
                    stoch = self.stochs[name]
                    
                    # Retrieve copy of trace
                    trace = stoch.get_trace(burn=burn, thin=thin, chain=chain, composite=composite)
                    
                    # Sample value from trace
                    sample = trace[random_integers(len(trace)) - 1]
                    
                    # Set current value to sampled value
                    stoch.set_value(sample)
                
                # Run calculate likelihood with sampled stochs
                try:
                    self()
                except (LikelihoodError, OverflowError, ZeroDivisionError):
                    # Posterior dies for some reason
                    pass
                
                try:
                    like_names = self._like_names
                    del(self._like_names)
                except AttributeError:
                    pass
                
                # Conform that length of GOF points is valid
                if len(self._gof_loss) == len(like_names):
                    valid_gof_points = True
            
            # Append points to list
            D_points.append(self._gof_loss)
        
        # Transpose and plot GOF points
        
        D_points = t([[y for y in x if shape(y)] for x in D_points], (1,2,0))
        
        # Keep track of number of simulation deviances that are
        # larger than the corresponding observed deviance
        sim_greater_obs = 0
        n = 0
        
        # Dictionary to hold GOF statistics
        stats = {}
        
        # Dictionary to hold GOF plot data
        plots = {}
        
        # Loop over the sets of points for plotting
        for name,points in zip(like_names,D_points):
            
            if plots.has_key(name):
                # Append points, if already exists
                plots[name] = concatenate((plots[name], points), 1)
            
            else:
                plots[name] = points
            
            count = sum(s>o for o,s in t(points))
            
            try:
                stats[name] += array([count,iterations])
            except KeyError:
                stats[name] = array([1.*count,iterations])
            
            sim_greater_obs += count
            n += iterations
        
        # Generate plots
        if plot:
            for name in plots:
                self.plotter.gof_plot(plots[name], name, color=color)
        
        # Report p(D(sim)>D(obs))
        for name in stats:
            num,denom = stats[name]
            print 'p( D(sim) > D(obs) ) for %s = %s' % (name,num/denom)
            outfile.write('%s,%f\n' % (name,num/denom))
        
        p = 1.*sim_greater_obs/n
        print 'Overall p( D(sim) > D(obs) ) =', p
        print
        outfile.write('overall,%f\n' % p)
        
        stats['overall'] = array([1.*sim_greater_obs,n])
        
        # Unset flag
        self._gof = False
        
        # Close output file
        outfile.close()
        
        return stats
