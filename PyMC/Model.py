# Changeset history
# 22/03/2007 -DH- Added methods to query the StepMethod's state and pass it to database.
# 20/03/2007 -DH- Separated Model from Sampler. Removed _prepare(). Commented __setattr__ because it breaks properties.

__docformat__='reStructuredText'

""" Summary"""

from numpy import zeros, floor
from Matplot import Plotter, show
import database
from PyMCObjects import Stochastic, Deterministic, Node, Variable, Potential
from Container import Container, ObjectContainer
import sys,os
from copy import copy
from threading import Thread
from Node import ContainerBase

GuiInterrupt = 'Computation halt'
Paused = 'Computation paused'

class Model(ObjectContainer):
    """
    The base class for all objects that fit probability models. Model is initialized with:

      >>> A = Model(input, output_path=None, verbose=0)

      :Parameters:
        - input : module, list, tuple, dictionary, set, object or nothing.
            Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
            If nothing, all nodes are collected from the base namespace.
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
       - sample_model_likelihood(iter): Generate and return iter samples of p(data and potentials|model).
         Can be used to generate Bayes' factors.

    :SeeAlso: Sampler, MAP, NormalApproximation, weight, Container, graph.
    """
    def __init__(self, input=None, output_path=None, verbose=0):
        """Initialize a Model instance.

        :Parameters:
          - input : module, list, tuple, dictionary, set, object or nothing.
              Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
              If nothing, all nodes are collected from the base namespace.
          - output_path : string
              The place where any output files should be put.
          - verbose : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
        """

        self.generations = []
        self.verbose = verbose
        self.output_path = output_path

        # Get stochs, dtrms, etc.
        if input is None:
            import __main__
            __main__.__dict__.update(self.__class__.__dict__)
            input = __main__
        
        ObjectContainer.__init__(self, input)
        
        for stoch_attr in ['extended_children', 'extended_parents', 'markov_blanket', 'moral_neighbors']:
            setattr(self, stoch_attr, {})
            for stoch in self.stochs | self.data:
                getattr(self, stoch_attr)[stoch] = getattr(stoch, stoch_attr)
        
        if not hasattr(self, 'generations'):
            self.find_generations()
                        
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

    def draw_from_prior(self):
        """
        Sets all variables to random values drawn from joint 'prior', meaning contributions 
        of data and potentials to the joint distribution are not considered.
        """
        
        for generation in self.generations:
            for stoch in generation:
                stoch.random()

    def seed(self):
        """
        Seed new initial values for the stochs.
        """
        for generation in self.generations:
            for stoch in generation:
                try:
                    if stoch.rseed is not None:
                        value = stoch.random(**stoch.parents.value)
                except:
                    pass
    
    


class Sampler(Model):
    """
    The base class for all objects that fit probability models using Monte Carlo methods. 
    Sampler is initialized with:

      >>> A = Sampler(input, db, output_path=None, verbose=0)

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

    Inherits all methods and attributes from Model. Subclasses must either define the _loop method:

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
            
    Or define a draw() method, which draws a single sample from the posterior. Subclasses may also want
    to override the default sample() method.
    
    :SeeAlso: Model, MCMC.
    """
    def __init__(self, input=None, db='ram', output_path=None, verbose=0, reinit_model=True, **kwds):
        """Initialize a Sampler instance.

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
        
        
        # Instantiate superclass
        if reinit_model:
            Model.__init__(self, input, output_path, verbose)
        
        # Specify database backend and save its keywords
        self._db_args = kwds        
        self._assign_database_backend(db)
        
        # Flag for model state
        self.status = 'ready'
        
        # Instantiate plotter 
        # Hardcoding the matplotlib backend raises error in
        # interactive use. DH
        try:
            self._plotter = Plotter(plotpath=output_path or self.__name__ + '_output/')
        except:
            self._plotter = 'Could not be instantiated.'        
        
        self._state = ['status', '_current_iter', '_iter']
    
    def sample(self, iter, length=None, verbose=0):
        """
        Draws iter samples from the posterior.
        """
        self._cur_trace_index=0
        self.max_trace_length = iter
        self._iter = iter
        
        # Flag for verbose output
        self.verbose = verbose        
        
        self.seed()        
        
        # Initialize database -> initialize traces.
        if length is None:
            length = iter
        self.db._initialize(length)

        # Loop
        self._current_iter = 0
        self._loop()

        # Finalize
        if self.status in ['running', 'halt']:
            print 'Sampling finished normally.'
            self.status = 'ready'
            
        self.save_state()
        self.db._finalize()        

    def _loop(self):
        """
        _loop(self, *args, **kwargs)

        Can be called after a sampling run is interrupted (by pausing, halting or a 
        KeyboardInterrupt) to continue the sampling run.

        _loop must be able to handle KeyboardInterrupts gracefully, and should monitor
        the sampler's status periodically. Available status values are:
        - 'ready': Ready to sample.
        - 'paused': A pause has been requested, or the sampler is paused. _loop should return control
            as soon as it is safe to do so.
        - 'halt': A halt has been requested, or the sampler is stopped. _loop should call halt as soon
            as it is safe to do so.
        - 'running': Sampling is in progress.
        """
        self.status='running'

        try:
            while self._current_iter < self._iter and not self.status == 'halt':
                if self.status == 'paused':
                    break

                i = self._current_iter

                self.draw()                
                self.tally()

                if not i % 10000 and self.verbose > 0:
                    print 'Iteration ', i, ' of ', self._iter

                self._current_iter += 1

        except KeyboardInterrupt:
            self.status='halt'

        if self.status == 'halt':
            self.halt()
    
    def draw(self):
        """
        Either draw() or _loop() must be overridden in subclasses of Sampler.
        """
        pass
    
    def plot(self):
        """
        Plots traces and histograms for variables.
        """

        # Loop over nodes
        for variable in self._variables_to_tally:            
            # Plot object
            self._plotter.plot(variable)

        # show()

    def status():
        doc = \
        """Status of sampler. May be one of running, paused, halt or ready.
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

    def halt(self):
        print 'Halting at iteration ', self._current_iter, ' of ', self._iter
        for variable in self._variables_to_tally:
            variable.trace.truncate(self._cur_trace_index)
           
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


    def isample(self, *args, **kwds):
        """
        Samples in interactive mode. Main thread of control stays in this function.
        """
        self._sampling_thread = Thread(target=self.sample, args=args, kwargs=kwds)
        self.status = 'running'
        self._sampling_thread.start()
        self.iprompt()

    def icontinue(self):
        """
        Restarts thread in interactive mode
        """
        self._sampling_thread = Thread(target=self._loop)
        self.status = 'running'        
        self._sampling_thread.start()
        self.iprompt()
        
    def iprompt(self):
        """
        Drive the sampler from the prompt.

        Commands:
          i -- print current iteration index
          p -- pause
          q -- quit
        """
        print self.iprompt.__doc__, '\n'
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
                        print self.iprompt.__doc__
        except KeyboardInterrupt:
            if not self.status == 'ready':
                self.status = 'halt'      
                
        if not self.status == 'ready':        
            print 'Waiting for current iteration to finish...'
            try:
                while self._sampling_thread.isAlive():
                    sleep(.1)
            except:
                pass

            print 'Exiting interactive prompt...'
            if self.status == 'paused':
                print 'Call icontinue method to continue, or call halt_step method to truncate traces and stop.'

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
            
        # Restore stochastic parameters state
        stoch_state = state.get('stochs', {})
        for sm in self.stochs:
            try:
                self.stoch.value = stoch_state[sm.__name__]
            except:
                pass
        
        return state
            
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
