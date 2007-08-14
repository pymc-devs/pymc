# Changeset history
# 22/03/2007 -DH- Added methods to query the SamplingMethod's state and pass it to database.
# 20/03/2007 -DH- Separated Model from Sampler. Removed _prepare(). Commented __setattr__ because it breaks properties.

__docformat__='reStructuredText'

""" Summary"""

from numpy import zeros, floor
from SamplingMethods import SamplingMethod, assign_method
from Matplot import Plotter, show
import database
from PyMCObjects import Parameter, Node, PyMCBase, Variable, Potential
from Container import Container
from utils import extend_children
import gc, sys,os
from copy import copy
from threading import Thread

GuiInterrupt = 'Computation halted'
Paused = 'Computation paused'

class Model(object):
    """
    Model manages MCMC loops. It is initialized with:

      >>> A = Model(prob_def, dbase=None)

    :Arguments:
        prob_def : class, module, dictionary
          Contains PyMC objects and SamplingMethods
        dbase : module name
          Database backend used to tally the samples.
          Implemented backends: None, hdf5, txt.

    Externally-accessible attributes:
      - nodes : All extant Nodes.
      - parameters : All extant Parameters with isdata = False.
      - data : All extant Parameters with isdata = True.
      - pymc_objects : All extant Parameters and Nodes.
      - sampling_methods : All extant SamplingMethods.

    Externally-accessible methods:
       - sample(iter) : At each MCMC iteration, calls each sampling_method's step() method.
         Tallies Parameters and Nodes as appropriate.
       - trace(parameter, burn, thin, slice) : Return the trace of parameter, 
         sliced according to slice or burn and thin arguments.
       - remember(trace_index) : Return the entire model to the tallied state indexed by trace_index.
       - DAG : Draw the model as a directed acyclic graph.

    :Note:
        All the plotting functions can probably go on the base namespace and take Parameters as
        arguments.

    :SeeAlso: Sampler, PyMCBase, Parameter, Node, and weight.
    """
    def __init__(self, input, db='ram', verbose=0):
        """Initialize a Model instance.

        :Parameters:
          - input : module
              A module containing the model definition, in terms of Parameters, 
              and Nodes.
          - dbase : string
              The name of the database backend that will store the values
              of the parameters and nodes sampled during the MCMC loop.
          - verbose : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
        """

        # Instantiate public attributes
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.potentials = set()
        self.containers = set()
        self.variables = set()
        self.extended_children = None
        self.verbose = verbose
        # Instantiate hidden attributes
        self._generations = []
        self.__name__ = None
        
        # Flag for model state
        self.status = 'ready'

        # Check for input name
        if hasattr(input, '__name__'):
            _filename = os.path.split(input.__file__)[-1]
            self.__name__ = os.path.splitext(_filename)[0]
        else:
            try:
                self.__name__ = input['__name__']
            except: 
                self.__name__ = 'PyMC_Model'

        # Change input into a dictionary
        if isinstance(input, dict):
            self.input_dict = input.copy()
        else:
            try:
                # If input is a module, reload it to make a fresh copy.
                reload(input)
            except TypeError:
                pass

            self.input_dict = input.__dict__.copy()

        # Look for PyMC objects
        for name, item in self.input_dict.iteritems():
            if  isinstance(item, PyMCBase) \
                or isinstance(item, SamplingMethod) \
                or isinstance(item, Container):
                self.__dict__[name] = item
                
            # Allocate to appropriate set
            self._fileitem(item)
            
        # Union of PyMC objects
        self.variables = self.nodes | self.parameters | self.data
        self.pymc_objects = self.variables | self.potentials
        
    def _fileitem(self, item):
        """
        Store an item into the proper set:
          - parameters
          - nodes
          - data
          - containers
        """
        # If a dictionary is passed in, open it up.
        if isinstance(item, Container):
            self.containers.add(item)
            self.parameters.update(item.parameters)
            self.data.update(item.data)
            self.nodes.update(item.nodes)

        # File away the PyMC objects
        elif isinstance(item, PyMCBase):
            # Add an attribute to the object referencing the model instance.

            if isinstance(item, Node):
                self.nodes.add(item)

            elif isinstance(item, Parameter):
                if item.isdata:
                    self.data.add(item)
                else:  self.parameters.add(item)
                
            elif isinstance(item, Potential):
                self.potentials.add(item)

        elif isinstance(item, SamplingMethod):
            self.nodes.update(item.nodes)
            self.parameters.update(item.parameters)
            self.data.update(item.data)
        
    def _extend_children(self):
        """
        Makes a dictionary of self's PyMC objects' 'extended children.'
        """
        self.extended_children = {}
        dummy = Variable('', '', {}, 0, None)
        for variable in self.variables:
            dummy.children = copy(variable.children)
            extend_children(dummy)
            self.extended_children[variable] = dummy.children

    def _parse_generations(self):
        """
        Parse up the _generations for model averaging.
        """
        if not self.extended_children:
            self._extend_children()

        # Find root generation
        self._generations.append(set())
        all_children = set()
        for parameter in self.parameters:
            all_children.update(self.extended_children[parameter] & self.parameters)
        self._generations[0] = self.parameters - all_children

        # Find subsequent _generations
        children_remaining = True
        gen_num = 0
        while children_remaining:
            gen_num += 1


            # Find children of last generation
            self._generations.append(set())
            for parameter in self._generations[gen_num-1]:
                self._generations[gen_num].update(self.extended_children[parameter] & self.parameters)


            # Take away parameters that have parents in the current generation.
            thisgen_children = set()
            for parameter in self._generations[gen_num]:
                thisgen_children.update(self.extended_children[parameter] & self.parameters)
            self._generations[gen_num] -= thisgen_children


            # Stop when no subsequent _generations remain
            if len(thisgen_children) == 0:
                children_remaining = False

    def sample_model_likelihood(self, iter):
        """
        Returns iter samples of (log p(data|this_model_params, this_model) | data, this_model)
        """
        loglikes = zeros(iter)

        if len(self._generations) == 0:
            self._parse_generations()

        try:
            for i in xrange(iter):
                if i % 10000 == 0:
                    print 'Sample ', i, ' of ', iter

                for generation in self._generations:
                    for parameter in generation:
                        parameter.random()

                for datum in self.data | self.potentials:
                    loglikes[i] += datum.logp

        except KeyboardInterrupt:
            print 'Sample ', i, ' of ', iter
            raise KeyboardInterrupt

        return loglikes


    def DAG(self, format='raw', prog='dot', path=None, consts=True, legend=True):
        """
        DAG(format='raw', path=None, consts=True)

        Draw the directed acyclic graph for this model and writes it to path.
        If self.__name__ is defined and path is None, the output file is
        ./'name'.'format'.

        Format is a string. Options are:
        'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg', 
        'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp', 
        'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

        format='raw' outputs a GraphViz dot file.
        
        If consts is True, constant parents are included in the graph; 
        otherwise they're not.

        Returns the pydot 'dot' object for further user manipulation.
        """

        import pydot

        # self.dot_object = pydot.Dot()

        pydot_nodes = {}
        pydot_subgraphs = {}
        
        # Get ready to separate self's PyMC objects that are contained in containers.
        uncontained_params = self.parameters.copy()
        uncontained_data = self.data.copy()
        uncontained_nodes = self.nodes.copy()
        uncontained_potentials = self.potentials.copy()
        
        # Use this to make a graphviz cluster corresponding to each container, and to
        # draw the model outside of any container.
        def create_graph(subgraph):
            
            if not hasattr(subgraph, 'dot_object'):
                subgraph.dot_object = pydot.Cluster(graph_name = subgraph.__name__, label = subgraph.__name__)
                # subgraph.dot_object.graph_name = subgraph.dot_object.graph_name[8:]
            
            # Create the pydot nodes from pymc objects.
            
            # Data are filled ellipses
            for datum in subgraph.data:
                pydot_nodes[datum] = pydot.Node(name=datum.__name__, style='filled')
                subgraph.dot_object.add_node(pydot_nodes[datum])

            # Parameters are open ellipses
            for parameter in subgraph.parameters:
                pydot_nodes[parameter] = pydot.Node(name=parameter.__name__)
                subgraph.dot_object.add_node(pydot_nodes[parameter])

            # Nodes are downward-pointing triangles
            for node in subgraph.nodes:
                pydot_nodes[node] = pydot.Node(name=node.__name__, shape='invtriangle')
                subgraph.dot_object.add_node(pydot_nodes[node])
                
            # Potentials are octagons outlined three times
            for potential in subgraph.potentials:
                pydot_nodes[node] = pydot.Node(name=potential.__name__, shape='tripleoctagon')
                subgraph.dot_object.add_node(pydot_nodes[node])

            # Create nodes for constant parents if applicable and connect them.
            # Constants are filled boxes.
            if consts:
                for pymc_object in subgraph.pymc_objects:
                    for key in pymc_object.parents.iterkeys():
                        if not isinstance(pymc_object.parents[key], Variable) and not isinstance(pymc_object.parents[key], Container):
                            parent_name = pymc_object.parents[key].__str__()
                            subgraph.dot_object.add_node(pydot.Node(name = parent_name, shape = 'box', style='filled'))
                            new_edge = pydot.Edge(  src = parent_name, 
                                                    dst = pymc_object.__name__, 
                                                    label = key)
                            subgraph.dot_object.add_edge(new_edge)

        # Create a grahpviz cluster for each container.
        for container in self.containers:
            uncontained_params -= container.parameters
            uncontained_nodes -= container.nodes
            uncontained_data -= container.data
            uncontained_potentials -= container.potentials
            
            create_graph(container)
        
        # A dummy class to hold the uncontained PyMC objects    
        class uncontained(object):
            def __init__(self):
                self.dot_object = pydot.Dot()
                self.parameters = uncontained_params
                self.nodes = uncontained_nodes
                self.data = uncontained_data
                self.potentials = uncontained_potentials
                self.pymc_objects = self.parameters | self.nodes | self.data | self.potentials
        
        # Make nodes for the uncontained objects
        U = uncontained()
        create_graph(U)
        
        for container in self.containers:
            U.dot_object.add_subgraph(container.dot_object)
            
        self.dot_object = U.dot_object
        
                
        # Create edges from parent-child relationships between PyMC objects.
        for pymc_object in self.pymc_objects:
            for key in pymc_object.parents.iterkeys():
                
                # If a parent is a container, unpack it.
                # Draw edges between child and all elements of container (if consts=True)
                # or all variables in container (if consts = False).
                if isinstance(pymc_object.parents[key], Container):
                    for obj in pymc_object.parents[key].all_objects:
                        add_edge = True
                        if isinstance(obj, Variable):
                            parent_name = obj.__name__
                        elif consts:
                            parent_name = obj.__str__()
                            container.dot_object.add_node(pydot.Node(name = parent_name, shape = 'box', style='filled'))
                        else:
                            add_edge = False
                        
                        if add_edge:                            
                            
                            new_edge = pydot.Edge(  src = parent_name, 
                                                    dst = pymc_object.__name__, 
                                                    label = key)
                                                
                            self.dot_object.add_edge(new_edge)
                
                
                # If a parent is a variable, draw edge to it.
                # Note constant parents already are connected.
                elif isinstance(pymc_object.parents[key], Variable):
                    parent_name = pymc_object.parents[key].__name__
                    new_edge = pydot.Edge(  src = parent_name, 
                                            dst = pymc_object.__name__, 
                                            label = key)
        
                    self.dot_object.add_edge(new_edge)                            
        
        # Add legend if requested
        if legend:
            legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend')
            legend.add_node(pydot.Node(name='data', style='filled'))
            legend.add_node(pydot.Node(name='parameters'))
            legend.add_node(pydot.Node(name='nodes', shape='invtriangle'))
            legend.add_node(pydot.Node(name='potentials', shape='tripleoctagon'))
            if consts:
                legend.add_node(pydot.Node(name='constants', style='filled', shape='box'))
            self.dot_object.add_subgraph(legend)

        # Draw the graph
        if not path == None:
            self.dot_object.write(path=path, format=format, prog=prog)
        else:
            ext=format
            if format=='raw':
                ext='dot'
            self.dot_object.write(path='./' + self.__name__ + '.' + ext, format=format, prog=prog)
            # print self.dot_object.create(format=format, prog=prog)

        return self.dot_object


    def status():
        doc = \
        """Status of model. May be one of running, paused, halted or ready.
          - `running` : The model is currently sampling.
          - `paused` : The model has been interrupted during sampling. It is
            ready to be restarted by `continuesample`.
          - `halted` : The model has been interrupted. It cannot be restarted.
            If sample is called again, a new chain will be initiated.
          - `ready` : The model is ready to sample.
        """
        def fget(self):
            return self.__status
        def fset(self, value):
            if value in ['running', 'paused', 'halted', 'ready']:
                self.__status=value
            else:
                raise AttributeError, value
        return locals()
    status = property(**status())

    def seed(self):
        """
        Seed new initial values for the parameters.
        """
        for parameters in self.parameters:
            try:
                if parameters.rseed is not None:
                    value = parameters.random(**parameters.parent_values)
            except:
                pass

    #
    # Tally
    #
    def tally(self):
        """
        tally()

        Records the value of all tracing pymc_objects.
        """
        if self.verbose > 2:
            print self.__name__ + ' tallying.'
        if self._cur_trace_index < self.max_trace_length:
            self.db.tally(self._cur_trace_index)

        self._cur_trace_index += 1
        if self.verbose > 2:
            print self.__name__ + ' done tallying.'

    #
    # Return to a sampled state
    #
    def remember(self, trace_index = None):
        """
        remember(trace_index = randint(trace length to date))

        Sets the value of all tracing pymc_objects to a value recorded in
        their traces.
        """
        if trace_index is None:
            trace_index = randint(self.cur_trace_index)

        for variable in self._variables_to_tally:
            variable.value = variable.trace()[trace_index]


    def save_traces(self, path='', fname=None):
        import cPickle

        if fname is None:
            try:
                fname = self.__name__ + '.pymc'
            except:
                fname = 'Model.pymc'

        trace_dict = {}
        for obj in self._variables_to_tally:
            trace_new = copy(obj.trace)
            trace_new.__delattr__('db')
            trace_new.__delattr__('obj')
            trace_dict[obj.__name__] = trace_new

        F = file(fname, 'w')
        cPickle.dump(trace_dict, F)
        F.close()



class Sampler(Model):
    def __init__(self, input, db='ram', output_path=None, verbose=0):
        
        # Instantiate superclass
        Model.__init__(self, input, db, verbose)
        
        # Instantiate and populate sampling methods set
        self.sampling_methods = set()

        for item in self.input_dict.iteritems():
            if isinstance(item[1], Container):
                self.__dict__[item[0]] = item[1]
                self.sampling_methods.update(item[1].sampling_methods)

            if isinstance(item[1], SamplingMethod):
                self.__dict__[item[0]] = item[1]
                self.sampling_methods.add(item[1])
                setattr(item[1], '_model', self)

        # Default SamplingMethod
        self._assign_samplingmethod()

        self._state = ['status', '_current_iter', '_iter', '_tune_interval', '_burn', '_thin']
            
        # Specify database backend
        self._assign_database_backend(db)
            
        # Instantiate plotter 
        # Hardcoding the matplotlib backend raises error in
        # interactive use. DH
        try:
            self._plotter = Plotter(plotpath=output_path or self.__name__ + '_output/')
        except:
            self._plotter = 'Could not be instantiated.'

    def _assign_database_backend(self, db):
        """Assign Trace instance to parameters and nodes and Database instance
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
        # Tallyable objects are listed in the _pymc_objects_to_tally set. 
        no_trace = getattr(database, 'no_trace')
        self._variables_to_tally = set()
        for object in self.parameters | self.nodes :
            if object.trace:
                self._variables_to_tally.add(object)
            else:
                object.trace = no_trace.Trace()

        # If not already done, load the trace backend from the database 
        # module, and assign a database instance to Model.
        if type(db) is str:
            module = getattr(database, db)
            self.db = module.Database()
        elif isinstance(db, database.base.Database):
            self.db = db
            self.restore_state()
        else:
            module = db.__module__
            self.db = db
        
        
        # Assign Trace instances to tallyable objects. 
        self.db.connect(self)
        
        
    def sample(self, iter, burn=0, thin=1, tune_interval=1000, verbose=0):
        """
        sample(iter, burn, thin, tune_interval)

        Prepare pymc_objects, initialize traces, run MCMC loop.
        """
        
        self._iter = iter
        self._burn = burn
        self._thin = thin
        self._tune_interval = tune_interval
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

    def interactive_sample(self, *args, **kwds):
        # LikelihoodErrors seem to be ending up in the listener
        # thread somehow. I seem to remember something weird about that in the threading
        # documentation? -AP
        self._thread = Thread(target=self.sample, args=args, kwargs=kwds)
        self._thread.start()
        self.interactive_prompt()

    def interactive_continue(self):
        # Restarts thread in interactive mode
        
        self._thread = Thread(target=self._loop)
        self._thread.start()
        
    def tune(self):
        
        # Only tune during burn-in
        if self._current_iter > self._burn:
            self._tuning = False
            return
            
        if self.verbose > 1:
            print '\tTuning at iteration', self._current_iter
        
        # Initialize counter for number of tuning parameters
        tuning_count = 0
        
        for sampling_method in self.sampling_methods:
            # Tune sampling methods
            tuning_count += sampling_method.tune(verbose=self.verbose)
        
        if not tuning_count:
            # If no sampling methods needed tuning, increment count
            self._tuned_count += 1
        else:
            # Otherwise re-initialize count
            self._tuned_count = 0
        
        # 5 consecutive clean intervals removed tuning
        if self._tuned_count == 5:
            if self.verbose > 0: print 'Finished tuning'
            self._tuning = False

    def _loop(self):
        
        # Set status flag
        self.status='running'
        
        try:
            while self._current_iter < self._iter:
                if self.status == 'paused':
                    raise Paused

                i = self._current_iter
                
                if i == self._burn and self.verbose>0: 
                    print 'Burn-in interval complete'
                    
                # Tune at interval
                if i and not (i % self._tune_interval) and self._tuning:
                    self.tune()
                    
                # Tell all the sampling methods to take a step
                for sampling_method in self.sampling_methods:
                        
                    # Step the sampling method
                    sampling_method.step()

                if not i % self._thin and i >= self._burn:
                    self.tally()
                
                if not i % 10000 and self.verbose > 0:
                    print 'Iteration ', i, ' of ', self._iter
                    # Uncommenting this causes errors in some models.
                    # gc.collect()
                    
                self._current_iter += 1


               #self.save_traces() Made obsolete by the pickle database.Right?
        except Paused:
           return None

        except KeyboardInterrupt:
            self.status='halted'
            print '\n Iteration ', i, ' of ', iter
            for variable in self._variables_to_tally:
               variable.trace.truncate(self._cur_trace_index)
            self.save_traces()


        # Finalize
        self.status='ready'
        self.save_state()
        self.db._finalize()
        

    def _assign_samplingmethod(self):
        """
        Make sure every parameter has a SamplingMethod. If not, 
        assign the default SM.
        """

        for parameter in self.parameters:

            # Is it a member of any SamplingMethod?
            homeless = True
            for sampling_method in self.sampling_methods:
                if parameter in sampling_method.parameters:
                    homeless = False
                    break

            # If not, make it a new SamplingMethod using the registry
            if homeless:
                new_method = assign_method(parameter)
                setattr(new_method, '_model', self)
                self.sampling_methods.add(new_method)

    def interactive_prompt(self):
        """
        Drive the sampler from the prompt.

        Commands:
          i -- print current iteration index
          p -- pause
          c -- continue
          q -- quit
        """
        print self.interactive_prompt.__doc__, '\n'
        while True:
            try:
                sys.stdout.write('PyMC> ')
                cmd = raw_input()
                if cmd == 'i':
                    print 'Current iteration: ', self._current_iter
                elif cmd == 'p':
                    self.status = 'paused'
                    print self.status
                elif cmd == 'c':
                    self.interactive_continue()
                    print 'Restarted'
                elif cmd == 'q':
                    print 'Exiting interactive prompt...'
                    break
                else:
                    print 'Unknown command'
                    print self.interactive_prompt.__doc__
            except KeyboardInterrupt:
                print 'Exiting interactive prompt...'
                break

    def get_state(self):
        """Return the sampler and sampling methods current state in order to
        restart sampling at a later time."""
        state = dict(sampler={}, sampling_methods={})
        # The state of the sampler itself.
        for s in self._state:
            state['sampler'][s] = getattr(self, s)

        # The state of each SamplingMethod.
        for sm in self.sampling_methods:
            smstate = {}
            for s in sm._state:
                if hasattr(sm, s):
                    smstate[s] = getattr(sm, s)
            state['sampling_methods'][sm._id] = smstate.copy()

        return state

    def save_state(self):
        """Tell the database to save the current state of the sampler."""
        self.db.savestate(self.get_state())

    def restore_state(self):
        """Restore the state of the sampler and of the sampling methods to
        the state stored in the database.
        """
        state = self.db.getstate()
        self.__dict__.update(state.get('sampler', {}))
        for sm in self.sampling_methods:
            tmp = state.get('sampling_methods', {})
            sm.__dict__.update(tmp.get(sm._id, {}))
            
    def plot(self):
        """
        Plots traces and histograms for nodes and parameters.
        """
        
        # Loop over PyMC objects
        for variable in self._variables_to_tally:            
            # Plot object
            self._plotter.plot(variable)
            
        show()
            
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
                
                # Loop over parameters
                for name in self.parameters:
                    
                    # Look up parameter
                    parameter = self.parameters[name]
                    
                    # Retrieve copy of trace
                    trace = parameter.get_trace(burn=burn, thin=thin, chain=chain, composite=composite)
                    
                    # Sample value from trace
                    sample = trace[random_integers(len(trace)) - 1]
                    
                    # Set current value to sampled value
                    parameter.set_value(sample)
                
                # Run calculate likelihood with sampled parameters
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
