# Changeset history
# 22/03/2007 -DH- Added methods to query the SamplingMethod's state and pass it to database.
# 20/03/2007 -DH- Separated Model from Sampler. Removed _prepare(). Commented __setattr__ because it breaks properties.

__docformat__='reStructuredText'

""" Summary"""

from numpy import zeros, floor
from SamplingMethods import SamplingMethod, assign_method
from PyMC2 import database
from PyMCObjects import Parameter, Node, PyMCBase
from Container import Container
from utils import extend_children
import gc, sys
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
        dbase: module name
          Database backend used to tally the samples.
          Implemented backends: None, hdf5, txt.

    Externally-accessible attributes:
      - nodes: All extant Nodes.
      - parameters: All extant Parameters with isdata = False.
      - data: All extant Parameters with isdata = True.
      - pymc_objects: All extant Parameters and Nodes.
      - sampling_methods: All extant SamplingMethods.

    Externally-accessible methods:
       - sample(iter,burn,thin): At each MCMC iteration, calls each sampling_method's step() method.
         Tallies Parameters and Nodes as appropriate.
       - trace(parameter, burn, thin, slice): Return the trace of parameter,
         sliced according to slice or burn and thin arguments.
       - remember(trace_index): Return the entire model to the tallied state indexed by trace_index.
       - DAG: Draw the model as a directed acyclic graph.

    :Note:
        All the plotting functions can probably go on the base namespace and take Parameters as
        arguments.

    :SeeAlso: Sampler, PyMCBase, Parameter, Node, and weight.
    """
    def __init__(self, input, db='ram'):
        """Initialize a Model instance.

        :Parameters:
          - input : module
              A module containing the model definition, in terms of Parameters,
              and Nodes.
          - dbase : string
              The name of the database backend that will store the values
              of the parameters and nodes sampled during the MCMC loop.
          - kwds : dict
              Arguments to pass to instantiate the Database object.
        """

        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.containers = set()
        self.extended_children = None

        self._generations = []
        self.__name__ = None
        self.status = 'ready'

        if hasattr(input,'__name__'):
            self.__name__ = input.__name__
        else:
            try:
                self.__name__ = input['__name__']
            except: 
                self.__name__ = 'PyMC_Model'

        #Change input into a dictionary
        if isinstance(input, dict):
            self.input_dict = input.copy()
        else:
            try:
                # If input is a module, reload it to make a fresh copy.
                reload(input)
            except TypeError:
                pass

            self.input_dict = input.__dict__.copy()

        for name, item in self.input_dict.iteritems():
            if  isinstance(item,PyMCBase) \
                or isinstance(item,SamplingMethod) \
                or isinstance(item,Container):
                self.__dict__[name] = item

            self._fileitem(item)

        self.pymc_objects = self.nodes | self.parameters | self.data

    def _fileitem(self, item):
        """
        Store an item into the proper set:
          - parameters
          - nodes
          - data
          - containers
        """
        # If a dictionary is passed in, open it up.
        if isinstance(item,Container):
            self.containers.add(item)
            self.parameters.update(item.parameters)
            self.data.update(item.data)
            self.nodes.update(item.nodes)

        # File away the PyMC objects
        elif isinstance(item,PyMCBase):
            # Add an attribute to the object referencing the model instance.

            if isinstance(item,Node):
                self.nodes.add(item)

            elif isinstance(item,Parameter):
                if item.isdata:
                    self.data.add(item)
                else:  self.parameters.add(item)

        elif isinstance(item, SamplingMethod):
            self.nodes.update(item.nodes)
            self.parameters.update(item.parameters)
            self.data.update(item.data)
        
    def _extend_children(self):
        """
        Makes a dictionary of self's PyMC objects' 'extended children.'
        """
        self.extended_children = {}
        dummy = PyMCBase('','',{},0,None)
        for pymc_object in self.pymc_objects:
            dummy.children = copy(pymc_object.children)
            extend_children(dummy)
            self.extended_children[pymc_object] = dummy.children

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

    def sample_model_likelihood(self,iter):
        """
        Returns iter samples of (log p(data|this_model_params, this_model) | data, this_model)
        """
        loglikes = zeros(iter)

        if len(self._generations) == 0:
            self._parse_generations()

        try:
            for i in xrange(iter):
                if i % 10000 == 0:
                    print 'Sample ',i,' of ',iter

                for generation in self._generations:
                    for parameter in generation:
                        parameter.random()

                for datum in self.data:
                    loglikes[i] += datum.logp

        except KeyboardInterrupt:
            print 'Sample ',i,' of ',iter
            raise KeyboardInterrupt

        return loglikes


    def DAG(self,format='raw',path=None,consts=True):
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

        self.dot_object = pydot.Dot()

        pydot_nodes = {}
        pydot_subgraphs = {}

        # Create the pydot nodes from pymc objects
        for datum in self.data:
            pydot_nodes[datum] = pydot.Node(name=datum.__name__,shape='box')
            self.dot_object.add_node(pydot_nodes[datum])

        for parameter in self.parameters:
            pydot_nodes[parameter] = pydot.Node(name=parameter.__name__)
            self.dot_object.add_node(pydot_nodes[parameter])

        for node in self.nodes:
            pydot_nodes[node] = pydot.Node(name=node.__name__,shape='invtriangle')
            self.dot_object.add_node(pydot_nodes[node])

        for container in self.containers:
            pydot_nodes[container] = pydot.Node(name=container.__name__,shape='house')

        # # Create subgraphs from pymc sampling methods
        # for sampling_method in self.sampling_methods:
        #     if not isinstance(sampling_method,OneAtATimeMetropolis):
        #         pydot_subgraphs[sampling_method] = pydot.Subgraph(graph_name = sampling_method.__class__.__name__)
        #         for pymc_object in sampling_method.pymc_objects:
        #             pydot_subgraphs[sampling_method].add_node(pydot_nodes[pymc_object])
        #         self.dot_object.add_subgraph(pydot_subgraphs[sampling_method])


        # Create edges from parent-child relationships
        counter = 0
        for pymc_object in self.pymc_objects:
            for key in pymc_object.parents.iterkeys():
                plot_edge=True
                if not isinstance(pymc_object.parents[key],PyMCBase) and not isinstance(pymc_object.parents[key],Container):
                    if consts:
                        parent_name = pymc_object.parents[key].__class__.__name__ + ' const ' + str(counter)
                        self.dot_object.add_node(pydot.Node(name = parent_name, shape = 'trapezium'))
                        counter += 1
                    else:
                        plot_edge=False
                else:
                    parent_name = pymc_object.parents[key].__name__

                if plot_edge:
                    new_edge = pydot.Edge(  src = parent_name,
                                            dst = pymc_object.__name__,
                                            label = key)


                    self.dot_object.add_edge(new_edge)

        # Draw the graph
        if not path == None:
            self.dot_object.write(path=path,format=format)
        else:
            ext=format
            if format=='raw':
                ext='dot'
            self.dot_object.write(path='./' + self.__name__ + '.' + ext,format=format)

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
        if self._cur_trace_index < self.max_trace_length:
            self.db.tally(self._cur_trace_index)

        self._cur_trace_index += 1

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

        for pymc_object in self._pymc_objects_to_tally:
            pymc_object.value = pymc_object.trace()[trace_index]


    def save_traces(self,path='',fname=None):
        import cPickle

        if fname is None:
            try:
                fname = self.__name__ + '.pymc'
            except:
                fname = 'Model.pymc'

        trace_dict = {}
        for obj in self._pymc_objects_to_tally:
            trace_new = copy(obj.trace)
            trace_new.__delattr__('db')
            trace_new.__delattr__('obj')
            trace_dict[obj.__name__] = trace_new

        F = file(fname,'w')
        cPickle.dump(trace_dict,F)
        F.close()



class Sampler(Model):
    def __init__(self, input, db='ram'):
        Model.__init__(self, input, db)
        self.sampling_methods = set()

        for item in self.input_dict.iteritems():
            if isinstance(item[1],Container):
                self.__dict__[item[0]] = item[1]
                self.sampling_methods.update(item[1].sampling_methods)

            if isinstance(item[1],SamplingMethod):
                self.__dict__[item[0]] = item[1]
                self.sampling_methods.add(item[1])
                setattr(item[1], '_model', self)

        # Default SamplingMethod
        self._assign_samplingmethod()

        self._state = ['status', '_current_iter', '_iter', '_thin', '_burn',
            '_tune_interval']
            
        self._assign_database_backend(db)

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
        no_trace = getattr(database,'no_trace')
        self._pymc_objects_to_tally = set()
        for object in self.parameters | self.nodes :
            if object.trace:
                self._pymc_objects_to_tally.add(object)
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
        
        
    def sample(self,iter=1000,burn=0,thin=1,tune_interval=1000,verbose=False):
        """
        sample(iter,burn,thin)

        Prepare pymc_objects, initialize traces, run MCMC loop.
        """
        if iter <= burn:
            raise 'Iteration (%i) must be greater than burn period (%i).'\
                %(iter,burn)

        self._iter = iter
        self._burn = burn
        self._thin = thin
        self._tune_interval = tune_interval
        self._verbose = verbose
        self._cur_trace_index = 0
        length = iter/thin
        self.max_trace_length = length

        self.seed()

        # Initialize database -> initialize traces.
        self.db._initialize(length)

        # Loop
        self._current_iter = 0
        self._loop()

    def interactive_sample(self, *args, **kwds):
        # David- nice work! LikelihoodErrors seem to be ending up in the listener
        # thread somehow. I seem to remember something weird about that in the threading
        # documentation?
        self._thread = Thread(target=self.sample, args=args, kwargs=kwds)
        self._thread.start()
        self.interactive_prompt()

    def interactive_continue(self):
        self._thread = Thread(target=self._loop)
        self._thread.start()

    def _loop(self):
        self.status='running'
        try:
            while self._current_iter < self._iter:
                if self.status == 'paused':
                    raise Paused

                i = self._current_iter

                # Tell all the sampling methods to take a step
                for sampling_method in self.sampling_methods:
                    sampling_method.step()

                if (i % self._thin) == 0:
                    self.tally()

                if (i % self._tune_interval) == 0:
                    self.tune()

                if i % 10000 == 0 and self._verbose:
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
            for pymc_object in self._pymc_objects_to_tally:
               pymc_object.trace.truncate(self._cur_trace_index)
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

    def tune(self):
        """
        Tell all samplingmethods to tune themselves.
        """
        for sampling_method in self.sampling_methods:
            sampling_method.tune()

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
