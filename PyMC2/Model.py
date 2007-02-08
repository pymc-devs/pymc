__docformat__='reStructuredText'

""" Summary"""

from numpy import zeros, floor
from PyMCObjects import PyMCBase, Parameter, Node
from SamplingMethods import SamplingMethod, OneAtATimeMetropolis
from PyMC2 import database
from PyMCObjectDecorators import extend_children
import gc

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

    :SeeAlso: SamplingMethod, OneAtATimeMetropolis, PyMCBase, Parameter, Node, and weight.
    """
    def __init__(self, input, dbase='memory_trace'):

        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.sampling_methods = set()
        self._generations = []
        self._prepared = False
        self.__name__ = None

        if hasattr(input,'__name__'):
            self.__name__ = input.__name__

        #Change input into a dictionary
        if isinstance(input, dict):
            input_dict = input
        else:
            try:
                # If input is a module, reload it to make a fresh copy.
                reload(input)
            except TypeError:
                pass

            input_dict = input.__dict__

        for item in input_dict.iteritems():
            self._fileitem(item)

        self._assign_database(dbase)

    def _fileitem(self, item):

        # If a dictionary is passed in, open it up.
        if isinstance(item[1],dict):
            for subitem in item[1].iteritems():
                self._fileitem(subitem)

        # If another iterable object is passed in, open it up.
        # Broadcast the same name over all the elements.
        """
        This doesn't work so hot, anyone have a better idea?
        I was trying to allow sets/tuples/lists
        of PyMC objects and SamplingMethods to be passed in.

        elif iterable(item[1]) == 1:
            for subitem in item[1]:
                self._fileitem((item[0],subitem))
        """
        # File away the SamplingMethods
        if isinstance(item[1],SamplingMethod):
            # Teach the SamplingMethod its name
            item[1].__name__ = item[0]
            #File it away
            self.__dict__[item[0]] = item[1]
            self.sampling_methods.add(item[1])
            setattr(self.__dict__[item[0]], '_model', self)

        # File away the PyMC objects
        elif isinstance(item[1],PyMCBase):
            self.__dict__[item[0]] = item[1]
            # Add an attribute to the object referencing the model instance.
            #setattr(self.__dict__[item[0]], '_model', self)

            if isinstance(item[1],Node):
                self.nodes.add(item[1])

            elif isinstance(item[1],Parameter):
                if item[1].isdata:
                    self.data.add(item[1])
                else:  self.parameters.add(item[1])

    #
    # Override __setattr__ so that PyMC objects are read-only once instantiated
    #
    def __setattr__(self, name, value):

        # Don't allow changes to PyMC object attributes
        if self.__dict__.has_key(name):
            if isinstance(self.__dict__[name],PyMCBase):
                raise AttributeError, 'Attempt to write read-only attribute of Model.'

            # Do allow other objects to be changed
            else:
                self.__dict__[name] = value

        # Allow new attributes to be created.
        else:
            self.__dict__[name] = value

    def _assign_database(self, dbase):
        """Assign trace instance to parameters and nodes.
        Assign database initialization methods to the Model class.

        Defined databases:
          - None: Traces stored in memory.
          - Txt: Traces stored in memory and saved in txt files at end of
                sampling. Not implemented.
          - SQLlite: Traces stored in sqllite database. Not implemented.
          - HDF5: Traces stored in HDF5 database. Partially implemented.
        """
        # Load the trace backend.
        if not dbase:
            dbase = 'no_trace'
        db = getattr(database, dbase)
        no_trace = getattr(database, 'no_trace')
        reload(db)

        # Assign database instance to Model.
        self.db = db.database(self)
        
        # Assign trace instance to parameters and nodes.
        for object in self.parameters | self.nodes :
            if object.trace:
                object.trace = db.trace(object, self)
            else:
                object.trace = no_trace.trace(object)

        
    #
    # Prepare for sampling
    #
    def _prepare(self):

        # Initialize database
        self.db._initialize()

        # Seed new initial values for the parameters.
        for parameters in self.parameters:
            try:
                if parameters.rseed is not None:
                    value = parameters.random(**parameters.parent_values)
            except:
                pass
        if self._prepared:
            return

        self._prepared = True

        # Tell all pymc_objects to get ready for sampling
        self.pymc_objects = self.nodes | self.parameters | self.data
        for pymc_object in self.pymc_objects:
            extend_children(pymc_object)

        # Take care of singleton parameters
        for parameter in self.parameters:

            # Is it a member of any SamplingMethod?
            homeless = True
            for sampling_method in self.sampling_methods:
                if parameter in sampling_method.parameters:
                    homeless = False
                    break

            # If not, make it a new one-at-a-time Metropolis-Hastings SamplingMethod
            if homeless:
                self.sampling_methods.add(OneAtATimeMetropolis(parameter))


    #
    # Initialize traces
    #
    def _init_traces(self, length):
        """
        init_traces(length)

        Enumerates the pymc_objects that are to be tallied and initializes traces
        for them.

        To be tallied, a pymc_object has to pass the following criteria:

            -   It is not included in the argument pymc_objects_not_to_tally.

            -   Its value has a shape.

            -   Its value can be made into a numpy array with a numerical
                dtype.
        """
        self._pymc_objects_to_tally = set()
        self._cur_trace_index = 0
        self.max_trace_length = length

        for pymc_object in self.parameters | self.nodes:
            if hasattr(pymc_object.trace,'_initialize'):
                pymc_object.trace._initialize(length)
                self._pymc_objects_to_tally.add(pymc_object)

    #
    # Tally
    #
    def tally(self):
        """
        tally()

        Records the value of all tracing pymc_objects.
        """
        if self._cur_trace_index < self.max_trace_length:
            for pymc_object in self._pymc_objects_to_tally:
                pymc_object.trace.tally(self._cur_trace_index)

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
        if trace_index:
            trace_index = randint(self.cur_trace_index)

        for pymc_object in self._pymc_objects_to_tally:
            pymc_object.value = pymc_object.trace()[trace_index]

    #
    # Run the MCMC loop!
    #
    def sample(self,iter,burn,thin):
        """
        sample(iter,burn,thin)

        Prepare pymc_objects, initialize traces, run MCMC loop.
        """

        # Do various preparations for sampling
        self._prepare()

        self._iter = iter
        self._burn = burn
        self._thin = thin

        # Initialize traces
        self._init_traces(iter/thin)
        
        try:
            for i in xrange(iter):

                # Tell all the sampling methods to take a step
                for sampling_method in self.sampling_methods:
                    sampling_method.step()

                if i % thin == 0:
                    self.tally()

                if i % 10000 == 0:
                    gc.collect()
                    print 'Iteration ', i, ' of ', iter
                    
        except KeyboardInterrupt:
            print '\n Iteration ', i, ' of ', iter

        # Tuning, etc.

        # Finalize
        self.db._finalize(burn, thin)

    def tune(self):
        """
        Tell all samplingmethods to tune themselves.
        """
        for sampling_method in self.sampling_methods:
            sampling_method.tune()

    def _parse_generations(self):
        """
        Parse up the _generations for model averaging.
        """
        self._prepare()
        
        # Find root generation
        self._generations.append(set())
        all_children = set()
        for parameter in self.parameters:
            all_children.update(parameter.children & self.parameters)
        self._generations[0] = self.parameters - all_children

        # Find subsequent _generations
        children_remaining = True
        gen_num = 0
        while children_remaining:
            gen_num += 1
            

            # Find children of last generation
            self._generations.append(set())
            for parameter in self._generations[gen_num-1]:
                self._generations[gen_num].update(parameter.children & self.parameters)

            
            # Take away parameters that have parents in the current generation.
            thisgen_children = set()
            for parameter in self._generations[gen_num]:
                thisgen_children.update(parameter.children & self.parameters)
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



    def DAG(self,format='raw',path=None):
        """
        DAG(format='raw', path=None)

        Draw the directed acyclic graph for this model and writes it to path.
        If self.__name__ is defined and path is None, the output file is
        ./'name'.'format'. If self.__name__ is undefined and path is None,
        the output file is ./model.'format'.

        Format is a string. Options are:
        'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg',
        'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp',
        'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

        format='raw' outputs a GraphViz dot file.

        Returns the pydot 'dot' object for further user manipulation.
        """

        if not self._prepared:
            self._prepare()

        if self.__name__ == None:
            self.__name__ = model

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

        # Create subgraphs from pymc sampling methods
        for sampling_method in self.sampling_methods:
            if not isinstance(sampling_method,OneAtATimeMetropolis):
                pydot_subgraphs[sampling_method] = subgraph(graph_name = sampling_method.__class__.__name__)
                for pymc_object in sampling_method.pymc_objects:
                    pydot_subgraphs[sampling_method].add_node(pydot_nodes[pymc_object])
                self.dot_object.add_subgraph(pydot_subgraphs[sampling_method])


        # Create edges from parent-child relationships
        counter = 0
        for pymc_object in self.pymc_objects:
            for key in pymc_object.parents.iterkeys():
                if not isinstance(pymc_object.parents[key],PyMCBase):

                    parent_name = pymc_object.parents[key].__class__.__name__ + ' const ' + str(counter)
                    self.dot_object.add_node(pydot.Node(name = parent_name, shape = 'trapezium'))
                    counter += 1
                else:
                    parent_name = pymc_object.parents[key].__name__

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
