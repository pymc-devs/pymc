# TODO: Make DAG depend on py2app, not pydot. Less dependencies.
# See modulegraph.modulegraph.Dot,
# in the Py2App part of site-packages.

__docformat__='reStructuredText'

""" Summary"""

from numpy import zeros, floor
from AbstractBase import *
from SamplingMethods import SamplingMethod, OneAtATimeMetropolis
from PyMC2 import database
from utils import extend_children
import gc
from copy import copy
GuiInterrupt = 'Computation halted'

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
        self.sampling_methods = set()
        self.extended_children = None
        self.containers = set()
        self._generations = []
        self._prepared = False
        self.__name__ = None
        self.sampling = False
        self.ready = False
        self._current_iter = 0  # Indicate that sampling is not started yet.
        
        if hasattr(input,'__name__'):
            self.__name__ = input.__name__
        else:
            self.__name__ = 'PyMC_Model'

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
            if  isinstance(item[1],PyMCBase) \
                or isinstance(item[1],SamplingMethod) \
                or isinstance(item[1],ContainerBase):
                self.__dict__[item[0]] = item[1]
 
            self._fileitem(item)

        self._assign_database_backend(db)

    def _fileitem(self, item):

        # If a dictionary is passed in, open it up.
        if isinstance(item[1],ContainerBase):
            self.containers.add(item[1])
            self.parameters.update(item[1].parameters)
            self.data.update(item[1].data)
            self.nodes.update(item[1].nodes)
            self.pymc_objects.update(item[1].pymc_objects)
            self.sampling_methods.update(item[1].sampling_methods)

        # File away the SamplingMethods
        if isinstance(item[1],SamplingMethod):
            # Teach the SamplingMethod its name
            #File it away
            self.sampling_methods.add(item[1])
            setattr(item[1], '_model', self)

        # File away the PyMC objects
        elif isinstance(item[1],PyMCBase):
            # Add an attribute to the object referencing the model instance.

            if isinstance(item[1],NodeBase):
                self.nodes.add(item[1])

            elif isinstance(item[1],ParameterBase):
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
        # If not already done, load the trace backend from the database module,
        # and assign database instance to Model.
        if type(db) is str:
            module = getattr(database, db)
            self.db = module.Database()
        else:
            module = db.__module__
            self.db = db
            
        no_trace = getattr(database,'no_trace')
        #reload(no_trace) is that necessary ?
        
        self._pymc_objects_to_tally = set()
        # Assign trace instance to parameters and nodes.
        for object in self.parameters | self.nodes :
            if object.trace:
                object.trace = module.Trace(object, self.db)
                self._pymc_objects_to_tally.add(object)
            else:
                object.trace = no_trace.Trace(object,self.db)


        
    #
    # Prepare for sampling
    #
    def _prepare(self):

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
        if trace_index is None:
            trace_index = randint(self.cur_trace_index)

        for pymc_object in self._pymc_objects_to_tally:
            pymc_object.value = pymc_object.trace()[trace_index]

    #
    # Run the MCMC loop!
    #
    def sample(self,iter=1000,burn=0,thin=1, verbose=True, tune_interval=1000):
        """
        sample(iter,burn,thin)

        Prepare pymc_objects, initialize traces, run MCMC loop.
        """
        self.sampling = True
        
        if iter <= burn:
            raise 'Iteration (%i) must be greater than burn period (%i).'\
                %(iter,burn)

        if self._current_iter == 0:
            # Do various preparations for sampling
            self._prepare()
    
            self._iter = iter
            self._burn = burn
            self._thin = thin
            self._cur_trace_index = 0
            length = iter/thin    
            self.max_trace_length = length
            
        # Initialize database -> initialize traces. 
        if not self.ready:
            self.db._initialize(length, self)
            self.ready = True
        
        try:
            while self._current_iter < self._iter:
                if not self.sampling:
                    return None
                
                i = self._current_iter 

                # Tell all the sampling methods to take a step
                for sampling_method in self.sampling_methods:
                    sampling_method.step()

                if (i % thin) == 0:
                    self.tally()

                if (i % tune_interval) == 0:
                    self.tune()

                if i % 10000 == 0 and verbose:
                    print 'Iteration ', i, ' of ', iter
                    # Uncommenting this causes errors in some models.
                    # gc.collect()
                
                self._current_iter += 1
                
            self.save_traces()
        except (KeyboardInterrupt, GuiInterrupt):
            print '\n Iteration ', i, ' of ', iter
            for pymc_object in self._pymc_objects_to_tally:
                pymc_object.trace.truncate(self._cur_trace_index)
            self.save_traces()
            self.sampling = False
            
            

        # Tuning, etc.

        # Finalize
        self.db._finalize()
        self.sampling = False
        self.ready = False
        self._current_iter = 0
        
    def tune(self):
        """
        Tell all samplingmethods to tune themselves.
        """
        for sampling_method in self.sampling_methods:
            sampling_method.tune()
    
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
        print trace_dict
        cPickle.dump(trace_dict,F)
        F.close()

    def _extend_children(self):
        """
        Makes a dictionary of self's PyMC objects' 'extended children.'
        """
        self.extended_children = {}
        dummy = PyMCBase()
        for pymc_object in self.pymc_objects:
            dummy.children = copy(pymc_object.children)
            extend_children(dummy)
            self.extended_children[pymc_object] = dummy.children
        

    def _parse_generations(self):
        """
        Parse up the _generations for model averaging.
        """
        if not self._prepared:
            self._prepare()
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
        DAG(format='raw', path=None)

        Draw the directed acyclic graph for this model and writes it to path.
        If self.__name__ is defined and path is None, the output file is
        ./'name'.'format'. 

        Format is a string. Options are:
        'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg',
        'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp',
        'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

        format='raw' outputs a GraphViz dot file.

        Returns the pydot 'dot' object for further user manipulation.
        """

        if not self._prepared:
            self._prepare()

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

        # Create subgraphs from pymc sampling methods
        for sampling_method in self.sampling_methods:
            if not isinstance(sampling_method,OneAtATimeMetropolis):
                pydot_subgraphs[sampling_method] = pydot.Subgraph(graph_name = sampling_method.__class__.__name__)
                for pymc_object in sampling_method.pymc_objects:
                    pydot_subgraphs[sampling_method].add_node(pydot_nodes[pymc_object])
                self.dot_object.add_subgraph(pydot_subgraphs[sampling_method])


        # Create edges from parent-child relationships
        counter = 0
        for pymc_object in self.pymc_objects:
            for key in pymc_object.parents.iterkeys():
                plot_edge=True
                if not isinstance(pymc_object.parents[key],PyMCBase) and not isinstance(pymc_object.parents[key],ContainerBase):
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
