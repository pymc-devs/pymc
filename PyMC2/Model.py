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
from Container import ContainerBase
from utils import extend_children, extend_parents
import gc, sys,os
from copy import copy
from threading import Thread
from thread import interrupt_main
from time import sleep

GuiInterrupt = 'Computation halt'
Paused = 'Computation paused'

# TODO: Should Model be a subclass of Container, to eliminate item-filing code duplication?
# TODO: Yes.
class Model(object):
    """
    Model is initialized with:

      >>> A = Model(prob_def, dbase=None)

    :Arguments:
        input : class, module, dictionary
          Contains variables, potentials and containers.
        db : module name
          Database backend used to tally the samples.
          Implemented backends: 'ram', 'no_trace', 'hdf5', 'txt', 'mysql', 'pickle', 'sqlite'.

    Externally-accessible attributes:
      - nodes
      - parameters (with isdata=False)
      - data (parameters with isdata=True)
      - variables
      - potentials
      - containers
      - pymc_objects
      - status: Not useful for the Model base class, but may be used by subclasses.
      
    The following attributes only exist after the appropriate method is called:
      - moral_edges: The edges of the moralized graph. A dictionary, keyed by parameter,
        whose values are sets of parameters. Edges exist between the key parameter and all parameters
        in the value. Created by method _moralize.
      - extended_children: The extended children of self's parameters. See the docstring of
        extend_children. This is a dictionary keyed by parameters. Created by method _extend_children.
      - generations: A list of sets of parameters. The members of each element only have parents in 
        previous elements. Created by method _parse_generations.

    Externally-accessible methods:
       - tally(index): Write all variables' current values to trace, at location index.
       - sample_model_likelihood(iter): Generate and return iter samples of p(data and potentials|model).
         Can be used to generate Bayes' factors.
       - save_traces(): Pickle and save traces to disk. XXX Do we still need this method?
       - graph(...): Draw graphical representation of model. See docstring.
       - seed() XXX I don't know what this does...
       - plot(): Visualize traces for all variables
       - remember(trace_index) : Return the entire model to the tallied state indexed by trace_index.

    :SeeAlso: Sampler, MAP, NormalApproximation, weight.
    """
    def __init__(self, input, db='ram', output_path=None, verbose=0):
        """Initialize a Model instance.

        :Parameters:
          - input : module
              A module containing the model definition, in terms of Parameters, 
              and Nodes.
          - db : string
              The name of the database backend that will store the values
              of the parameters and nodes sampled during the MCMC loop.
          - output_path : string
              The place where any output files should be put.
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
        self.generations = []
        self.__name__ = None
        
        # Flag for model state
        self.status = 'ready'

        # Check for input name
        if hasattr(input, '__file__'):
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
                or isinstance(item, ContainerBase):
                self.__dict__[name] = item
                
            # Allocate to appropriate set
            self._fileitem(item)
            
        # Union of PyMC objects
        self.variables = self.nodes | self.parameters | self.data
        self.pymc_objects = self.variables | self.potentials
        
        
        # Moved this stuff here so I can use it from NormalApproximation. -AP
        # Specify database backend
        self._assign_database_backend(db)
        
        # Instantiate plotter 
        # Hardcoding the matplotlib backend raises error in
        # interactive use. DH
        try:
            self._plotter = Plotter(plotpath=output_path or self.__name__ + '_output/')
        except:
            self._plotter = 'Could not be instantiated.'
        
    def _fileitem(self, item):
        """
        Store an item into the proper set:
          - parameters
          - nodes
          - data
          - containers
          - potentials
        """
        # If a dictionary is passed in, open it up.
        if isinstance(item, ContainerBase):
            self.containers.add(item)
            self.parameters.update(item.parameters)
            self.data.update(item.data)
            self.nodes.update(item.nodes)
            self.potentials.update(item.potentials)

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
    
    def _moralize(self):
        """
        Creates moral adjacency matrix for self.
        
        self.moral_edges[parameter] returns a list of the parameters with whom
        parameter shares an edge in the moral graph.
        """
        # Extend children
        self._extend_children()
        
        # Initialize moral edges dictionary.
        self.moral_edges = {}
        for parameter in self.parameters | self.data:
            self.moral_edges[parameter] = set([])
        
        # Fill in.
        remaining_params = copy(self.parameters | self.data)
        for parameter in self.parameters:
            self_and_children = set([parameter]) | self.extended_children[parameter]
            remaining_params.remove(parameter)
            for other_parameter in remaining_params:
                other_self_and_children = set([other_parameter]) | self.extended_children[other_parameter]
                if len(self_and_children.intersection(other_self_and_children))>0:
                    self.moral_edges[parameter].add(other_parameter)
                    self.moral_edges[other_parameter].add(parameter)

                    
    def _get_maximal_cliques(self):
        """
        Creates list of maximal cliques for self. Each has an attribute called
        logp, which gives the log-potential associated with the clique.
        """
        
        # Moralize self
        self._moralize()
    
        # Find maximal cliques
        self.maximal_cliques = []
        remaining_params = copy(self.parameters | self.data)
        while len(remaining_params)>0:
            parameter = remaining_params.pop()
            this_clique = set([parameter])
            self.maximal_cliques.append(this_clique)
            
            for other_parameter in remaining_params:
                if all([other_parameter in self.moral_edges[clique_parameter] for clique_parameter in this_clique]):
                    this_clique.add(other_parameter)
            for clique_parameter in this_clique:
                remaining_params.discard(clique_parameter)
                
        # TODO: Find edges between maximal cliques, potentials associated with them.
        # TODO: Make clique a subclass of Container -> make Container able to wrap sets!
        # TODO: Make mapping from parameter to maximal clique containing it.
                
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
        self._extend_children()

        # Find root generation
        self.generations.append(set())
        all_children = set()
        for parameter in self.parameters:
            all_children.update(self.extended_children[parameter] & self.parameters)
        self.generations[0] = self.parameters - all_children

        # Find subsequent _generations
        children_remaining = True
        gen_num = 0
        while children_remaining:
            gen_num += 1


            # Find children of last generation
            self.generations.append(set())
            for parameter in self.generations[gen_num-1]:
                self.generations[gen_num].update(self.extended_children[parameter] & self.parameters)


            # Take away parameters that have parents in the current generation.
            thisgen_children = set()
            for parameter in self.generations[gen_num]:
                thisgen_children.update(self.extended_children[parameter] & self.parameters)
            self.generations[gen_num] -= thisgen_children


            # Stop when no subsequent _generations remain
            if len(thisgen_children) == 0:
                children_remaining = False

    def sample_model_likelihood(self, iter):
        """
        Returns iter samples of (log p(data|this_model_params, this_model) | data, this_model)
        """
        # TODO: Restructure this using the actor model in stackless branch: if all my extended parents have sampled their value, then I can also.
        loglikes = zeros(iter)

        if len(self.generations) == 0:
            self._parse_generations()

        try:
            for i in xrange(iter):
                if i % 10000 == 0:
                    print 'Sample ', i, ' of ', iter

                for generation in self.generations:
                    for parameter in generation:
                        parameter.random()

                for datum in self.data | self.potentials:
                    loglikes[i] += datum.logp

        except KeyboardInterrupt:
            print 'Sample ', i, ' of ', iter
            raise KeyboardInterrupt

        return loglikes


    def graph(self, format='raw', prog='dot', path=None, consts=False, legend=False, 
            collapse_nodes = False, collapse_potentials = False, collapse_containers = False):
        """
        M.graph(format='raw', 
                prog='dot', 
                path=None, 
                consts=False, 
                legend=True, 
                collapse_nodes = False, 
                collapse_potentials = False, 
                collapse_containers = False)

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
        
        If collapse_nodes is True, Nodes (variables that are determined by their
        parents) are made implicit.
        
        If collapse_containers is True, containers are shown as single graph nodes.
        
        If collapse_potentials is True, potentials are displayed as undirected edges.

        Returns the pydot 'dot' object for further user manipulation.
        
        NOTE: Will endow all containers with an innocuous 'parents' attribute.
        """

        import pydot

        pydot_nodes = {}
        pydot_subgraphs = {}
        obj_substitute_names = {}
        shown_objects = set([])
        
        # Get ready to separate self's PyMC objects that are contained in containers.
        uncontained_params = self.parameters.copy()
        uncontained_data = self.data.copy()
        uncontained_nodes = self.nodes.copy()
        uncontained_potentials = self.potentials.copy()
        
        for container in self.containers:
            container.dot_object = pydot.Cluster(graph_name = container.__name__, label = container.__name__)
            uncontained_params -= container.parameters
            uncontained_nodes -= container.nodes
            uncontained_data -= container.data
            uncontained_potentials -= container.potentials
            
        # Use this to make a graphviz cluster corresponding to each container, and to
        # draw the model outside of any container.
        def create_graph(subgraph):
            
            # Data are filled ellipses
            for datum in subgraph.data:
                pydot_nodes[datum] = pydot.Node(name=datum.__name__, style='filled')
                subgraph.dot_object.add_node(pydot_nodes[datum])
                shown_objects.add(datum)
                obj_substitute_names[datum] = [datum.__name__]

            # Parameters are open ellipses
            for parameter in subgraph.parameters:
                pydot_nodes[parameter] = pydot.Node(name=parameter.__name__)
                subgraph.dot_object.add_node(pydot_nodes[parameter])
                shown_objects.add(parameter)
                obj_substitute_names[parameter] = [parameter.__name__]

            # Nodes are downward-pointing triangles
            for node in subgraph.nodes:

                if not collapse_nodes:
                    pydot_nodes[node] = pydot.Node(name=node.__name__, shape='invtriangle')
                    subgraph.dot_object.add_node(pydot_nodes[node])
                    shown_objects.add(node)
                    obj_substitute_names[node] = [node.__name__]

                else:
                    obj_substitute_names[node] = []
                    for parent in node.parents.values():
                        if isinstance(parent, Variable):
                            obj_substitute_names[node].append(parent.__name__)
                        elif consts:
                            subgraph.dot_object.add_node(pydot.Node(name=parent.__str__(), style='filled'))
                            obj_substitute_names[node].append(parent.__str__())
                
            # Potentials are octagons outlined three times
            for potential in subgraph.potentials:
                if not collapse_potentials:
                    pydot_nodes[potential] = pydot.Node(name=potential.__name__, shape='tripleoctagon')
                    subgraph.dot_object.add_node(pydot_nodes[potential])
                    shown_objects.add(potential)

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

            # Get containers ready to be graph nodes.
            if collapse_containers:
                shown_objects.add(container)
                obj_substitute_names[container] = [container.__name__]
                U.dot_object.add_node(pydot.Node(name=container.__name__,shape='box'))
                for variable in container.variables:
                    obj_substitute_names[variable] = [container.__name__]
                container.parents = {}
                for pymc_object in container.pymc_objects:
                    for key in pymc_object.parents.keys():
                        if not pymc_object.parents[key] in self.pymc_objects:
                            container.parents[pymc_object.__name__ + '_' + key] = pymc_object.parents[key]

            # Create a grahpviz cluster for each container.
            else:
                create_graph(container)
                U.dot_object.add_subgraph(container.dot_object)
                obj_substitute_names[container] = set()
                for variable in container.variables:
                    obj_substitute_names[container] |= set(obj_substitute_names[variable])
            
        self.dot_object = U.dot_object
        
        # If the user has requested potentials be collapsed, draw in the undirected edges.
        # TODO: Unpack container parents here.
        if collapse_potentials:
            for pot in self.potentials:
                pot_parents = set()
                for parent in pot.parents.values():
                    if isinstance(parent, Variable):
                        pot_parents |= set(obj_substitute_names[parent])
                remaining_parents = copy(pot_parents)
                for p1 in pot_parents:
                    remaining_parents.discard(p1)
                    for p2 in remaining_parents:
                        new_edge = pydot.Edge(src = p2, dst = p1, label=pot.__name__, arrowhead='none')
                        self.dot_object.add_edge(new_edge)
                
        # Create edges from parent-child relationships between PyMC objects.
        
        for pymc_object in self.pymc_objects | self.containers:
            
            if pymc_object in shown_objects:
                parent_dict = pymc_object.parents
            
                for key in parent_dict.iterkeys():
                
                    # If a parent is a container, unpack it.
                    # Draw edges between child and all elements of container (if consts=True)
                    # or all variables in container (if consts = False).
                    if isinstance(parent_dict[key], ContainerBase) or isinstance(parent_dict[key], Variable):
                        for name in obj_substitute_names[parent_dict[key]]:
                            self.dot_object.add_edge(pydot.Edge(src=name, dst=pymc_object.__name__, label=key))
                    elif consts:
                        U.dot_object.add_node(pydot.Node(name=parent_dict[key].__str__(), style='filled'))
                        self.dot_object.add_edge(pydot.Edge(src=parent_dict[key].__str__(), dst=pymc_object.__name__, label=key))                        
                
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
        Seed new initial values for the parameters.
        """
        for parameters in self.parameters:
            try:
                if parameters.rseed is not None:
                    value = parameters.random(**parameters.parent_values)
            except:
                pass

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
        
    def _get_logp(self):
        """
        Current joint probability of model.
        """
        return sum([p.logp for p in self.parameters]) + sum([p.logp for p in self.data]) + sum([p.logp for p in self.potentials])

    logp = property(_get_logp)
    
    def tally(self, index):
        self.db.tally(index)
    
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
        
        # Moved this to Model so I can use it in NormalApproximation. -AP
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
        
    def plot(self):
        """
        Plots traces and histograms for nodes and parameters.
        """

        # Loop over PyMC objects
        for variable in self._variables_to_tally:            
            # Plot object
            self._plotter.plot(variable)

        # show()
    
    


class Sampler(Model):
    # TODO: Docstring!!
    def __init__(self, input, db='ram', output_path=None, verbose=0):
        
        # Instantiate superclass
        Model.__init__(self, input, db, output_path, verbose)
        
        # Instantiate and populate sampling methods set
        self.sampling_methods = set()

        for item in self.input_dict.iteritems():
            if isinstance(item[1], ContainerBase):
                self.__dict__[item[0]] = item[1]
                self.sampling_methods.update(item[1].sampling_methods)

            if isinstance(item[1], SamplingMethod):
                self.__dict__[item[0]] = item[1]
                self.sampling_methods.add(item[1])
                setattr(item[1], '_model', self)

        # Default SamplingMethod
        self._assign_samplingmethod()

        self._state = ['status', '_current_iter', '_iter', '_tune_interval', '_burn', '_thin']
        
    def _assign_samplingmethod(self):
        """
        Make sure every parameter has a sampling method. If not, 
        assign a sampling method from the registry.
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
        
    def _loop(self):
        # Set status flag
        self.status='running'

        try:
            while self._current_iter < self._iter and not self.status == 'halt':
                if self.status == 'paused':
                    print 'Pausing at iteration ', self._current_iter, ' of ', self._iter
                    return None

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

        except KeyboardInterrupt:
            self.status='halt'
            
        if self.status == 'halt':
            self.halt_sampling()
            
        # Finalize
        print 'Sampling finished normally.'
        self.status = 'ready'
        self.save_state()
        self.db._finalize()
        # TODO: This should interrupt the main thread immediately, but it waits until 
        # TODO: return is pressed before doing its thing. Bug report filed at python.org.
        try:
            interrupt_main()
        except KeyboardInterrupt:
            pass

    def halt_sampling(self):
        print 'Halting at iteration ', self._current_iter, ' of ', self._iter
        for variable in self._variables_to_tally:
           variable.trace.truncate(self._cur_trace_index)
    
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
        
    def tune(self):
        """
        Tell all sampling methods to tune themselves.
        """

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
                print 'Call interactive_continue method to continue, or call halt_sampling method to truncate traces and stop.'

    def get_state(self):
        """
        Return the sampler and sampling methods current state in order to
        restart sampling at a later time.
        """
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
        """
        Tell the database to save the current state of the sampler.
        """
        self.db.savestate(self.get_state())

    def restore_state(self):
        """
        Restore the state of the sampler and of the sampling methods to
        the state stored in the database.
        """
        state = self.db.getstate()
        self.__dict__.update(state.get('sampler', {}))
        for sm in self.sampling_methods:
            tmp = state.get('sampling_methods', {})
            sm.__dict__.update(tmp.get(sm._id, {}))
                        
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
