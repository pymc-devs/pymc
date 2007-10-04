from Model import *

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
    for pymc_object in self.containers.extend(self.pymc_objects):
        
        if pymc_object in shown_objects:
            if hasattr(pymc_object,'owner'):
                pymc_object = pymc_object.owner
            parent_dict = pymc_object.parents
        
            for key in parent_dict.iterkeys():
            
                # If a parent is a container, unpack it.
                # Draw edges between child and all elements of container (if consts=True)
                # or all variables in container (if consts = False).
                if isinstance(parent_dict[key], ContainerBase) or isinstance(parent_dict[key], Variable):
                    key_val = parent_dict[key]
                    if isinstance(key_val, ContainerBase):
                        key_val = key_val.container
                        
                        # TODO: Fix bug here.
                        for name in obj_substitute_names[key_val]:
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
