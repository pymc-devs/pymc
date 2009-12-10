from pymc import *

__all__ = ['graph', 'moral_graph']

try:
    import pydot
    pydot_imported = True
except:
    pydot_imported = False

def moral_graph(model, format='raw', prog='dot', path=None):
    """
    moral_graph(model,format='raw', prog='dot', path=None)

    Draws the moral graph for this model and writes it to path.
    Returns the pydot 'dot' object for further user manipulation.

    GraphViz and PyDot must be installed to use this function.

    :Parameters:
      model : PyMC Model instance
      format : string
        'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg',
        'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp',
        'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'
      prog : string
        'dot', 'neato'
      path : string
        If model.__name__ is defined and path is None, the output file is
        ./'name'.'format'.

    :Note:
      format='raw' outputs a GraphViz dot file.
    """
    if not pydot_imported:
        raise ImportError, 'PyDot must be installed to use the moral_graph function.\n PyDot is available from http://dkbza.org/pydot.html'

    model.moral_dot_object = pydot.Dot()

    # Data are filled ellipses
    for datum in model.observed_stochastics:
        model.moral_dot_object.add_node(pydot.Node(name=datum.__name__, style='filled'))

    # Stochastics are open ellipses
    for s in model.stochastics:
        model.moral_dot_object.add_node(pydot.Node(name=s.__name__))

    gone_already = set()
    for s in model.stochastics | model.observed_stochastics:
        gone_already.add(s)
        for other_s in s.moral_neighbors:
            if not other_s in gone_already:
                model.moral_dot_object.add_edge(pydot.Edge(src=other_s.__name__, dst=s.__name__, arrowhead='none'))

    # Draw the graph
    if not path == None:
        model.moral_dot_object.write(path=path, format=format, prog=prog)
    else:
        ext=format
        if format=='raw':
            ext='dot'
        model.moral_dot_object.write(path='./' + model.__name__ + '.' + ext, format=format, prog=prog)

    return model.moral_dot_object


def graph(model, format='raw', prog='dot', path=None, consts=False, legend=False,
        collapse_deterministics = False, collapse_potentials = False, label_edges=True):
    """
    graph(  model,
            format='raw',
            prog='dot',
            path=None,
            consts=False,
            legend=True,
            collapse_deterministics = False,
            collapse_potentials = False)

    Draws the graph for this model and writes it to path.
    Returns the pydot 'dot' object for further user manipulation.

    GraphViz and PyDot must be installed to use this function.

    :Parameters:
      model : PyMC Model instance
      format : string
        'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg',
        'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp',
        'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'
      prog : string
        'dot', 'neato'
      path : string
        If model.__name__ is defined and path is None, the output file is
        ./'name'.'format'.
      consts : boolean
        If True, constant parents are included in the graph.
      legend : boolean
        If True, a graph legend is created.
      collapse_deterministics : boolean
        If True, all deterministic dependencies are collapsed.
      collapse_potentials : boolean
        If True, all potentials are converted to undirected edges.
    """

    if not pydot_imported:
        raise ImportError, 'PyDot must be installed to use the graph function.\n PyDot is available from http://dkbza.org/pydot.html'
    pydot_nodes = {}
    pydot_subgraphs = {}
    obj_substitute_names = {}
    shown_objects = set([])
    model.dot_object = pydot.Dot()
    
    def get_obj_names(obj, key):

        if isinstance(obj, Stochastic):
            if obj_substitute_names.has_key(obj):
                return obj_substitute_names[obj]
            if obj.observed:
                datum = obj
                # Data are filled ellipses
                pydot_nodes[datum] = pydot.Node(name=datum.__name__, style='filled')
                model.dot_object.add_node(pydot_nodes[datum])
                shown_objects.add(datum)
                obj_substitute_names[datum] = [datum.__name__]
            else:
                s = obj
                # Stochastics are open ellipses
                pydot_nodes[s] = pydot.Node(name=s.__name__)
                model.dot_object.add_node(pydot_nodes[s])
                shown_objects.add(s)
                obj_substitute_names[s] = [s.__name__]
    
        elif isinstance(obj, Deterministic):
            if obj_substitute_names.has_key(obj):
                return obj_substitute_names[obj]
            d = obj
            # Deterministics are downward-pointing triangles
            if not collapse_deterministics:
                pydot_nodes[d] = pydot.Node(name=d.__name__, shape='invtriangle')
                model.dot_object.add_node(pydot_nodes[d])
                shown_objects.add(d)
                obj_substitute_names[d] = [d.__name__]
            else:
                obj_substitute_names[d] = []
                    
        elif isinstance(obj, Potential):
            if obj_substitute_names.has_key(obj):
                return obj_substitute_names[obj]
            potential = obj
            # Potentials are squares
            if not collapse_potentials:
                pydot_nodes[potential] = pydot.Node(name=potential.__name__, shape='box')
                model.dot_object.add_node(pydot_nodes[potential])
                shown_objects.add(potential)
                obj_substitute_names[potential] = [potential.__name__]
            else:
                obj_substitute_names[potential]=[]


        elif consts:
            if obj_substitute_names.has_key(key):
                return
            else:
                obj_substitute_names[key] = [key]
                model.dot_object.add_node(pydot.Node(name=key, style='filled'))
                return
    
        else:
            return
            
        return obj_substitute_names[obj]
    
    connected = []

    def maybe_connect_parent(src, dst, label):
        if (src,dst,label) in connected:
            return False
        else:
            connected.append((src,dst,label))
            model.dot_object.add_edge(pydot.Edge(src=src, dst=dst, label=label))
            return True
    
    def connect_parents(node):

        if collapse_deterministics:
            parent_tups = [(s.__name__, s) for s in node.extended_parents]
            if consts: 
                parent_tups += filter(lambda x: not isinstance(x[1],Variable), node.parents.items())
            parent_dict = dict(parent_tups)
        else:
            parent_dict = node.parents

        for key in parent_dict.iterkeys():
            key_val = parent_dict[key]
            label = label_edges*key or ''

            if hasattr(key_val,'__name__'):
                const_node_name = key_val.__name__
            elif len(key_val.__str__()) <= 10:
                const_node_name = key_val.__str__()
            else:
                const_node_name = key_val.__class__.__name__

            if isinstance(key_val, Variable):
                if any([maybe_connect_parent(name, node.__name__, label) for name in get_obj_names(key_val, None)]):
                    connect_parents(key_val)
                        
            elif isinstance(key_val, ContainerBase):
                for var in key_val.variables:
                    if any([maybe_connect_parent(name, node.__name__, label) for name in get_obj_names(var, None)]):
                        connect_parents(var)
                        
            elif consts:
                get_obj_names(key_val, const_node_name)
                maybe_connect_parent(const_node_name, node.__name__, label)
                            
    # Create edges from parent-child relationships between nodes.
    if collapse_potentials:
        shownodes = model.variables
    else:
        shownodes = model.nodes
        
    for node in shownodes:
        get_obj_names(node,None)
        if node in shown_objects:
            connect_parents(node)
            
    if collapse_potentials:
        for potential in model.potentials:
            if collapse_deterministics:
                potential_parents = reduce(lambda x,y: x.union(set(y)), [get_obj_names(p,None) for p in potential.extended_parents], set())
            else:
                potential_parents=set()
                for parent in potential.parents.values():
                    if isinstance(parent, Variable):
                        potential_parents |= set(get_obj_names(parent, None))
                    elif isinstance(parent, ContainerBase):
                        for ult_parent in parent.variables:
                            potential_parents |= set(get_obj_names(parent, None))
            remaining_parents = copy(potential_parents)
            for p1 in potential_parents:
                remaining_parents.discard(p1)
                for p2 in remaining_parents:
                    new_edge = pydot.Edge(src = p2, dst = p1, label=potential.__name__, arrowhead='none')
                    model.dot_object.add_edge(new_edge)
            

    # Add legend if requested
    if legend:
        legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend')
        legend.add_node(pydot.Node(name='data', style='filled'))
        legend.add_node(pydot.Node(name='stochastics'))
        legend.add_node(pydot.Node(name='deterministics', shape='invtriangle'))
        legend.add_node(pydot.Node(name='potentials', shape='box'))
        if consts:
            legend.add_node(pydot.Node(name='constants', style='filled'))
        model.dot_object.add_subgraph(legend)

    # Draw the graph
    if not path == None:
        model.dot_object.write(path=path, format=format, prog=prog)
    else:
        ext=format
        if format=='raw':
            ext='dot'
        model.dot_object.write(path='./' + model.__name__ + '.' + ext, format=format, prog=prog)

    return model.dot_object

# Alias as dag
dag = graph
