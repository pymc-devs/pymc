from Model import *

try:
    import pydot
    pydot_imported = True
except:
    pydot_imported = False
    
def moral_graph(model, format='raw', prog='dot', path=None):
    """
    moral_graph(model,
                format='raw', 
                prog='dot', 
                path=None)

    Draws the moral graph for this model and writes it to path.
    If model.__name__ is defined and path is None, the output file is
    ./'name'.'format'.

    Format is a string. Options are:
    'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg', 
    'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp', 
    'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

    format='raw' outputs a GraphViz dot file.
    
    Returns the pydot 'dot' object for further user manipulation.
    """
    if not pydot_imported:
        raise ImportError, 'PyDot must be installed to use the moral_graph function.\n PyDot is available from http://dkbza.org/pydot.html'

    model.moral_dot_object = pydot.Dot()
        
    # Data are filled ellipses
    for datum in model.data:
        model.moral_dot_object.add_node(pydot.Node(name=datum.__name__, style='filled'))

    # Stochastics are open ellipses
    for stoch in model.stochs:
        model.moral_dot_object.add_node(pydot.Node(name=datum.__name__))
    
    gone_already = set()
    for stoch in model.stochs | model.data:
        gone_already.add(stoch)
        for other_stoch in stoch.moral_neighbors:
            if not other_stoch in gone_already:
                model.moral_dot_object.add_edge(pydot.Edge(src=other_stoch.__name__, dst=stoch.__name__, arrowhead='none'))
    
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
        collapse_dtrms = False, collapse_potentials = False):
    """
    graph(  model,
            format='raw', 
            prog='dot', 
            path=None, 
            consts=False, 
            legend=True, 
            collapse_dtrms = False, 
            collapse_potentials = False)

    Draws the graph for this model and writes it to path.
    If model.__name__ is defined and path is None, the output file is
    ./'name'.'format'.

    Format is a string. Options are:
    'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg', 
    'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp', 
    'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

    format='raw' outputs a GraphViz dot file.
    
    If consts is True, constant parents are included in the graph; 
    otherwise they're not.
    
    If collapse_dtrms is True, Deterministics (variables that are determined by their
    parents) are made implicit.
    
    If collapse_potentials is True, potentials are displayed as undirected edges.

    Returns the pydot 'dot' object for further user manipulation.
    """

    if not pydot_imported:
        raise ImportError, 'PyDot must be installed to use the graph function.\n PyDot is available from http://dkbza.org/pydot.html'

    pydot_nodes = {}
    pydot_subgraphs = {}
    obj_substitute_names = {}
    shown_objects = set([])
    model.dot_object = pydot.Dot()
    
        
    # Data are filled ellipses
    for datum in model.data:
        pydot_nodes[datum] = pydot.Node(name=datum.__name__, style='filled')
        model.dot_object.add_node(pydot_nodes[datum])
        shown_objects.add(datum)
        obj_substitute_names[datum] = [datum.__name__]

    # Stochastics are open ellipses
    for stoch in model.stochs:
        pydot_nodes[stoch] = pydot.Node(name=stoch.__name__)
        model.dot_object.add_node(pydot_nodes[stoch])
        shown_objects.add(stoch)
        obj_substitute_names[stoch] = [stoch.__name__]

    # Deterministics are downward-pointing triangles
    for dtrm in model.dtrms:

        if not collapse_dtrms:
            pydot_nodes[dtrm] = pydot.Node(name=dtrm.__name__, shape='invtriangle')
            model.dot_object.add_node(pydot_nodes[dtrm])
            shown_objects.add(dtrm)
            obj_substitute_names[dtrm] = [dtrm.__name__]

        else:
            obj_substitute_names[dtrm] = []
            for parent in dtrm.parents.values():
                if isinstance(parent, Variable):
                    obj_substitute_names[dtrm].append(parent.__name__)
                elif consts:
                    model.dot_object.add_node(pydot.Node(name=parent.__str__(), style='filled'))
                    obj_substitute_names[dtrm].append(parent.__str__())
        
    # Potentials are octagons outlined three times
    for potential in model.potentials:
        if not collapse_potentials:
            pydot_nodes[potential] = pydot.Node(name=potential.__name__, shape='box')
            model.dot_object.add_node(pydot_nodes[potential])
            shown_objects.add(potential)
    
        else:
            potential_parents = set()
            for parent in potential.parents.values():
                if isinstance(parent, Variable):
                    potential_parents |= set(obj_substitute_names[parent])
            remaining_parents = copy(potential_parents)
            for p1 in potential_parents:
                remaining_parents.discard(p1)
                for p2 in remaining_parents:
                    new_edge = pydot.Edge(src = p2, dst = p1, label=potential.__name__, arrowhead='none')
                    model.dot_object.add_edge(new_edge)
            
    # Create edges from parent-child relationships between nodes.
    for node in model.nodes:
        
        if node in shown_objects:

            parent_dict = node.parents
            
            for key in parent_dict.iterkeys():
                
                key_val = parent_dict[key]
                
                if hasattr(key_val,'__name__'):
                    const_node_name = parent_dict.__name__
                elif len(key_val.__str__()) <= 10:
                    const_node_name = key_val.__str__()
                else:
                    const_node_name = key_val.__class__.__name__
            
                if isinstance(parent_dict[key], Variable):

                    for name in obj_substitute_names[key_val]:
                        model.dot_object.add_edge(pydot.Edge(src=name, dst=node.__name__, label=key))
                elif isinstance(parent_dict[key], ContainerBase):
                    for var in key_val.variables:
                        for name in obj_substitute_names[var]:
                            model.dot_object.add_edge(pydot.Edge(src=name, dst=node.__name__, label=key))
                    if len(key_val.variables)==0:
                        if consts:
                            model.dot_object.add_node(pydot.Node(name=const_node_name, style='filled'))
                            model.dot_object.add_edge(pydot.Edge(src=const_node_name, dst=node.__name__, label=key))                        
                    
                elif consts:
                    model.dot_object.add_node(pydot.Node(name=const_node_name, style='filled'))
                    model.dot_object.add_edge(pydot.Edge(src=const_node_name, dst=node.__name__, label=key))                        
            
    # Add legend if requested
    if legend:
        legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend')
        legend.add_node(pydot.Node(name='data', style='filled'))
        legend.add_node(pydot.Node(name='stochs'))
        legend.add_node(pydot.Node(name='dtrms', shape='invtriangle'))
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
        # print model.dot_object.create(format=format, prog=prog)

    return model.dot_object
