import itertools

from theano.gof.graph import ancestors

from .util import get_default_varnames
import pymc3 as pm
from .model import build_dependence_dag_from_model
import networkx as nx


def powerset(iterable):
    """All *nonempty* subsets of an iterable.

    From itertools docs.

    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))


class ModelGraph(object):
    def __init__(self, model):
        self.model = model
        self.var_names = get_default_varnames(self.model.named_vars, include_transformed=False)
        self.var_list = self.model.named_vars.values()
        self.transform_map = {v.transformed: v.name for v in self.var_list if hasattr(v, 'transformed')}
        self._deterministics = None

    def get_deterministics(self, var):
        """Compute the deterministic nodes of the graph"""
        deterministics = []
        attrs = ('transformed', 'logpt')
        for v in self.var_list:
            if v != var and all(not hasattr(v, attr) for attr in attrs):
                deterministics.append(v)
        return deterministics

    def _ancestors(self, var, func, blockers=None):
        """Get ancestors of a function that are also named PyMC3 variables"""
        return set([j for j in ancestors([func], blockers=blockers) if j in self.var_list and j != var])

    def _get_ancestors(self, var, func):
        """Get all ancestors of a function, doing some accounting for deterministics

        Specifically, if a deterministic is an input, theano.gof.graph.ancestors will
        return only the inputs *to the deterministic*.  However, if we pass in the
        deterministic as a blocker, it will skip those nodes.
        """
        deterministics = self.get_deterministics(var)
        upstream = self._ancestors(var, func)

        # Usual case
        if upstream == self._ancestors(var, func, blockers=upstream):
            return upstream
        else: # deterministic accounting
            for d in powerset(upstream):
                blocked = self._ancestors(var, func, blockers=d)
                if set(d) == blocked:
                    return d
        raise RuntimeError('Could not traverse graph. Consider raising an issue with developers.')

    def _filter_parents(self, var, parents):
        """Get direct parents of a var, as strings"""
        keep = set()
        for p in parents:
            if p == var:
                continue
            elif p.name in self.var_names:
                keep.add(p.name)
            elif p in self.transform_map:
                if self.transform_map[p] != var.name:
                    keep.add(self.transform_map[p])
            else:
                raise AssertionError('Do not know what to do with {}'.format(str(p)))
        return keep

    def get_parents(self, var):
        """Get the named nodes that are direct inputs to the var"""
        if hasattr(var, 'transformed'):
            func = var.transformed.logpt
        elif hasattr(var, 'logpt'):
            func = var.logpt
        else:
            func = var

        parents = self._get_ancestors(var, func)
        return self._filter_parents(var, parents)

    def make_compute_graph(self):
        """Get map of var_name -> set(input var names) for the model"""
        input_map = {}
        for var_name in self.var_names:
            input_map[var_name] = self.get_parents(self.model[var_name])
        return input_map

    def _make_node(self, var_name, graph):
        """Attaches the given variable to a graphviz Digraph"""
        v = self.model[var_name]

        # styling for node
        attrs = {}
        if isinstance(v, pm.model.ObservedRV):
            attrs['style'] = 'filled'

        # Get name for node
        if hasattr(v, 'distribution'):
            distribution = v.distribution.__class__.__name__
        else:
            distribution = 'Deterministic'
            attrs['shape'] = 'box'

        graph.node(var_name,
                '{var_name} ~ {distribution}'.format(var_name=var_name, distribution=distribution),
                **attrs)

    def get_plates(self):
        """ Rough but surprisingly accurate plate detection.

        Just groups by the shape of the underlying distribution.  Will be wrong
        if there are two plates with the same shape.

        Returns
        -------
        dict: str -> set[str]
        """
        plates = {}
        for var_name in self.var_names:
            v = self.model[var_name]
            if hasattr(v, 'observations'):
                shape = v.observations.shape
            elif hasattr(v, 'dshape'):
                shape = v.dshape
            else:
                shape = v.tag.test_value.shape
            if shape == (1,):
                shape = tuple()
            if shape not in plates:
                plates[shape] = set()
            plates[shape].add(var_name)
        return plates

    def make_graph(self):
        """Make graphviz Digraph of PyMC3 model

        Returns
        -------
        graphviz.Digraph
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError('This function requires the python library graphviz, along with binaries. '
                              'The easiest way to install all of this is by running\n\n'
                              '\tconda install -c conda-forge python-graphviz')
        graph = graphviz.Digraph(self.model.name)
        for shape, var_names in self.get_plates().items():
            label = ' x '.join(map('{:,d}'.format, shape))
            if label:
                # must be preceded by 'cluster' to get a box around it
                with graph.subgraph(name='cluster' + label) as sub:
                    for var_name in var_names:
                        self._make_node(var_name, sub)
                    # plate label goes bottom right
                    sub.attr(label=label, labeljust='r', labelloc='b', style='rounded')
            else:
                for var_name in var_names:
                    self._make_node(var_name, graph)

        for key, values in self.make_compute_graph().items():
            for value in values:
                graph.edge(value, key)
        return graph


def model_to_graphviz(model=None):
    """Produce a graphviz Digraph from a PyMC3 model.

    Requires graphviz, which may be installed most easily with
        conda install -c conda-forge python-graphviz

    Alternatively, you may install the `graphviz` binaries yourself,
    and then `pip install graphviz` to get the python bindings.  See
    http://graphviz.readthedocs.io/en/stable/manual.html
    for more information.
    """
    model = pm.modelcontext(model)
    return ModelGraph(model).make_graph()


class OtherModelGraph(object):
    def __init__(self, model):
        self.model = model
        try:
            graph = model.dependence_dag
        except AttributeError:
            graph = build_dependence_dag_from_model(model)
        self.set_graph(graph)

    def set_graph(self, graph):
        self.graph = graph
        self.node_names = {}
        unnamed_count = 0
        for n in self.graph.nodes():
            try:
                name = n.name
            except AttributeError:
                name = 'Unnamed {}'.format(unnamed_count)
                unnamed_count += 1
            self.node_names[n] = name

    def draw(self, pos=None, draw_nodes=False, ax=None,
             edge_kwargs=None,
             label_kwargs=None,
             node_kwargs=None):
        graph = self.graph
        if edge_kwargs is None:
            edge_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}
        if label_kwargs is None:
            label_kwargs = {}
        label_kwargs.setdefault('bbox', {'boxstyle': 'round',
                                         'facecolor': 'lightgray'})
        if pos is None:
            try:
                pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
            except Exception:
                pos = nx.shell_layout(graph)
        d = nx.get_edge_attributes(graph, 'deterministic')
        edgelist = list(d.keys())
        edge_color = [float(v) for v in d.values()]
        labels = {n: n.name for n in graph}

        if draw_nodes:
            nx.draw_networkx_edges(graph, pos=pos, ax=ax, **node_kwargs)
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, edgelist=edgelist,
                               edge_color=edge_color, **edge_kwargs)
        nx.draw_networkx_labels(graph, pos=pos, ax=ax, labels=labels,
                                **label_kwargs)

    def get_plates(self, graph=None, ignore_transformed=True):
        """ Groups nodes by the shape of the underlying distribution, and if
        the nodes form a disconnected component of the graph.

        Parameters
        ----------
        graph: networkx.DiGraph (optional)
            The graph object from which to get the plates. If None, self.graph
            will be used.
        ignore_transformed: bool (optional)
            If True, the transformed variables will be ignored while getting
            the plates.

        Returns
        -------
        list of tuples: (shape, set(nodes_in_plate))
        """
        if graph is None:
            graph = self.graph
        if ignore_transformed:
            transforms = set([n.transformed for n in graph
                              if hasattr(n, 'transformed')])
            nbunch = [n for n in graph if n not in transforms]
            graph = nx.subgraph(graph, nbunch)
        shape_plates = {}
        for node in graph:
            if hasattr(node, 'observations'):
                shape = node.observations.shape
            elif hasattr(node, 'dshape'):
                shape = node.dshape
            else:
                try:
                    shape = node.tag.test_value.shape
                except AttributeError:
                    shape = tuple()
            if shape == (1,):
                shape = tuple()
            if shape not in shape_plates:
                shape_plates[shape] = set()
            shape_plates[shape].add(node)
        plates = []
        for shape, nodes in shape_plates.items():
            # We want to find the disconnected components that have a common
            # shape. These will be the plates
            subgraph = nx.subgraph(graph, nodes).to_undirected()
            for G in nx.connected_component_subgraphs(subgraph, copy=False):
                plates.append((shape, set(G.nodes())))
        return plates

    def make_graph(self, ignore_transformed=True, edge_cmap=None):
        """Make graphviz Digraph of PyMC3 model

        Returns
        -------
        graphviz.Digraph
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError('This function requires the python library graphviz, along with binaries. '
                              'The easiest way to install all of this is by running\n\n'
                              '\tconda install -c conda-forge python-graphviz')

        G = self.graph
        if ignore_transformed:
            transforms = set([n.transformed for n in G
                              if hasattr(n, 'transformed')])
            nbunch = [n for n in G if n not in transforms]
            G = nx.subgraph(G, nbunch)
        graph = graphviz.Digraph(self.model.name)
        nclusters = 0
        for shape, nodes in self.get_plates(graph=G):
            label = ' x '.join(map('{:,d}'.format, shape))
            if label:
                # must be preceded by 'cluster' to get a box around it
                with graph.subgraph(name='cluster {}'.format(nclusters)) as sub:
                    nclusters += 1
                    for node in nodes:
                        self._make_node(node, sub)
                    # plate label goes bottom right
                    sub.attr(label=label, labeljust='r', labelloc='b', style='rounded')
            else:
                for node in nodes:
                    self._make_node(node, graph)

        for from_node, to_node, ats in G.edges(data=True):
            if edge_cmap is None:
                edge_color = '#000000'
            else:
                from matplotlib import colors
                val = float(ats['deterministic'])
                edge_color = colors.to_hex(edge_cmap(val), keep_alpha=True)
            graph.edge(self.node_names[from_node],
                       self.node_names[to_node],
                       color=edge_color)
        return graph

    def _make_node(self, node, graph):
        """Attaches the given variable to a graphviz Digraph"""
        # styling for node
        attrs = {}
        if isinstance(node, pm.model.ObservedRV):
            attrs['style'] = 'filled'

        # Get name for node
        if hasattr(node, 'distribution'):
            distribution = node.distribution.__class__.__name__
        else:
            distribution = 'Deterministic'
            attrs['shape'] = 'box'
        var_name = self.node_names[node]

        graph.node(var_name,
                '{var_name} ~ {distribution}'.format(var_name=var_name, distribution=distribution),
                **attrs)


def crude_draw(model, *args, **kwargs):
    OtherModelGraph(model).draw(*args, **kwargs)
