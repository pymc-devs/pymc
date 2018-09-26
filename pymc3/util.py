import re
import numbers
from collections.abc import Hashable
import functools
import numpy as np
from numpy import asscalar
import theano

LATEX_ESCAPE_RE = re.compile(r'(%|_|\$|#|&)', re.MULTILINE)


def escape_latex(strng):
    """Consistently escape LaTeX special characters for _repr_latex_ in IPython

    Implementation taken from the IPython magic `format_latex`

    Examples
    --------
        escape_latex('disease_rate')  # 'disease\_rate'

    Parameters
    ----------
    strng : str
        string to escape LaTeX characters

    Returns
    -------
    str
        A string with LaTeX escaped
    """
    if strng is None:
        return u'None'
    return LATEX_ESCAPE_RE.sub(r'\\\1', strng)


def get_transformed_name(name, transform):
    """
    Consistent way of transforming names

    Parameters
    ----------
    name : str
        Name to transform
    transform : transforms.Transform
        Should be a subclass of `transforms.Transform`

    Returns
    -------
    str
        A string to use for the transformed variable
    """
    return "{}_{}__".format(name, transform.name)


def is_transformed_name(name):
    """
    Quickly check if a name was transformed with `get_transormed_name`

    Parameters
    ----------
    name : str
        Name to check

    Returns
    -------
    bool
        Boolean, whether the string could have been produced by
        `get_transormed_name`
    """
    return name.endswith('__') and name.count('_') >= 3


def get_untransformed_name(name):
    """
    Undo transformation in `get_transformed_name`. Throws ValueError if name
    wasn't transformed

    Parameters
    ----------
    name : str
        Name to untransform

    Returns
    -------
    str
        String with untransformed version of the name.
    """
    if not is_transformed_name(name):
        raise ValueError(
            u'{} does not appear to be a transformed name'.format(name))
    return '_'.join(name.split('_')[:-3])


def get_default_varnames(var_iterator, include_transformed):
    """Helper to extract default varnames from a trace.

    Parameters
    ----------
    varname_iterator : iterator
        Elements will be cast to string to check whether it is transformed,
        and optionally filtered
    include_transformed : boolean
        Should transformed variable names be included in return value

    Returns
    -------
    list
        List of variables, possibly filtered
    """
    if include_transformed:
        return list(var_iterator)
    else:
        return [var for var in var_iterator
                if not is_transformed_name(str(var))]


def get_variable_name(variable):
    """Returns the variable data type if it is a constant, otherwise
    returns the argument name.
    """
    name = variable.name
    if name is None:
        if hasattr(variable, 'get_parents'):
            try:
                names = [get_variable_name(item)
                         for item in variable.get_parents()[0].inputs]
                # do not escape_latex these, since it is not idempotent
                return 'f(%s)' % ',~'.join([n for n in names
                                            if isinstance(n, str)])
            except IndexError:
                pass
        value = variable.eval()
        if not value.shape:
            return asscalar(value)
        return 'array'
    return r'\text{%s}' % name


def update_start_vals(a, b, model):
    """Update a with b, without overwriting existing keys. Values specified for
    transformed variables on the original scale are also transformed and
    inserted.
    """
    if model is not None:
        for free_RV in model.free_RVs:
            tname = free_RV.name
            for name in a:
                if (is_transformed_name(tname) and
                        get_untransformed_name(tname) == name):
                    transform_func = [d.transformation for d in
                                      model.deterministics if d.name == name]
                    if transform_func:
                        b[tname] = transform_func[0].forward_val(
                            a[name], point=b)

    a.update({k: v for k, v in b.items() if k not in a})


def get_transformed(z):
    if hasattr(z, 'transformed'):
        z = z.transformed
    return z


def biwrap(wrapper):
    @functools.wraps(wrapper)
    def enhanced(*args, **kwargs):
        is_bound_method = hasattr(args[0], wrapper.__name__) if args else False
        if is_bound_method:
            count = 1
        else:
            count = 0
        if len(args) > count:
            newfn = wrapper(*args, **kwargs)
            return newfn
        else:
            newwrapper = functools.partial(wrapper, *args, **kwargs)
            return newwrapper
    return enhanced


class ConstantNodeException(Exception):
    pass


def not_shared_or_constant_variable(x):
    return (isinstance(x, theano.Variable) and
            not(isinstance(x, theano.Constant) or
                isinstance(x, theano.tensor.sharedvar.SharedVariable))
            )


class DependenceDAG(object):
    """
    `DependenceDAG` instances represent the directed acyclic graph (DAG) that
    represents the conditional and deterministic relationships between
    random variables and other kinds of `theano` Variables.

    Conditional relations are only relevant for random variables, and they
    imply that a random variable's distribution is conditionally dependent on
    some other variable's value. Conditional relations are extracted from
    the random distribution's `conditional_on` attribute.

    Deterministic relations are defined by a variable's `theano` `Apply` node's
    owner inputs. The ownership relations are recursively searched to get the
    named `theano.Variables` that are linked together.

    The `DependenceDAG` is used to allow `distribution.draw_values` to
    efficiently move through the DAG starting from independent variables,
    depth zero nodes, down in the relational hierarchy. This allows
    `distribution.draw_values` to be able to sample from the joint probability
    distribution of given random variables instead of combining the results
    drawn from the two marginal distributions.

    The relationships are split into deterministic and conditional because
    a given variable could be either computed directly from other nodes
    or its value could be sampled conditional to other variables (e.g.
    e.g. autotransformed RVs). In this case, deterministic relations should
    have precedence over conditional ones.

    The depth of the nodes contained in the DAG is calculated as the
    maximum between the deterministic and conditional depths. The deterministic
    depth is maximum depth of the deterministic parents, plus one. The
    conditional depth is the same, but with respect to the conditional parents.
    The nodes depths enable the structuring of the DAG into layers of nodes
    that are both conditionally and deterministically independent from each
    other. These layers can then be transversed by other sampling functions.

    """

    def __init__(self, nodes=set(),
                 deterministic_parents={}, deterministic_children={},
                 conditional_parents={}, conditional_children={},
                 deterministic_depth={}, conditional_depth={},
                 check_integrity=True):
        """
        Parameters
        ----------
        nodes: set (optional)
            set of nodes (edges) stored in the DAG
        deterministic_parents: dictionary (optional)
            Dictionary whose keys are nodes (stored in nodes set), and whose
            values are a set of nodes. The sets' nodes are the ones needed to
            deterministically compute the key node's value.
        deterministic_children: dictionary (optional)
            Dictionary whose keys are nodes (stored in nodes set), and whose
            values are a set of nodes. The sets' nodes are the ones that
            require the value of the key's node to be able to deterministically
            their own value.
        conditional_parents: dictionary (optional)
            Dictionary whose keys are nodes (stored in nodes set), and whose
            values are a set of nodes, whose values are needed to
            conditionally draw random sample the key node's value.
        conditional_children: dictionary (optional)
            Dictionary whose keys are nodes (stored in nodes set), and whose
            values are a set of nodes. The sets' nodes are the ones that
            require the value of the key's node to be able to randomly sample
            their own value.
        deterministic_depth: dictionary (optional)
            Dictionary whose keys are nodes (stored in nodes set), and whose
            values are ints, which represent the key's deterministic distance.
        conditional_depth: dictionary (optional)
            Dictionary whose keys are nodes (stored in nodes set), and whose
            values are ints, which represent the key's conditional distance.
        check_integrity: bool (optional)
            If True, the integrity of the DAG is checked. This means that the
            supplied dictionaries keys must all be in the `nodes` set, their
            depths must be positive integers, their relations (both
            deterministic and conditional parents and children) must all be
            members of the `nodes` set. For now, the acyclicness of the graph
            is not tested.

        """
        self.nodes = set(nodes)
        self.deterministic_parents = deterministic_parents.copy()
        self.deterministic_children = deterministic_children.copy()
        self.conditional_parents = conditional_parents.copy()
        self.conditional_children = conditional_children.copy()
        self.deterministic_depth = deterministic_depth.copy()
        self.conditional_depth = conditional_depth.copy()
        self.depth = {node: max([self.deterministic_depth[node],
                                 self.conditional_depth[node]])
                      for node in self.nodes}
        if check_integrity:
            self.check_integrity()

    def __contains__(self, node):
        return node in self.nodes

    def __iter__(self):
        for node in self.nodes:
            yield node

    def __str__(self):
        s = ('<{}.{} object at {}>'.
             format(self.__class__.__module__,
                    self.__class__.__name__,
                    hex(id(self)))
             )
        return s

    def __repr__(self):
        s = ('<{}.{} object at {}>\n'
             'Contents:\n'
             'nodes = {}\n'
             'deterministic_parents = {}\n'
             'deterministic_children = {}\n'
             'deterministic_depth = {}\n'
             'conditional_parents = {}\n'
             'conditional_children = {}\n'
             'conditional_depth = {}\n'
             'depth = {}\n'.format(self.__class__.__module__,
                                   self.__class__.__name__,
                                   hex(id(self)),
                                   self.nodes,
                                   self.deterministic_parents,
                                   self.deterministic_children,
                                   self.deterministic_depth,
                                   self.conditional_parents,
                                   self.conditional_children,
                                   self.conditional_depth,
                                   self.depth,
                                   )
             )
        return s

    def __eq__(self, other):
        if len(self.nodes) != len(other.nodes):
            return False
        for at_name in ['deterministic_parents',
                        'deterministic_children',
                        'conditional_parents',
                        'conditional_children',
                        'deterministic_depth',
                        'conditional_depth']:
            sat = getattr(self, at_name)
            oat = getattr(other, at_name)
            for node in self:
                svals = sat[node]
                ovals = oat[node]
                if isinstance(svals, set):
                    if len(svals) != len(ovals):
                        return False
                    for sval in svals:
                        if sval not in ovals:
                            return False
                else:
                    if svals != ovals:
                        return False
        return True

    def check_integrity(self):
        """
        Check the integrity of the DependenceDAG. This means that the
        instance's relationship dictionaries keys must all be in the `nodes`
        set, their depths must be positive integers, their relations (both
        deterministic and conditional parents and children) must all be
        members of the `nodes` set. For now, the acyclicness of the graph
        is not tested.

        """
        nodes = self.nodes
        for at_name in ['deterministic_parents',
                        'deterministic_children',
                        'conditional_parents',
                        'conditional_children']:
            at = getattr(self, at_name)
            if (len(nodes) != len(at.keys()) or
                    any([node not in at for node in nodes])):
                raise ValueError(
                    'DependenceDAG is broken. The set of stored nodes is '
                    'different from the keys of attribute {}'.
                    format(at_name))
            for node, relations in at.items():
                # Test if parents and children listed are included in nodes
                for relation in relations:
                    if relation not in nodes:
                        raise ValueError(
                            'DependenceDAG is broken. The set of parents '
                            'or children nodes, contain nodes that are '
                            'not stored in the nodes attribute. '
                            'Encountered for `{at_name}` relationship '
                            'of node `{node}`. The element `{relation}` '
                            'is not found in the stored nodes set.'.
                            format(at_name=at_name,
                                   node=node,
                                   relation=relation))
        for at_name in ['deterministic_depth',
                        'conditional_depth',
                        'depth']:
            at = getattr(self, at_name)
            if (len(nodes) != len(at.keys()) or
                    any([node not in at for node in nodes])):
                raise ValueError(
                    'DependenceDAG is broken. The set of stored nodes is '
                    'different from the keys of attribute {}'.
                    format(at_name))
            for n, d in at.items():
                if not isinstance(d, int):
                    raise TypeError(
                        'DependenceDAG is broken. The {} of node `{}` holds a '
                        'depth value, {}, that is not an integer'.
                        format(at_name, n, d))
                if d < 0:
                    raise ValueError(
                        'DependenceDAG is broken. The {} of node `{}` holds a '
                        'depth value, {}, that is lower than zero'.
                        format(at_name, n, d))
                if at_name == 'depth':
                    expected_depth = max([self.deterministic_depth[n],
                                          self.conditional_depth[n]])
                    if d != expected_depth:
                        raise ValueError(
                            "DependenceDAG is broken. The depth of node `{}` "
                            "is different from the expected depth, which "
                            "is computable from the node's deterministic and "
                            "conditional depths. Expected depth {}, instead "
                            "got {}".
                            format(n, expected_depth, d))
        return True

    def add(self, node, force=False, return_added_node=False,
            accept_cons_shared=False):
        """Add a node and its conditional and deterministic parents along with
        their relations, recursively into the `DependenceDAG` instance. To
        allow method chaining, self is returned at the end of this call.

        Parameters
        ----------
        node: The variable to add to the DAG (mandatory)
            By default `theano.Variable`'s, which could be a pymc random
            variable, Deterministic or Potential, are allowed. TensorConstants
            and SharedVariables are not allowed by default, but this behavior
            can be changed with either the `force` or `accept_cons_shared`
            inputs.
            Other unhashable types are only accepted if `force=True`, and they
            are wrapped by `WrapAsHashable` instances.
        force: bool (optional)
            If True, any type of node, except None, is allowed to be added.
        accept_cons_shared: bool (optional)
            If True, `theano` `TensorConstant`s and `theano` `SharedVariable`s
            are allowed to be added to the DAG. These are treated separately
            a priori because `_draw_value` handles these cases differently.
        return_added_node: bool (optional)
            If True, the node which was added to the DAG is returned along with
            the `DependenceDAG` instance. This may be useful because the added
            node may be a `WrapAsHashable` instance which wraps to inputed node
            depending on its type.

        Returns
        -------
        The `DependenceDAG` instance to which the node was added (`self`).
        If `return_added_node` is `True`, the `DependenceDAG` instance is
        packed into a tuple along with the node that was actually added into
        the DAG. Usually this node is the input node, but depending on the
        node's type, it could be a `WrapAsHashable` instance.
        """
        if node is None:
            raise TypeError('None is not allowed to be added as a node in a '
                            'DependenceDAG')
        if not isinstance(node, theano.Variable):
            if not force:
                raise TypeError(
                    "By default, it is not allowed to add nodes that "
                    "are not `theano.Variable`'s to a `DependenceDAG` "
                    "instance. Got node `{}` of type `{}`. "
                    "However, this kind of node could be added by "
                    "passing `force=True` to `add`. It would be "
                    "wrapped by a `WrapAsHashable` instead. This "
                    "wrapped node can be returned by `add` by passing "
                    "`return_added_node=True`.".
                    format(node, type(node)))
            node = WrapAsHashable(node)
        elif not (not_shared_or_constant_variable(node) or
                  hasattr(node, 'distribution')):
            if not (force or accept_cons_shared):
                raise ConstantNodeException(
                    'Supplied node, of type `{}`, does not have a '
                    '`distribution` attribute or is an instance of a `theano` '
                    '`Constant` or `SharedVariable`. This node could be '
                    'accepted by passing either `force=True` or '
                    '`accept_cons_shared=True` to `add`.'.
                    format(type(node)))
        if node in self:
            # Should we raise a warning with we attempt to add a node that is
            # already in the DAG??
            if return_added_node:
                return self, node
            else:
                return self

        # Add node into the nodes set and then initiate all node relations and
        # values to their defaults
        self.nodes.add(node)
        self.deterministic_parents[node] = set()
        self.deterministic_children[node] = set()
        self.conditional_parents[node] = set()
        self.conditional_children[node] = set()
        self.deterministic_depth[node] = 0
        self.conditional_depth[node] = 0
        self.depth[node] = 0

        # Try to get the conditional parents of node and add them
        try:
            cond = node.distribution.conditional_on
        except AttributeError:
            cond = None
        if cond is not None:
            conditional_depth = 0
            for conditional_parent in self.walk_down_ownership(cond):
                if conditional_parent not in self:
                    try:
                        self.add(conditional_parent)
                    except ConstantNodeException:
                        continue
                self.conditional_parents[node].add(conditional_parent)
                parent_conditional_depth = self.depth[conditional_parent]
                if conditional_depth <= parent_conditional_depth:
                    conditional_depth = parent_conditional_depth + 1
                self.conditional_children[conditional_parent].add(node)
                self.conditional_depth[node] = conditional_depth

        # Try to get the deterministic parents of node and add them
        if not_shared_or_constant_variable(node):
            deterministic_depth = 0
            for deterministic_parent in self.walk_down_ownership([node],
                                                                 ignore=True):
                if deterministic_parent not in self:
                    try:
                        self.add(deterministic_parent)
                    except ConstantNodeException:
                        continue
                self.deterministic_parents[node].add(deterministic_parent)
                parent_deterministic_depth = self.depth[deterministic_parent]
                if deterministic_depth <= parent_deterministic_depth:
                    deterministic_depth = parent_deterministic_depth + 1
                self.deterministic_children[deterministic_parent].add(node)
                self.deterministic_depth[node] = deterministic_depth

        # Compute the node's depth
        self.depth[node] = max([self.deterministic_depth[node],
                                self.conditional_depth[node]])
        if return_added_node:
            return self, node
        return self

    def walk_down_ownership(self, node_list, ignore=False):
        """This function goes through an iterable of nodes provided in
        `node_list`, yielding the non None named nodes in the ownership graph.
        With the optional input `ignore`, a node without a name can be yielded.
        """
        for node in node_list:
            if hasattr(node, 'name') and node.name is not None and not ignore:
                yield node
            elif not_shared_or_constant_variable(node):
                owner = node.owner
                if owner is not None:
                    for parent in self.walk_down_ownership(owner.inputs):
                        yield parent

    def copy(self):
        # We assume that self already passes the `check_integrity` test
        return DependenceDAG(
                nodes=self.nodes,
                deterministic_parents=self.deterministic_parents,
                deterministic_children=self.deterministic_children,
                conditional_parents=self.conditional_parents,
                conditional_children=self.conditional_children,
                deterministic_depth=self.deterministic_depth,
                conditional_depth=self.conditional_depth,
                check_integrity=False)

    def get_sub_dag(self, input_nodes, force=True, return_index=False):
        """Get a new DependenceDAG instance which is like a right outer join
        of `self` with a list of input nodes provided in `input_nodes`.
        What this means is that it will look for the `input_nodes` inside
        `self`, the nodes which are contained in `self` will be copied, along
        with all their parents onto a new DependenceDAG instance. Then, the
        remaining nodes will be added to the `DependenceDAG` instance.
        Finally, this instance is returned. In summary, it copies the shared
        part of `self`, given the nodes in `input_nodes`, and then adds onto
        that.

        Parameters
        ----------
        input_nodes: list or scalar (mandatory)
            If it is a scalar `input_nodes` will be converted to a list as
            `[input_nodes]`. `input_nodes` is a list of nodes that will be
            used to create a new `DependenceDAG` instance. The part of the DAG
            that is shared with `self`, will be copied, and the rest will be
            added.
        force: bool (optional)
            If True, the nodes that must be added, will be added with the
            force flag set to True. [Default is True]
        return_index: bool (optional)
            If True, this function will also return a dictionary of indices
            to nodes `{index: node}`. Each key will be the position of the
            node provided in `input_nodes` and the value will be the node
            that was added to the DependenceDAG, which could either be a
            `WrapAsHashable` instance or `input_nodes[index]` itself.

        Returns
        -------
        The `DependenceDAG` instance that results from the right outer join
        operation.
        If `return_index` is `True`, the `DependenceDAG` instance is
        packed into a tuple along with the dictionary of indices to nodes
        `{index: node}`. Each key will be the position of the node provided in
        `input_nodes` and the value will be the node that was added to the
        returned `DependenceDAG` instance, which could either be a
        `WrapAsHashable` instance or `input_nodes[index]` itself.
        """
        if not isinstance(input_nodes, list):
            stack = [input_nodes]
        else:
            stack = input_nodes.copy()

        # We first stack the input_nodes along with their indeces in the
        # input_nodes iterable.
        stack = list(zip(stack, range(len(stack))))
        nodes_to_copy = set()
        nodes_to_add = []
        index = {}

        # Main loop, pop the stack, check if the node is in self, if so, mark
        # it for copying, and add its parents to the stack, if not, mark it
        # for addition
        while True:
            try:
                node, node_index = stack.pop()
            except Exception:
                break
            try:
                node_in_self = node in self
            except TypeError:  # Unhashable node are marked for addition
                node_in_self = False
            if node_in_self:
                # This node is already known to the DAG and can be copied
                # without any extra computation
                if node_index is not None:
                    index[node_index] = node
                nodes_to_copy.add(node)

                # Now we need to add this node's parents to the stack.
                # However, the parents must be ignored for the return_index
                # computation, so we zip them with a list of Nones
                cond_parents = [(p, None) for p in
                                self.conditional_parents[node]]
                stack.extend(cond_parents)
                det_parents = [(p, None) for p in
                               self.deterministic_parents[node]]
                stack.extend(det_parents)
            else:
                # This node is unknown to the DAG and it must be added taking
                # care of all its potencial conditional and deterministic
                # dependence
                nodes_to_add.append((node, node_index))

        # Copy the known nodes
        if nodes_to_copy:
            det_pars = {k: self.deterministic_parents[k]
                        for k in nodes_to_copy}
            det_chis = {k: set([c for c in self.deterministic_children[k]
                                if c in nodes_to_copy])
                        for k in nodes_to_copy}
            con_pars = {k: self.conditional_parents[k]
                        for k in nodes_to_copy}
            con_chis = {k: set([c for c in self.conditional_children[k]
                                if c in nodes_to_copy])
                        for k in nodes_to_copy}
            det_deps = {k: self.deterministic_depth[k]
                        for k in nodes_to_copy}
            con_deps = {k: self.conditional_depth[k]
                        for k in nodes_to_copy}
            output = DependenceDAG(nodes=nodes_to_copy,
                                   deterministic_parents=det_pars,
                                   deterministic_children=det_chis,
                                   conditional_parents=con_pars,
                                   conditional_children=con_chis,
                                   deterministic_depth=det_deps,
                                   conditional_depth=con_deps,
                                   check_integrity=False)
        else:
            output = DependenceDAG(check_integrity=False)

        # Add in the unknown nodes
        for node, node_index in nodes_to_add:
            # We ask the added node to be returned, because `add` could have
            # wrapped it with a WrapAsHashable instance
            output, added_node = output.add(node,
                                            force=force,
                                            return_added_node=True)
            if node_index is not None:
                index[node_index] = added_node
        if return_index:
            return output, index
        return output

    def get_nodes_in_depth_layers(self):
        """Get the DependenceDAG as a list of layers. Each list holds a list
        of the nodes with a common depth. All nodes at depth `d` will be
        in layers[`d`].

        """
        layers = {}
        max_depth = 0
        for node, depth in self.depth.items():
            if depth not in layers:
                layers[depth] = [node]
            else:
                layers[depth].append(node)
            if depth > max_depth:
                max_depth = depth
        if not layers:
            return []
        return [layers[depth] for depth in range(max_depth + 1)]


class WrapAsHashable(object):
    """Generic class that can provide a `__hash__` and `__eq__` method to any
    object, which will be referred to as node. It also provides a multipurpose
    `get_value` function to get the value of the wrapped node.

    If a node is hashable, then __hash__ and __eq__ will both return the same
    as the node's __hash__ and __eq__. If the node is unhashable, the __eq__
    will compare the node's id with the other object's id, and the __hash__
    will be made from a hash of (type(node), id(node)).

    """
    def __init__(self, node):
        self.node_is_hashable = isinstance(node, Hashable)
        self.node = node

    def __hash__(self):
        if self.node_is_hashable:
            return self.node.__hash__()
        else:
            return hash((type(self.node), id(self.node)))

    def __eq__(self, other):
        if isinstance(other, WrapAsHashable):
            if self.node_is_hashable and other.node_is_hashable:
                return self.node == other.node
            elif not self.node_is_hashable and not other.node_is_hashable:
                return id(self.node) == id(other.node)
            else:
                return False
        else:
            if self.node_is_hashable:
                return self.node == other
            else:
                return id(self.node) == id(other)

    def get_value(self):
        node = self.node
        if isinstance(node, numbers.Number):
            return node
        elif isinstance(node, np.ndarray):
            return node
        elif isinstance(node, theano.tensor.TensorConstant):
            return node.value
        elif isinstance(node, theano.tensor.sharedvar.SharedVariable):
            return node.get_value()
        else:
            try:
                return node.get_value()
            except AttributeError:
                raise TypeError("WrapAsHashable instance's node, {}, does not "
                                "define a `get_value` method and is not of "
                                "known types. type(node) = {}.".
                                format(node, type(node)))
