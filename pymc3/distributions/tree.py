import numpy as np
import math
from copy import deepcopy


class TreeStructureError(Exception):
    """Base (catch-all) tree structure exception."""


class TreeNodeError(Exception):
    """Base (catch-all) tree node exception."""


class Tree:
    """ Full binary tree
    A full binary tree is a tree where each node has exactly zero or two children.
    This structure is used as the basic component of the Bayesian Additive Regression Tree (BART)
    Attributes
    ----------
    tree_structure : dict
        A dictionary that represents the nodes stored in breadth-first order, based in the array method
        for storing binary trees (https://en.wikipedia.org/wiki/Binary_tree#Arrays).
        The dictionary's keys are integers that represent the nodes position.
        The dictionary's values are objects of type SplitNode or LeafNode that represent the nodes of the tree itself.
    num_nodes : int
        Total number of nodes.
    idx_leaf_nodes : list
        List with the index of the leaf nodes of the tree.
    idx_prunable_split_nodes : list
        List with the index of the prunable splitting nodes of the tree. A splitting node is prunable if both
        its children are leaf nodes.
    tree_id : int
        Identifier used to get the previous tree in the ParticleGibbs algorithm used in BART.

    Parameters
    ----------
    tree_id : int, optional
    """

    def __init__(self, tree_id=0):
        self.tree_structure = {}
        self.num_nodes = 0
        self.idx_leaf_nodes = []
        self.idx_prunable_split_nodes = []
        self.tree_id = tree_id

    def __getitem__(self, index):
        return self.get_node(index)

    def __setitem__(self, index, node):
        self.set_node(index, node)

    def __delitem__(self, index):
        self.delete_node(index)

    def __iter__(self):
        return iter(self.tree_structure.values())

    def __eq__(self, other):
        # The idx_leaf_nodes and idx_prunable_split_nodes are transformed to sets to correctly check for equivalence
        # in case the values are not ordered in the same way.
        return (
            self.tree_structure == other.tree_structure
            and self.num_nodes == other.num_nodes
            and set(self.idx_leaf_nodes) == set(other.idx_leaf_nodes)
            and set(self.idx_prunable_split_nodes) == set(other.idx_prunable_split_nodes)
            and self.tree_id == other.tree_id
        )

    def __hash__(self):
        # Method added to create a set of trees.
        return 0

    def __len__(self):
        return len(self.tree_structure)

    def __repr__(self):
        return "Tree(num_nodes={})".format(self.num_nodes)

    def __str__(self):
        lines = self._build_tree_string(index=0, show_index=False, delimiter="-")[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def _build_tree_string(self, index, show_index=False, delimiter="-"):
        """Recursively walk down the binary tree and build a pretty-print string.

        In each recursive call, a "box" of characters visually representing the
        current (sub)tree is constructed line by line. Each line is padded with
        whitespaces to ensure all lines in the box have the same length. Then the
        box, its width, and start-end positions of its root node value repr string
        (required for drawing branches) are sent up to the parent call. The parent
        call then combines its left and right sub-boxes to build a larger box etc.
        """
        if index not in self.tree_structure:
            return [], 0, 0, 0

        line1 = []
        line2 = []
        current_node = self.get_node(index)
        if show_index:
            node_repr = "{}{}{}".format(index, delimiter, str(current_node))
        else:
            node_repr = str(current_node)

        new_root_width = gap_size = len(node_repr)

        left_child = current_node.get_idx_left_child()
        right_child = current_node.get_idx_right_child()

        # Get the left and right sub-boxes, their widths, and root repr positions
        l_box, l_box_width, l_root_start, l_root_end = self._build_tree_string(
            left_child, show_index, delimiter
        )
        r_box, r_box_width, r_root_start, r_root_end = self._build_tree_string(
            right_child, show_index, delimiter
        )

        # Draw the branch connecting the current root node to the left sub-box
        # Pad the line with whitespaces where necessary
        if l_box_width > 0:
            l_root = (l_root_start + l_root_end) // 2 + 1
            line1.append(" " * (l_root + 1))
            line1.append("_" * (l_box_width - l_root))
            line2.append(" " * l_root + "/")
            line2.append(" " * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the current root node
        line1.append(node_repr)
        line2.append(" " * new_root_width)

        # Draw the branch connecting the current root node to the right sub-box
        # Pad the line with whitespaces where necessary
        if r_box_width > 0:
            r_root = (r_root_start + r_root_end) // 2
            line1.append("_" * r_root)
            line1.append(" " * (r_box_width - r_root + 1))
            line2.append(" " * r_root + "\\")
            line2.append(" " * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right sub-boxes with the branches drawn above
        gap = " " * gap_size
        new_box = ["".join(line1), "".join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else " " * l_box_width
            r_line = r_box[i] if i < len(r_box) else " " * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root repr positions
        return new_box, len(new_box[0]), new_root_start, new_root_end

    def copy(self):
        return deepcopy(self)

    def get_node(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError("Node index must be a non-negative int")
        if index not in self.tree_structure:
            raise TreeStructureError("Node missing at index {}".format(index))
        return self.tree_structure[index]

    def set_node(self, index, node):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError("Node index must be a non-negative int")
        if not isinstance(node, SplitNode) and not isinstance(node, LeafNode):
            raise TreeStructureError("Node class must be SplitNode or LeafNode")
        if index in self.tree_structure:
            raise TreeStructureError("Node index already exist in tree")
        if self.num_nodes == 0 and index != 0:
            raise TreeStructureError("Root node must have index zero")
        parent_index = node.get_idx_parent_node()
        if self.num_nodes != 0 and parent_index not in self.tree_structure:
            raise TreeStructureError("Invalid index, node must have a parent node")
        if self.num_nodes != 0 and not isinstance(self.get_node(parent_index), SplitNode):
            raise TreeStructureError("Parent node must be of class SplitNode")
        if index != node.index:
            raise TreeStructureError("Node must have same index as tree index")
        self.tree_structure[index] = node
        self.num_nodes += 1
        if isinstance(node, LeafNode):
            self.idx_leaf_nodes.append(index)

    def delete_node(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError("Node index must be a non-negative int")
        if index not in self.tree_structure:
            raise TreeStructureError("Node missing at index {}".format(index))
        current_node = self.get_node(index)
        left_child_idx = current_node.get_idx_left_child()
        right_child_idx = current_node.get_idx_right_child()
        if left_child_idx in self.tree_structure or right_child_idx in self.tree_structure:
            raise TreeStructureError("Invalid removal of node, leaving two orphans nodes")
        if isinstance(current_node, LeafNode):
            self.idx_leaf_nodes.remove(index)
        del self.tree_structure[index]
        self.num_nodes -= 1

    def make_digraph(self, name="Tree"):
        """Make graphviz Digraph of the tree.

        Parameters
        ----------
        name : str
            Name used for the Digraph. Useful to differentiate digraph of trees.

        Returns
        -------
        graphviz.Digraph
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "This function requires the python library graphviz, along with binaries. "
                "The easiest way to install all of this is by running\n\n"
                "\tconda install -c conda-forge python-graphviz"
            )
        graph = graphviz.Digraph(name)
        graph = self._digraph_tree_traversal(0, graph)
        return graph

    def _digraph_tree_traversal(self, index, graph):
        if index not in self.tree_structure.keys():
            return graph
        current_node = self.get_node(index)
        style = ""
        if isinstance(current_node, SplitNode):
            shape = "box"
            if index in self.idx_prunable_split_nodes:
                style = "filled"
        else:
            shape = "ellipse"
        node_name = "{}_{}".format(graph.name, index)
        graph.node(name=node_name, label=str(current_node), shape=shape, style=style)

        parent_index = current_node.get_idx_parent_node()
        if parent_index in self.tree_structure:
            tail_name = "{}_{}".format(graph.name, parent_index)
            head_name = "{}_{}".format(graph.name, index)
            if current_node.is_left_child():
                graph.edge(tail_name=tail_name, head_name=head_name, label="T")
            else:
                graph.edge(tail_name=tail_name, head_name=head_name, label="F")

        left_child = current_node.get_idx_left_child()
        right_child = current_node.get_idx_right_child()
        graph = self._digraph_tree_traversal(left_child, graph)
        graph = self._digraph_tree_traversal(right_child, graph)

        return graph

    def predict_output(self, num_observations):
        output = np.zeros(num_observations)
        for node_index in self.idx_leaf_nodes:
            current_node = self.get_node(node_index)
            output[current_node.idx_data_points] = current_node.value
        return output

    def out_of_sample_predict(self, x):
        """
        Predict output of tree for an unobserved point.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        float
            Value of the leaf value where the unobserved point lies.
        """
        leaf_node = self._traverse_tree(x=x, node_index=0)
        return leaf_node.value

    def _traverse_tree(self, x, node_index=0):
        """
        Traverse the tree starting from a particular node given an unobserved point.

        Parameters
        ----------
        x : np.ndarray
        node_index : int

        Returns
        -------
        LeafNode
        """
        current_node = self.get_node(node_index)
        if isinstance(current_node, SplitNode):
            if current_node.evaluate_splitting_rule(x):
                left_child = current_node.get_idx_left_child()
                final_node = self._traverse_tree(x, left_child)
            else:
                right_child = current_node.get_idx_right_child()
                final_node = self._traverse_tree(x, right_child)
        else:
            final_node = current_node
        return final_node

    def grow_tree(self, index_leaf_node, new_split_node, new_left_node, new_right_node):
        """
        Grow the tree from a particular node.

        Parameters
        ----------
        index_leaf_node : int
        new_split_node : SplitNode
        new_left_node : LeafNode
        new_right_node : LeafNode
        """
        current_node = self.get_node(index_leaf_node)
        if not isinstance(current_node, LeafNode):
            raise TreeStructureError("The tree grows from the leaves")
        if not isinstance(new_split_node, SplitNode):
            raise TreeStructureError("The node that replaces the leaf node must be SplitNode")
        if not isinstance(new_left_node, LeafNode) or not isinstance(new_right_node, LeafNode):
            raise TreeStructureError("The new leaves must be LeafNode")

        self.delete_node(index_leaf_node)
        self.set_node(index_leaf_node, new_split_node)
        self.set_node(new_left_node.index, new_left_node)
        self.set_node(new_right_node.index, new_right_node)

        # The new SplitNode is a prunable node since it has both children.
        self.idx_prunable_split_nodes.append(index_leaf_node)
        # If the parent of the node from which the tree is growing was a prunable node,
        # remove from the list since one of its children is a SplitNode now
        parent_index = current_node.get_idx_parent_node()
        if parent_index in self.idx_prunable_split_nodes:
            self.idx_prunable_split_nodes.remove(parent_index)

    @staticmethod
    def init_tree(tree_id, leaf_node_value, idx_data_points):
        """

        Parameters
        ----------
        tree_id
        leaf_node_value
        idx_data_points

        Returns
        -------

        """
        new_tree = Tree(tree_id)
        new_tree[0] = LeafNode(index=0, value=leaf_node_value, idx_data_points=idx_data_points)
        return new_tree

    def get_tree_depth(self):
        """

        Returns
        -------

        """
        max_depth = 0
        for idx_leaf_node in self.idx_leaf_nodes:
            leaf_node = self.get_node(idx_leaf_node)
            if max_depth < leaf_node.depth:
                max_depth = leaf_node.depth
        return max_depth


class BaseNode:
    def __init__(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeNodeError("Node index must be a non-negative int")
        self.index = index
        self.depth = int(math.floor(math.log(index + 1, 2)))

    def __eq__(self, other):
        return self.index == other.index and self.depth == other.depth

    def get_idx_parent_node(self):
        return (self.index - 1) // 2

    def get_idx_left_child(self):
        return self.index * 2 + 1

    def get_idx_right_child(self):
        return self.get_idx_left_child() + 1

    def is_left_child(self):
        return bool(self.index % 2)

    def get_idx_sibling(self):
        return (self.index + 1) if self.is_left_child() else (self.index - 1)


class SplitNode(BaseNode):
    def __init__(self, index, idx_split_variable, split_value):
        super().__init__(index)

        if not isinstance(idx_split_variable, int) or idx_split_variable < 0:
            raise TreeNodeError("Index of split variable must be a non-negative int")
        if not isinstance(split_value, float):
            raise TreeNodeError("Node split value type must be float")

        self.idx_split_variable = idx_split_variable
        self.split_value = split_value

    def __repr__(self):
        return "SplitNode(index={}, idx_split_variable={}, split_value={})".format(
            self.index, self.idx_split_variable, self.split_value
        )

    def __str__(self):
        return "x[{}] <= {}".format(self.idx_split_variable, self.split_value)

    def __eq__(self, other):
        if isinstance(other, SplitNode):
            return (
                super().__eq__(other)
                and self.idx_split_variable == other.idx_split_variable
                and self.split_value == other.split_value
            )
        else:
            return NotImplemented

    def evaluate_splitting_rule(self, x):
        if x is np.NaN:
            return False
        else:
            return x[self.idx_split_variable] <= self.split_value

    def prior_log_probability_node(self, alpha, beta):
        """
        Calculate the log probability of the node being a SplitNode.
        Taken from equation 7 in [Chipman2010].

        Parameters
        ----------
        alpha : float
        beta : float

        Returns
        -------
        float

        References
        ----------
        .. [Chipman2010] Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian
            additive regression trees. The Annals of Applied Statistics, 4(1), 266-298.,
            `link <https://projecteuclid.org/download/pdfview_1/euclid.aoas/1273584455>`__
        """
        return np.log(alpha * np.power(1.0 + self.depth, -beta))


class LeafNode(BaseNode):
    def __init__(self, index, value, idx_data_points):
        super().__init__(index)
        if not isinstance(value, float):
            raise TreeNodeError("Leaf node value type must be float")
        if (
            not isinstance(idx_data_points, np.ndarray)
            or idx_data_points.dtype.type is not np.int32
        ):
            raise TreeNodeError("Index of data points must be a numpy.ndarray of np.int32")
        if len(idx_data_points) == 0:
            raise TreeNodeError("Index of data points can not be empty")
        self.value = value
        self.idx_data_points = idx_data_points

    def __repr__(self):
        return "LeafNode(index={}, value={}, len(idx_data_points)={})".format(
            self.index, self.value, len(self.idx_data_points)
        )

    def __str__(self):
        return "{}".format(self.value)

    def __eq__(self, other):
        if isinstance(other, LeafNode):
            return (
                super().__eq__(other)
                and self.value == other.value
                and np.array_equal(self.idx_data_points, other.idx_data_points)
            )
        else:
            return NotImplemented

    def prior_log_probability_node(self, alpha, beta):
        """
        Calculate the log probability of the node being a LeafNode.
        Taken from equation 7 in [Chipman2010].

        Parameters
        ----------
        alpha : float
        beta : float

        Returns
        -------
        float

        References
        ----------
        .. [Chipman2010] Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian
            additive regression trees. The Annals of Applied Statistics, 4(1), 266-298.,
            `link <https://projecteuclid.org/download/pdfview_1/euclid.aoas/1273584455>`__
        """
        return np.log(1.0 - alpha * np.power(1.0 + self.depth, -beta))
