#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

from copy import deepcopy

import numpy as np


class Tree:
    """Full binary tree
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

    def __init__(self, tree_id=0, num_observations=0):
        self.tree_structure = {}
        self.num_nodes = 0
        self.idx_leaf_nodes = []
        self.idx_prunable_split_nodes = []
        self.tree_id = tree_id
        self.num_observations = num_observations

    def __getitem__(self, index):
        return self.get_node(index)

    def __setitem__(self, index, node):
        self.set_node(index, node)

    def copy(self):
        return deepcopy(self)

    def get_node(self, index):
        return self.tree_structure[index]

    def set_node(self, index, node):
        self.tree_structure[index] = node
        self.num_nodes += 1
        if isinstance(node, LeafNode):
            self.idx_leaf_nodes.append(index)

    def delete_node(self, index):
        current_node = self.get_node(index)
        if isinstance(current_node, LeafNode):
            self.idx_leaf_nodes.remove(index)
        del self.tree_structure[index]
        self.num_nodes -= 1

    def predict_output(self):
        output = np.zeros(self.num_observations)
        for node_index in self.idx_leaf_nodes:
            current_node = self.get_node(node_index)
            output[current_node.idx_data_points] = current_node.value

        return output

    def predict_out_of_sample(self, x):
        """
        Predict output of tree for an unobserved point x.

        Parameters
        ----------
        x : numpy array

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
            if x[current_node.idx_split_variable] <= current_node.split_value:
                left_child = current_node.get_idx_left_child()
                current_node = self._traverse_tree(x, left_child)
            else:
                right_child = current_node.get_idx_right_child()
                current_node = self._traverse_tree(x, right_child)
        return current_node

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
        new_tree = Tree(tree_id, len(idx_data_points))
        new_tree[0] = LeafNode(index=0, value=leaf_node_value, idx_data_points=idx_data_points)
        return new_tree


class BaseNode:
    def __init__(self, index):
        self.index = index
        self.depth = int(math.floor(math.log(index + 1, 2)))

    def get_idx_parent_node(self):
        return (self.index - 1) // 2

    def get_idx_left_child(self):
        return self.index * 2 + 1

    def get_idx_right_child(self):
        return self.get_idx_left_child() + 1


class SplitNode(BaseNode):
    def __init__(self, index, idx_split_variable, split_value):
        super().__init__(index)

        self.idx_split_variable = idx_split_variable
        self.split_value = split_value


class LeafNode(BaseNode):
    def __init__(self, index, value, idx_data_points):
        super().__init__(index)
        self.value = value
        self.idx_data_points = idx_data_points
