import numpy as np
from .distribution import NoDistribution
from .tree import Tree, SplitNode, LeafNode

__all__ = ["BART"]


class BARTParamsError(Exception):
    """Base (catch-all) BART hyper parameters exception."""


class BaseBART(NoDistribution):
    def __init__(self, X, Y, m=200, alpha=0.25, *args, **kwargs):
        self.X = X
        self.Y = Y
        super().__init__(
            shape=X.shape[0], dtype="float64", testval=0, *args, **kwargs
        )  # FIXME dtype and testvalue are nonsensical

        if not isinstance(self.X, np.ndarray) or self.X.dtype.type is not np.float64:
            raise BARTParamsError(
                "The design matrix X type must be numpy.ndarray where every item"
                " type is numpy.float64"
            )
        if self.X.ndim != 2:
            raise BARTParamsError("The design matrix X must have two dimensions")
        # if not isinstance(self.Y, np.ndarray) or self.Y.dtype.type is not np.float64:
        #    raise BARTParamsError(
        #       "The response matrix Y type must be numpy.ndarray where every item"
        #        " type is numpy.float64"
        #    )
        if self.Y.ndim != 1:
            raise BARTParamsError("The response matrix Y must have one dimension")
        if self.X.shape[0] != self.Y.shape[0]:
            raise BARTParamsError(
                "The design matrix X and the response matrix Y must have the same number of elements"
            )
        if not isinstance(m, int):
            raise BARTParamsError("The number of trees m type must be int")
        if m < 1:
            raise BARTParamsError("The number of trees m must be greater than zero")

        if alpha <= 0 or 1 <= alpha:
            raise BARTParamsError(
                "The value for the alpha parameter for the tree structure "
                "must be in the interval (0, 1)"
            )

        self.num_observations = X.shape[0]
        self.number_variates = X.shape[1]
        self.m = m
        self.alpha = alpha
        self.trees = self.init_list_of_trees()

    def init_list_of_trees(self):
        initial_value_leaf_nodes = self.Y.mean() / self.m
        initial_idx_data_points_leaf_nodes = np.array(range(self.num_observations), dtype="int32")
        list_of_trees = []
        for i in range(self.m):
            new_tree = Tree.init_tree(
                tree_id=i,
                leaf_node_value=initial_value_leaf_nodes,
                idx_data_points=initial_idx_data_points_leaf_nodes,
            )
            list_of_trees.append(new_tree)
        # Diff trick to speed computation of residuals.
        # Taken from Section 3.1 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
        # The sum_trees_output will contain the sum of the predicted output for all trees.
        # When R_j is needed we subtract the current predicted output for tree T_j.
        self.sum_trees_output = np.full_like(self.Y, self.Y.mean())

        return list_of_trees

    def __iter__(self):
        return iter(self.trees)

    def __repr_latex(self):
        raise NotImplementedError

    def get_available_predictors(self, idx_data_points_split_node):
        possible_splitting_variables = []
        for j in range(self.number_variates):
            x_j = self.X[idx_data_points_split_node, j]
            x_j = x_j[~np.isnan(x_j)]
            for i in range(1, len(x_j)):
                if x_j[i - 1] != x_j[i]:
                    possible_splitting_variables.append(j)
                    break
        return possible_splitting_variables

    def get_available_splitting_rules(self, idx_data_points_split_node, idx_split_variable):
        x_j = self.X[idx_data_points_split_node, idx_split_variable]
        x_j = x_j[~np.isnan(x_j)]
        values, indices = np.unique(x_j, return_index=True)
        # The last value is not consider since if we choose it as the value of
        # the splitting rule assignment, it would leave the right subtree empty.
        return values[:-1], indices[:-1]

    def grow_tree(self, tree, index_leaf_node):
        # This can be unsuccessful when there are not available predictors
        successful_grow_tree = False
        current_node = tree.get_node(index_leaf_node)

        available_predictors = self.get_available_predictors(current_node.idx_data_points)

        if not available_predictors:
            return successful_grow_tree

        index_selected_predictor = discrete_uniform_sampler(len(available_predictors))
        selected_predictor = available_predictors[index_selected_predictor]

        available_splitting_rules, _ = self.get_available_splitting_rules(
            current_node.idx_data_points, selected_predictor
        )
        index_selected_splitting_rule = discrete_uniform_sampler(len(available_splitting_rules))
        selected_splitting_rule = available_splitting_rules[index_selected_splitting_rule]

        new_split_node = SplitNode(
            index=index_leaf_node,
            idx_split_variable=selected_predictor,
            split_value=selected_splitting_rule,
        )

        left_node_idx_data_points, right_node_idx_data_points = self.get_new_idx_data_points(
            new_split_node, current_node.idx_data_points
        )

        left_node_value = self.draw_leaf_value(left_node_idx_data_points)
        right_node_value = self.draw_leaf_value(right_node_idx_data_points)

        new_left_node = LeafNode(
            index=current_node.get_idx_left_child(),
            value=left_node_value,
            idx_data_points=left_node_idx_data_points,
        )
        new_right_node = LeafNode(
            index=current_node.get_idx_right_child(),
            value=right_node_value,
            idx_data_points=right_node_idx_data_points,
        )
        tree.grow_tree(index_leaf_node, new_split_node, new_left_node, new_right_node)
        successful_grow_tree = True

        return successful_grow_tree

    def get_new_idx_data_points(self, current_split_node, idx_data_points):
        idx_split_variable = current_split_node.idx_split_variable
        split_value = current_split_node.split_value

        left_idx = np.nonzero(self.X[idx_data_points, idx_split_variable] <= split_value)
        left_node_idx_data_points = idx_data_points[left_idx]
        right_idx = np.nonzero(~(self.X[idx_data_points, idx_split_variable] <= split_value))
        right_node_idx_data_points = idx_data_points[right_idx]

        return left_node_idx_data_points, right_node_idx_data_points

    def get_residuals_loo(self, tree):
        R_j = self.Y - (self.sum_trees_output - tree.predict_output(self.num_observations))
        return R_j

    def get_residuals(self):
        R_j = self.Y - self.sum_trees_output
        return R_j

    def draw_leaf_value(self, idx_data_points):
        # draw the residual mean
        R_j = self.get_residuals()[idx_data_points]
        draw = R_j.mean()
        return draw


def discrete_uniform_sampler(upper_value):
    return int(np.random.random() * upper_value)


class BART(BaseBART):
    """
    BART distribution.

    Distributon representing a sum over trees

    Parameters
    ----------
    X :
        The design matrix.
    Y :
        The response vector.
    m : int
        Number of trees
    alpha : float
        Control the prior probability over the depth of the trees. Must be in the interval (0, 1),
        altought it is recomenned to be between in the interval (0, 0.5].
    """

    def __init__(self, X, Y, m=200, alpha=0.25):
        super().__init__(X, Y, m, alpha)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        X = (type(self.X),)
        Y = (type(self.Y),)
        m = (self.m,)
        alpha = self.alpha
        m = self.m
        name = r"\text{%s}" % name
        return r"""${} \sim \text{{BART}}(\mathit{{alpha}}={},~\mathit{{m}}={})$""".format(
            name, alpha, m
        )
