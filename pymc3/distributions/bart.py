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

import numpy as np

from pandas import DataFrame, Series
from scipy.special import expit

from pymc3.distributions.distribution import NoDistribution
from pymc3.distributions.tree import LeafNode, SplitNode, Tree

__all__ = ["BART"]


class BaseBART(NoDistribution):
    def __init__(
        self,
        X,
        Y,
        m=50,
        alpha=0.25,
        split_prior=None,
        inv_link=None,
        jitter=False,
        *args,
        **kwargs,
    ):

        self.jitter = jitter
        self.X, self.Y, self.missing_data = self.preprocess_XY(X, Y)

        super().__init__(shape=X.shape[0], dtype="float64", testval=self.Y.mean(), *args, **kwargs)

        if self.X.ndim != 2:
            raise ValueError("The design matrix X must have two dimensions")

        if self.Y.ndim != 1:
            raise ValueError("The response matrix Y must have one dimension")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                "The design matrix X and the response matrix Y must have the same number of elements"
            )
        if not isinstance(m, int):
            raise ValueError("The number of trees m type must be int")
        if m < 1:
            raise ValueError("The number of trees m must be greater than zero")

        if alpha <= 0 or 1 <= alpha:
            raise ValueError(
                "The value for the alpha parameter for the tree structure "
                "must be in the interval (0, 1)"
            )
        self.m = m
        self.alpha = alpha

        if inv_link is None:
            self.inv_link = self.link = lambda x: x
        elif isinstance(inv_link, str):
            if inv_link == "logistic":
                self.inv_link = expit
                # The link function is just a rough approximation in order to allow the PGBART
                # sampler to propose reasonable values for the leaf nodes. The regularizing term
                # 2 * self.m ** 0.5 is inspired by Chipman's DOI: 10.1214/09-AOAS285
                self.link = lambda x: (x - 0.5) * 2 * self.m ** 0.5
            elif inv_link == "exp":
                self.inv_link = np.exp
                self.link = np.log
                self.Y[self.Y == 0] += 0.0001
            else:
                raise ValueError("Accepted strings are 'logistic' or 'exp'")
        else:
            self.inv_link, self.link = inv_link

        self.init_mean = self.link(self.Y.mean())
        self.Y_un = self.link(self.Y)

        self.num_observations = X.shape[0]
        self.num_variates = X.shape[1]
        self.available_predictors = list(range(self.num_variates))
        self.ssv = SampleSplittingVariable(split_prior, self.num_variates)
        self.initial_value_leaf_nodes = self.init_mean / self.m
        self.trees = self.init_list_of_trees()
        self.all_trees = []
        self.mean = fast_mean()
        self.prior_prob_leaf_node = compute_prior_probability(alpha)

    def preprocess_XY(self, X, Y):
        if isinstance(Y, (Series, DataFrame)):
            Y = Y.to_numpy()
        if isinstance(X, (Series, DataFrame)):
            X = X.to_numpy()
        missing_data = np.any(np.isnan(X))
        if self.jitter:
            X = np.random.normal(X, np.nanstd(X, 0) / 100000)
        Y = Y.astype(float)
        return X, Y, missing_data

    def init_list_of_trees(self):
        initial_value_leaf_nodes = self.initial_value_leaf_nodes
        initial_idx_data_points_leaf_nodes = np.array(range(self.num_observations), dtype="int32")
        list_of_trees = []
        for i in range(self.m):
            new_tree = Tree.init_tree(
                tree_id=i,
                leaf_node_value=initial_value_leaf_nodes,
                idx_data_points=initial_idx_data_points_leaf_nodes,
            )
            list_of_trees.append(new_tree)
        # Diff trick to speed computation of residuals. From Section 3.1 of Kapelner, A and Bleich, J.
        # bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
        # The sum_trees_output will contain the sum of the predicted output for all trees.
        # When R_j is needed we subtract the current predicted output for tree T_j.
        self.sum_trees_output = np.full_like(self.Y, self.init_mean)

        return list_of_trees

    def __iter__(self):
        return iter(self.trees)

    def __repr_latex(self):
        raise NotImplementedError

    def get_available_splitting_rules(self, idx_data_points_split_node, idx_split_variable):
        x_j = self.X[idx_data_points_split_node, idx_split_variable]
        if self.missing_data:
            x_j = x_j[~np.isnan(x_j)]
        values = np.unique(x_j)
        # The last value is never available as it would leave the right subtree empty.
        return values[:-1]

    def grow_tree(self, tree, index_leaf_node):
        current_node = tree.get_node(index_leaf_node)

        index_selected_predictor = self.ssv.rvs()
        selected_predictor = self.available_predictors[index_selected_predictor]
        available_splitting_rules = self.get_available_splitting_rules(
            current_node.idx_data_points, selected_predictor
        )
        # This can be unsuccessful when there are not available splitting rules
        if available_splitting_rules.size == 0:
            return False, None

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

        return True, index_selected_predictor

    def get_new_idx_data_points(self, current_split_node, idx_data_points):
        idx_split_variable = current_split_node.idx_split_variable
        split_value = current_split_node.split_value

        left_idx = self.X[idx_data_points, idx_split_variable] <= split_value
        left_node_idx_data_points = idx_data_points[left_idx]
        right_node_idx_data_points = idx_data_points[~left_idx]

        return left_node_idx_data_points, right_node_idx_data_points

    def get_residuals(self):
        """Compute the residuals."""
        R_j = self.Y_un - self.sum_trees_output
        return R_j

    def draw_leaf_value(self, idx_data_points):
        """Draw the residual mean."""
        R_j = self.get_residuals()[idx_data_points]
        draw = self.mean(R_j)
        return draw

    def predict(self, X_new):
        """Compute out of sample predictions evaluated at X_new"""
        trees = self.all_trees
        num_observations = X_new.shape[0]
        pred = np.zeros((len(trees), num_observations))
        for draw, trees_to_sum in enumerate(trees):
            new_Y = np.zeros(num_observations)
            for tree in trees_to_sum:
                new_Y += [tree.predict_out_of_sample(x) for x in X_new]
            pred[draw] = new_Y
        return self.inv_link(pred)


def compute_prior_probability(alpha):
    """
    Calculate the probability of the node being a LeafNode (1 - p(being SplitNode)).
    Taken from equation 19 in [Rockova2018].

    Parameters
    ----------
    alpha : float

    Returns
    -------
    list with probabilities for leaf nodes

    References
    ----------
    .. [Rockova2018] Veronika Rockova, Enakshi Saha (2018). On the theory of BART.
    arXiv, `link <https://arxiv.org/abs/1810.00787>`__
    """
    prior_leaf_prob = [0]
    depth = 1
    while prior_leaf_prob[-1] < 1:
        prior_leaf_prob.append(1 - alpha ** depth)
        depth += 1
    return prior_leaf_prob


def fast_mean():
    """If available use Numba to speed up the computation of the mean."""
    try:
        from numba import jit
    except ImportError:
        return np.mean

    @jit
    def mean(a):
        count = a.shape[0]
        suma = 0
        for i in range(count):
            suma += a[i]
        return suma / count

    return mean


def discrete_uniform_sampler(upper_value):
    """Draw from the uniform distribution with bounds [0, upper_value)."""
    return int(np.random.random() * upper_value)


class SampleSplittingVariable:
    def __init__(self, prior, num_variates):
        self.prior = prior
        self.num_variates = num_variates

        if self.prior is not None:
            self.prior = np.asarray(self.prior)
            self.prior = self.prior / self.prior.sum()
            if self.prior.size != self.num_variates:
                raise ValueError(
                    f"The size of split_prior ({self.prior.size}) should be the "
                    f"same as the number of covariates ({self.num_variates})"
                )
            self.enu = list(enumerate(np.cumsum(self.prior)))

    def rvs(self):
        if self.prior is None:
            return int(np.random.random() * self.num_variates)
        else:
            r = np.random.random()
            for i, v in self.enu:
                if r <= v:
                    return i


class BART(BaseBART):
    """
    BART distribution.

    Distribution representing a sum over trees

    Parameters
    ----------
    X : array-like
        The design matrix.
    Y : array-like
        The response vector.
    m : int
        Number of trees
    alpha : float
        Control the prior probability over the depth of the trees. Even when it can takes values in
        the interval (0, 1), it is recommended to be in the interval (0, 0.5].
    split_prior : array-like
        Each element of split_prior should be in the [0, 1] interval and the elements should sum
        to 1. Otherwise they will be normalized.
        Defaults to None, all variable have the same a prior probability
    inv_link : str or tuple of functions
        Inverse link function defaults to None, i.e. the identity function. Accepted strings are
        ``logistic`` or ``exp``.
    jitter : bool
        Whether to jitter the X values or not. Defaults to False. When values of X are repeated,
        jittering X has the effect of increasing the number of effective spliting variables,
        otherwise it does not have any effect.
    """

    def __init__(self, X, Y, m=50, alpha=0.25, split_prior=None, inv_link=None, jitter=False):
        super().__init__(X, Y, m, alpha, split_prior, inv_link)

    def _str_repr(self, name=None, dist=None, formatting="plain"):
        if dist is None:
            dist = self
        X = (type(self.X),)
        Y = (type(self.Y),)
        alpha = self.alpha
        m = self.m

        if "latex" in formatting:
            return f"$\\text{{{name}}} \\sim  \\text{{BART}}(\\text{{alpha = }}\\text{{{alpha}}}, \\text{{m = }}\\text{{{m}}})$"
        else:
            return f"{name} ~ BART(alpha = {alpha}, m = {m})"
