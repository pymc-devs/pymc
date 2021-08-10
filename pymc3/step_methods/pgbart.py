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

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from pandas import DataFrame, Series

from aesara import function as aesara_function

from pymc3.aesaraf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc3.distributions.bart import BARTRV
from pymc3.distributions.tree import LeafNode, SplitNode, Tree
from pymc3.model import modelcontext
from pymc3.step_methods.arraystep import ArrayStepShared, Competence
from pymc3.blocking import RaveledVars


_log = logging.getLogger("pymc3")


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step

    Parameters
    ----------
    vars: list
        List of variables for sampler
    num_particles : int
        Number of particles for the conditional SMC sampler. Defaults to 10
    max_stages : int
        Maximum number of iterations of the conditional SMC sampler. Defaults to 100.
    chunk = int
        Number of trees fitted per step. Defaults to  "auto", which is the 10% of the `m` trees.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    References
    ----------
    .. [Lakshminarayanan2015] Lakshminarayanan, B. and Roy, D.M. and Teh, Y. W., (2015),
        Particle Gibbs for Bayesian Additive Regression Trees.
        ArviX, `link <https://arxiv.org/abs/1502.04622>`__
    """

    name = "bartsampler"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{"variable_inclusion": np.ndarray}]

    def __init__(self, vars=None, num_particles=10, max_stages=5000, chunk="auto", model=None):
        _log.warning("BART is experimental. Use with caution.")
        model = modelcontext(model)
        initial_values = model.initial_point
        value_bart = inputvars(vars)[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        self.X, self.Y, self.missing_data = preprocess_XY(self.bart.X, self.bart.Y)
        self.m = self.bart.m
        self.alpha = self.bart.alpha
        self.k = self.bart.k
        self.split_prior = self.bart.split_prior

        self.init_mean = self.Y.mean()
        # if data is binary
        if np.all(np.unique(self.Y) == [0, 1]):
            self.mu_std = 6 / (self.k * self.m ** 0.5)
        # maybe we need to check for count data
        else:
            self.mu_std = self.Y.std() / (self.k * self.m ** 0.5)

        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))
        self.ssv = SampleSplittingVariable(self.split_prior, self.num_variates)
        self.initial_value_leaf_nodes = self.init_mean / self.m

        self.trees, sum_trees_output = init_list_of_trees(
            self.initial_value_leaf_nodes, self.num_observations, self.m, self.Y, self.init_mean
        )
        self.mean = fast_mean()
        self.normal = NormalSampler()
        self.prior_prob_leaf_node = compute_prior_probability(self.alpha)

        self.tune = True
        self.idx = 0
        self.iter = 0
        self.sum_trees = []
        self.chunk = chunk

        if self.chunk == "auto":
            self.chunk = max(1, int(self.m * 0.1))
        self.num_particles = num_particles
        self.log_num_particles = np.log(num_particles)
        self.indices = list(range(1, num_particles))
        self.max_stages = max_stages

        shared = make_shared_replacements(initial_values, vars, model)
        self.likelihood_logp = logp(initial_values, [model.datalogpt], vars, shared)
        self.init_leaf_nodes = self.initial_value_leaf_nodes
        self.init_likelihood = self.likelihood_logp(sum_trees_output)
        self.init_log_weight = self.init_likelihood - self.log_num_particles
        self.old_trees_particles_list = []
        for i in range(self.m):
            p = ParticleTree(
                self.trees[i],
                self.prior_prob_leaf_node,
                self.init_log_weight,
                self.init_likelihood,
            )
            self.old_trees_particles_list.append(p)
        super().__init__(vars, shared)

    def astep(self, q: RaveledVars) -> Tuple[RaveledVars, List[Dict[str, Any]]]:
        point_map_info = q.point_map_info
        sum_trees_output = q.data

        variable_inclusion = np.zeros(self.num_variates, dtype="int")

        if self.idx == self.m:
            self.idx = 0

        for idx in range(self.idx, self.idx + self.chunk):
            if idx >= self.m:
                break
            self.idx += 1
            tree = self.trees[idx]
            sum_trees_output_noi = sum_trees_output - tree.predict_output()
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree
            particles = self.init_particles(tree.tree_id)

            for t in range(1, self.max_stages):
                # Get old particle at stage t
                particles[0] = self.get_old_tree_particle(tree.tree_id, t)
                # sample each particle (try to grow each tree)
                for c in range(1, self.num_particles):
                    particles[c].sample_tree_sequential(
                        self.ssv,
                        self.available_predictors,
                        self.X,
                        self.missing_data,
                        sum_trees_output,
                        self.mean,
                        self.m,
                        self.normal,
                        self.mu_std,
                    )
                # Update weights. Since the prior is used as the proposal,the weights
                # are updated additively as the ratio of the new and old log_likelihoods
                for p in particles:
                    new_likelihood = self.likelihood_logp(
                        sum_trees_output_noi + p.tree.predict_output()
                    )
                    p.log_weight += new_likelihood - p.old_likelihood_logp
                    p.old_likelihood_logp = new_likelihood

                # Normalize weights
                W_t, normalized_weights = self.normalize(particles)

                # Set the new weights
                for p in particles:
                    p.log_weight = W_t

                # Resample all but first particle
                re_n_w = normalized_weights[1:] / normalized_weights[1:].sum()

                new_indices = np.random.choice(self.indices, size=len(self.indices), p=re_n_w)
                particles[1:] = particles[new_indices]

                # Check if particles can keep growing, otherwise stop iterating
                non_available_nodes_for_expansion = np.ones(self.num_particles - 1)
                for c in range(1, self.num_particles):
                    if len(particles[c].expansion_nodes) != 0:
                        non_available_nodes_for_expansion[c - 1] = 0
                if np.all(non_available_nodes_for_expansion):
                    break

            # Get the new tree and update
            new_tree = np.random.choice(particles, p=normalized_weights)
            self.old_trees_particles_list[tree.tree_id] = new_tree
            self.trees[idx] = new_tree.tree
            sum_trees_output = sum_trees_output_noi + new_tree.tree.predict_output()

            if not self.tune:
                self.iter += 1
                self.sum_trees.append(new_tree.tree)
                if not self.iter % self.m:
                    # XXX update the all_trees variable in BARTRV to be used in the rng_fn method
                    # this fails for chains > 1 as the variable is not shared between proccesses
                    self.bart.all_trees.append(self.sum_trees)
                    self.sum_trees = []
                for index in new_tree.used_variates:
                    variable_inclusion[index] += 1

        stats = {"variable_inclusion": variable_inclusion}
        sum_trees_output = RaveledVars(sum_trees_output, point_map_info)
        return sum_trees_output, [stats]

    @staticmethod
    def competence(var, has_grad):
        """
        PGBART is only suitable for BART distributions
        """
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    def normalize(self, particles):
        """
        Use logsumexp trick to get W_t and softmax to get normalized_weights
        """
        log_w = np.array([p.log_weight for p in particles])
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        w_ = np.exp(log_w_)
        w_sum = w_.sum()
        W_t = log_w_max + np.log(w_sum) - self.log_num_particles
        normalized_weights = w_ / w_sum
        # stabilize weights to avoid assigning exactly zero probability to a particle
        normalized_weights += 1e-12

        return W_t, normalized_weights

    def get_old_tree_particle(self, tree_id, t):
        old_tree_particle = self.old_trees_particles_list[tree_id]
        old_tree_particle.set_particle_to_step(t)
        return old_tree_particle

    def init_particles(self, tree_id):
        """
        Initialize particles
        """
        prev_tree = self.get_old_tree_particle(tree_id, 0)
        likelihood = self.likelihood_logp(prev_tree.tree.predict_output())
        prev_tree.old_likelihood_logp = likelihood
        prev_tree.log_weight = likelihood - self.log_num_particles
        particles = [prev_tree]

        initial_idx_data_points_leaf_nodes = np.arange(self.num_observations, dtype="int32")
        new_tree = Tree.init_tree(
            tree_id=tree_id,
            leaf_node_value=self.init_leaf_nodes,
            idx_data_points=initial_idx_data_points_leaf_nodes,
        )

        prior_prob = self.prior_prob_leaf_node
        for _ in range(1, self.num_particles):
            particles.append(
                ParticleTree(new_tree, prior_prob, self.init_log_weight, self.init_likelihood)
            )

        return np.array(particles)

    def resample(self, particles, weights):
        """
        Resample a set of particles given its weights
        """
        particles = np.random.choice(particles, size=len(particles), p=weights)
        return particles


class ParticleTree:
    """
    Particle tree
    """

    def __init__(self, tree, prior_prob_leaf_node, log_weight=0, likelihood=0):
        self.tree = tree.copy()  # keeps the tree that we care at the moment
        self.expansion_nodes = [0]
        self.tree_history = [self.tree]
        self.expansion_nodes_history = [self.expansion_nodes]
        self.log_weight = log_weight
        self.prior_prob_leaf_node = prior_prob_leaf_node
        self.old_likelihood_logp = likelihood
        self.used_variates = []

    def sample_tree_sequential(
        self, ssv, available_predictors, X, missing_data, sum_trees_output, mean, m, normal, mu_std
    ):
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            prob_leaf = self.prior_prob_leaf_node[self.tree[index_leaf_node].depth]

            if prob_leaf < np.random.random():
                grow_successful, index_selected_predictor = grow_tree(
                    self.tree,
                    index_leaf_node,
                    ssv,
                    available_predictors,
                    X,
                    missing_data,
                    sum_trees_output,
                    mean,
                    m,
                    normal,
                    mu_std,
                )
                if grow_successful:
                    # Add new leaf nodes indexes
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
                    self.used_variates.append(index_selected_predictor)

            self.tree_history.append(self.tree)
            self.expansion_nodes_history.append(self.expansion_nodes)

    def set_particle_to_step(self, t):
        if len(self.tree_history) <= t:
            self.tree = self.tree_history[-1]
            self.expansion_nodes = self.expansion_nodes_history[-1]
        else:
            self.tree = self.tree_history[t]
            self.expansion_nodes = self.expansion_nodes_history[t]


def preprocess_XY(X, Y):
    if isinstance(Y, (Series, DataFrame)):
        Y = Y.to_numpy()
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()
    missing_data = np.any(np.isnan(X))
    Y = Y.astype(float)
    return X, Y, missing_data


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


def init_list_of_trees(initial_value_leaf_nodes, num_observations, m, Y, init_mean):
    initial_value_leaf_nodes = initial_value_leaf_nodes
    initial_idx_data_points_leaf_nodes = np.array(range(num_observations), dtype="int32")
    list_of_trees = []
    for i in range(m):
        new_tree = Tree.init_tree(
            tree_id=i,
            leaf_node_value=initial_value_leaf_nodes,
            idx_data_points=initial_idx_data_points_leaf_nodes,
        )
        list_of_trees.append(new_tree)
    sum_trees_output = np.full_like(Y, init_mean)

    return list_of_trees, sum_trees_output


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


def grow_tree(
    tree,
    index_leaf_node,
    ssv,
    available_predictors,
    X,
    missing_data,
    sum_trees_output,
    mean,
    m,
    normal,
    mu_std,
):
    current_node = tree.get_node(index_leaf_node)

    index_selected_predictor = ssv.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    available_splitting_rules = get_available_splitting_rules(
        X[current_node.idx_data_points, selected_predictor],
        missing_data,
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

    left_node_idx_data_points, right_node_idx_data_points = get_new_idx_data_points(
        new_split_node, current_node.idx_data_points, X
    )

    left_node_value = draw_leaf_value(
        sum_trees_output[left_node_idx_data_points], mean, m, normal, mu_std
    )
    right_node_value = draw_leaf_value(
        sum_trees_output[right_node_idx_data_points], mean, m, normal, mu_std
    )

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


def get_available_splitting_rules(X_j, missing_data):
    if missing_data:
        X_j = X_j[~np.isnan(X_j)]
    values = np.unique(X_j)
    # The last value is never available as it would leave the right subtree empty.
    return values[:-1]


def get_new_idx_data_points(current_split_node, idx_data_points, X):
    idx_split_variable = current_split_node.idx_split_variable
    split_value = current_split_node.split_value

    left_idx = X[idx_data_points, idx_split_variable] <= split_value
    left_node_idx_data_points = idx_data_points[left_idx]
    right_node_idx_data_points = idx_data_points[~left_idx]

    return left_node_idx_data_points, right_node_idx_data_points


def draw_leaf_value(sum_trees_output_idx, mean, m, normal, mu_std):
    """Draw Gaussian distributed leaf values"""
    mu_mean = mean(sum_trees_output_idx) / m
    # add cache of precomputed normal values
    draw = normal.random() * mu_std + mu_mean
    return draw


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
    """Draw from the uniform distribution with bounds [0, upper_value).

    This is the same and np.random.randit(upper_value) but faster.
    """
    return int(np.random.random() * upper_value)


class NormalSampler:
    """
    Cache samples from a standard normal distribution
    """

    def __init__(self):
        self.size = 1000
        self.cache = []

    def random(self):
        if not self.cache:
            self.update()
        return self.cache.pop()

    def update(self):
        self.cache = np.random.normal(loc=0.0, scale=1, size=self.size).tolist()


def logp(point, out_vars, vars, shared):
    """Compile Aesara function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc3.Distribution` for the output variables
    vars: List
        containing :class:`pymc3.Distribution` for the input variables
    shared: List
        containing :class:`aesara.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    f = aesara_function([inarray0], out_list[0])
    f.trust_input = True
    return f
