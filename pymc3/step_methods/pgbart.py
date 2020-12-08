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

import numpy as np

from theano import function as theano_function

from pymc3.distributions import BART
from pymc3.distributions.tree import Tree
from pymc3.model import modelcontext
from pymc3.step_methods.arraystep import ArrayStepShared, Competence
from pymc3.theanof import inputvars, join_nonshared_inputs, make_shared_replacements

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
        _log.warning("The BART model is experimental. Use with caution.")
        model = modelcontext(model)
        vars = inputvars(vars)
        self.bart = vars[0].distribution

        self.tune = True
        self.idx = 0
        self.iter = 0
        self.sum_trees = []
        self.chunk = chunk

        if chunk == "auto":
            self.chunk = max(1, int(self.bart.m * 0.1))
        self.bart.chunk = self.chunk
        self.num_particles = num_particles
        self.log_num_particles = np.log(num_particles)
        self.indices = list(range(1, num_particles))
        self.max_stages = max_stages
        self.old_trees_particles_list = []
        for i in range(self.bart.m):
            p = ParticleTree(self.bart.trees[i], self.bart.prior_prob_leaf_node)
            self.old_trees_particles_list.append(p)

        shared = make_shared_replacements(vars, model)
        self.likelihood_logp = logp([model.datalogpt], vars, shared)
        super().__init__(vars, shared)

    def astep(self, _):
        bart = self.bart
        num_observations = bart.num_observations
        variable_inclusion = np.zeros(bart.num_variates, dtype="int")

        # For the tunning phase we restrict max_stages to a low number, otherwise it is almost sure
        # we will reach max_stages given that our first set of m trees is not good at all.
        # Can set max_stages as a function of the number of variables/dimensions?
        if self.tune:
            max_stages = 5
        else:
            max_stages = self.max_stages

        if self.idx == bart.m:
            self.idx = 0

        for idx in range(self.idx, self.idx + self.chunk):
            if idx >= bart.m:
                break
            self.idx += 1
            tree = bart.trees[idx]
            R_j = bart.get_residuals_loo(tree)
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree
            particles = self.init_particles(tree.tree_id, R_j, num_observations)

            for t in range(1, max_stages):
                # Get old particle at stage t
                particles[0] = self.get_old_tree_particle(tree.tree_id, t)
                # sample each particle (try to grow each tree)
                for c in range(1, self.num_particles):
                    particles[c].sample_tree_sequential(bart)
                # Update weights. Since the prior is used as the proposal,the weights
                # are updated additively as the ratio of the new and old log_likelihoods
                for p_idx, p in enumerate(particles):
                    new_likelihood = self.likelihood_logp(p.tree.predict_output(num_observations))
                    p.log_weight += new_likelihood - p.old_likelihood_logp
                    p.old_likelihood_logp = new_likelihood

                # Normalize weights
                W, normalized_weights = self.normalize(particles)

                # Resample all but first particle
                re_n_w = normalized_weights[1:] / normalized_weights[1:].sum()
                new_indices = np.random.choice(self.indices, size=len(self.indices), p=re_n_w)
                particles[1:] = particles[new_indices]

                # Set the new weights
                w_t = W - self.log_num_particles
                for p in particles:
                    p.log_weight = w_t

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
            bart.trees[idx] = new_tree.tree
            new_prediction = new_tree.tree.predict_output(num_observations)
            bart.sum_trees_output = bart.Y - R_j + new_prediction

            if not self.tune:
                self.iter += 1
                self.sum_trees.append(new_tree.tree)
                if not self.iter % bart.m:
                    bart.all_trees.append(self.sum_trees)
                    self.sum_trees = []
                for index in new_tree.used_variates:
                    variable_inclusion[index] += 1

        stats = {"variable_inclusion": variable_inclusion}

        return bart.sum_trees_output, [stats]

    @staticmethod
    def competence(var, has_grad):
        """
        PGBART is only suitable for BART distributions
        """
        if isinstance(var.distribution, BART):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    def normalize(self, particles):
        """
        Use logsumexp trick to get W and softmax to get normalized_weights
        """
        log_w = np.array([p.log_weight for p in particles])
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        w_ = np.exp(log_w_)
        w_sum = w_.sum()
        W = log_w_max + np.log(w_sum)
        normalized_weights = w_ / w_sum
        # stabilize weights to avoid assigning exactly zero probability to a particle
        normalized_weights += 1e-12

        return W, normalized_weights

    def get_old_tree_particle(self, tree_id, t):
        old_tree_particle = self.old_trees_particles_list[tree_id]
        old_tree_particle.set_particle_to_step(t)
        return old_tree_particle

    def init_particles(self, tree_id, R_j, num_observations):
        """
        Initialize particles
        """
        # The first particle is from the tree we are trying to replace
        prev_tree = self.get_old_tree_particle(tree_id, 0)
        likelihood = self.likelihood_logp(prev_tree.tree.predict_output(num_observations))
        prev_tree.old_likelihood_logp = likelihood
        prev_tree.log_weight = likelihood - self.log_num_particles
        particles = [prev_tree]

        # The rest of the particles are identically initialized
        initial_value_leaf_nodes = R_j.mean()
        initial_idx_data_points_leaf_nodes = np.arange(num_observations, dtype="int32")
        new_tree = Tree.init_tree(
            tree_id=tree_id,
            leaf_node_value=initial_value_leaf_nodes,
            idx_data_points=initial_idx_data_points_leaf_nodes,
        )
        likelihood_logp = self.likelihood_logp(new_tree.predict_output(num_observations))
        log_weight = likelihood_logp - self.log_num_particles
        for i in range(1, self.num_particles):
            particles.append(
                ParticleTree(new_tree, self.bart.prior_prob_leaf_node, log_weight, likelihood_logp)
            )

        return np.array(particles)

    def resample(self, particles, weights):
        """
        resample a set of particles given its weights
        """
        particles = np.random.choice(particles, size=len(particles), p=weights)
        return particles


class ParticleTree:
    """
    Particle tree
    """

    def __init__(self, tree, prior_prob_leaf_node, log_weight=0, likelihood=0):
        self.tree = tree.copy()  # keeps the tree that we care at the moment
        self.expansion_nodes = tree.idx_leaf_nodes.copy()  # This should be the array [0]
        self.tree_history = [self.tree]
        self.expansion_nodes_history = [self.expansion_nodes]
        self.log_weight = 0
        self.prior_prob_leaf_node = prior_prob_leaf_node
        self.old_likelihood_logp = likelihood
        self.used_variates = []

    def sample_tree_sequential(self, bart):
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            prob_leaf = self.prior_prob_leaf_node[self.tree[index_leaf_node].depth]

            if prob_leaf < np.random.random():
                grow_successful, index_selected_predictor = bart.grow_tree(
                    self.tree, index_leaf_node
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


def logp(out_vars, vars, shared):
    """Compile Theano function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc3.Distribution` for the output variables
    vars: List
        containing :class:`pymc3.Distribution` for the input variables
    shared: List
        containing :class:`theano.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(out_vars, vars, shared)
    f = theano_function([inarray0], out_list[0])
    f.trust_input = True
    return f
