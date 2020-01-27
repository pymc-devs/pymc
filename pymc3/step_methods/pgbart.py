import numpy as np

from .arraystep import ArrayStepShared, Competence
from ..distributions import BART
from ..distributions.tree import Tree
from ..model import modelcontext
from ..theanof import inputvars, make_shared_replacements, floatX
from ..smc.smc import logp_forw


class PGBART(ArrayStepShared):
    """
    """

    name = "bartsampler"
    default_blocked = False

    def __init__(self, vars=None, m=2, num_particles=10, max_stages=5000, model=None):
        model = modelcontext(model)
        vars = inputvars(vars)
        self.bart = vars[0].distribution

        self.num_particles = num_particles
        self.max_stages = max_stages
        self.previous_trees_particles_list = []
        for i in range(self.bart.m):
            p = Particle(self.bart.trees[i])
            self.previous_trees_particles_list.append(p)

        shared = make_shared_replacements(vars, model)
        self.likelihood_logp = logp_forw([model.datalogpt], vars, shared)
        super().__init__(vars, shared)

    def astep(self, q_0):
        # Step 4 of algorithm
        bart = self.bart
        num_observations = bart.num_observations
        likelihood_logp = self.likelihood_logp
        output = np.zeros((bart.m, num_observations))
        log_num_particles = np.log(self.num_particles)
        for idx, tree in enumerate(bart.trees):
            R_j = bart.get_residuals(tree)
            bart.Y_shared.set_value(R_j)
            list_of_particles = self.init_particles(tree.tree_id, R_j, bart.num_observations)
            # Step 5 of algorithm
            old_likelihoods = np.array(
                [
                    likelihood_logp(p.tree.predict_output(num_observations))
                    for p in list_of_particles
                ]
            )
            log_weights = np.array(old_likelihoods)

            list_of_particles[0] = self.get_previous_tree_particle(tree.tree_id, 0)
            log_weights[0] = likelihood_logp(
                list_of_particles[0].tree.predict_output(num_observations)
            )
            old_likelihoods[0] = log_weights[0]
            log_weights -= log_num_particles

            for p_idx, p in enumerate(list_of_particles):
                p.weight = log_weights[p_idx]

            for t in range(1, self.max_stages + 2):
                # Step 7 of algorithm
                list_of_particles[0] = self.get_previous_tree_particle(tree.tree_id, t)
                for c in range(
                    1, self.num_particles
                ):  # This should be embarrassingly parallelizable
                    # Step 9 of algorithm
                    list_of_particles[c].sample_tree_sequential(bart)
                # for c in range(self.num_particles):
                # list_of_particles[c].update_weight()

                # Step 12 of algorithm
                for p_idx, p in enumerate(list_of_particles):
                    new_likelihood = likelihood_logp(p.tree.predict_output(num_observations))
                    # p.weight is actually log_weight
                    # print(new_likelihood - old_likelihoods[p_idx])
                    p.weight += new_likelihood - old_likelihoods[p_idx]
                    old_likelihoods[p_idx] = new_likelihood
                    log_weights[p_idx] = p.weight

                W, normalized_weights = self._normalize(log_weights)

                # Step 15 of algorithm
                # list_of_particles = self.resample(list_of_particles, normalized_weights)
                # resample all but first particle
                re_n_w = normalized_weights[1:] / normalized_weights[1:].sum()
                indices = range(1, len(list_of_particles))
                new_indices = np.random.choice(indices, size=len(list_of_particles) - 1, p=re_n_w)
                list_of_particles[1:] = np.array(list_of_particles)[new_indices]
                old_likelihoods[1:] = old_likelihoods[new_indices]

                # Step 16 of algorithm
                w_t = W - log_num_particles
                for c in range(self.num_particles):
                    list_of_particles[c].weight = w_t

                #                ### XXX Revisar esto!!!!
                #                # Step 17 of algorithm
                #                non_available_nodes_for_expansion  = np.ones(self.num_particles)
                #                for c in range(self.num_particles):
                #                    if len(list_of_particles[c].expansion_nodes) != 0:
                #                        non_available_nodes_for_expansion[c] = 0
                #                if np.all(non_available_nodes_for_expansion):
                #                    break
                non_available_nodes_for_expansion = True
                for c in range(self.num_particles):
                    if len(list_of_particles[c].expansion_nodes) != 0:
                        non_available_nodes_for_expansion = False
                        break
                if non_available_nodes_for_expansion:
                    break

            # print(idx, t, normalized_weights)
            new_tree = np.random.choice(list_of_particles, p=normalized_weights)
            self.previous_trees_particles_list[tree.tree_id] = new_tree
            bart.trees[idx] = new_tree.tree
            new_prediction = new_tree.tree.predict_output(num_observations)
            output[idx] = new_prediction
            bart.sum_trees_output = bart.Y - R_j + new_prediction

        return np.sum(output, axis=0)

    @staticmethod
    def competence(var, has_grad):
        """
        PGBART is only suitable for BART distributions
        """
        if isinstance(var.distribution, BART):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    def _normalize(self, log_w):  # this function needs a home sweet home
        """
        use logsumexp trick to get W and softmax to get normalized_weights
        """
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        w_ = np.exp(log_w_)
        w_sum = w_.sum()
        W = log_w_max + np.log(w_sum)
        normalized_weights = w_ / w_sum
        # stabilize weights to avoid assigning exactly zero probability to a particle
        normalized_weights += 1e-12

        return W, normalized_weights

    def get_previous_tree_particle(self, tree_id, t):
        previous_tree_particle = self.previous_trees_particles_list[tree_id]
        previous_tree_particle.set_particle_to_step(t)
        return previous_tree_particle

    def init_particles(self, tree_id, R_j, num_observations):
        list_of_particles = []
        initial_value_leaf_nodes = R_j.mean()
        initial_idx_data_points_leaf_nodes = np.array(range(num_observations), dtype="int32")
        new_tree = Tree.init_tree(
            tree_id=tree_id,
            leaf_node_value=initial_value_leaf_nodes,
            idx_data_points=initial_idx_data_points_leaf_nodes,
        )
        for _ in range(self.num_particles):
            new_particle = Particle(new_tree)
            list_of_particles.append(new_particle)
        return list_of_particles

    def resample(self, list_of_particles, normalized_weights):
        list_of_particles = np.random.choice(
            list_of_particles, size=len(list_of_particles), p=normalized_weights
        )
        return list_of_particles


class Particle:
    def __init__(self, tree):
        self.tree = tree.copy()  # Mantiene el arbol que nos interesa en este momento
        self.expansion_nodes = self.tree.idx_leaf_nodes.copy()  # This should be the array [0]
        self.tree_history = [self.tree.copy()]
        self.expansion_nodes_history = [self.expansion_nodes.copy()]
        self.weight = 0.0

    def sample_tree_sequential(self, bart):
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            log_prob = self.tree[index_leaf_node].prior_log_probability_node(bart.alpha, bart.beta)

            if np.exp(log_prob) < np.random.random():
                self.grow_successful = bart.grow_tree(self.tree, index_leaf_node)
                # TODO: in case the grow_tree fails, should we try to sample the tree from another leaf node?
                if self.grow_successful:
                    # Add new leaf nodes indexes
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
            self.tree_history.append(self.tree.copy())
            self.expansion_nodes_history.append(self.expansion_nodes.copy())

    def set_particle_to_step(self, t):
        if len(self.tree_history) <= t:
            self.tree = self.tree_history[-1]
            self.expansion_nodes = self.expansion_nodes_history[-1]
        else:
            self.tree = self.tree_history[t]
            self.expansion_nodes = self.expansion_nodes_history[t]

    def init_weight(self):
        # TODO
        return 1.0

    def update_weight(self):
        # TODO
        pass
