from collections import namedtuple

from ..arraystep import Competence
from .base_hmc import BaseHMC
from pymc3.theanof import floatX
from pymc3.vartypes import continuous_types

import numpy as np
import numpy.random as nr

__all__ = ['NUTS']


def bern(p):
    return nr.uniform() < p


class NUTS(BaseHMC):
    """
    Automatically tunes step size and adjust number of steps for good performance.

    Implements "Algorithm 6: Efficient No-U-Turn Sampler with Dual Averaging" in:

    Hoffman, Matthew D., & Gelman, Andrew. (2011).
    The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.
    """
    default_blocked = True
    generates_stats = True
    stats_dtypes = [{
        'depth': np.int64,
        'step_size': np.float64,
        'tune': np.bool,
        'mean_tree_accept': np.float64,
        'h_bar': np.float64,
        'step_size_bar': np.float64,
        'tree_size': np.float64,
        'diverging': np.bool,
        'energy_change': np.float64,
        'max_energy_change': np.float64,
    }]

    def __init__(self, vars=None, Emax=1000, target_accept=0.8,
                 gamma=0.05, k=0.75, t0=10, adapt_step_size=True, **kwargs):
        """
        Parameters
        ----------
        vars : list of Theano variables, default continuous vars
        Emax : float, default 1000
            Maximum energy change allowed during leapfrog steps. Larger
            deviations will abort the integration.
        target_accept : float (0,1) default .8
            target for avg accept probability between final branch and initial position
        step_scale : float, default 0.25
            Size of steps to take, automatically scaled down by 1/n**(1/4).
            If step size adaptation is switched off, the resulting step size
            is used. If adaptation is enabled, it is used as initial guess.
        gamma : float, default .05
        k : float (.5,1) default .75
            scaling of speed of adaptation
        t0 : int, default 10
            slows inital adapatation
        adapt_step_size : bool
            Whether step size should be enabled. If this is disabled,
            `k`, `t0`, `gamma` and `target_accept` are ignored.
        kwargs: passed to BaseHMC

        The step size adaptation stops when `self.tune` is set to False.
        """
        super(NUTS, self).__init__(vars, use_single_leapfrog=True, **kwargs)

        self.Emax = Emax

        self.target_accept = target_accept
        self.gamma = gamma
        self.k = k
        self.t0 = t0

        self.h_bar = 0
        self.mu = np.log(self.step_size * 10)
        self.log_step_size = np.log(self.step_size)
        self.log_step_size_bar = 0
        self.m = 1
        self.adapt_step_size = adapt_step_size

        self.tune = True

    def astep(self, q0):
        p0 = self.potential.random()
        start_energy = self.compute_energy(q0, p0)

        if not self.adapt_step_size:
            step_size = self.step_size
        elif self.tune:
            step_size = np.exp(self.log_step_size)
        else:
            step_size = np.exp(self.log_step_size_bar)

        u = nr.uniform()
        start = Edge(q0, p0, self.dlogp(q0), start_energy)
        tree = Tree(self.leapfrog, start, u, step_size, self.Emax)

        while True:
            direction = bern(0.5) * 2 - 1
            diverging, turning = tree.extend(direction)
            q = tree.proposal.q

            if diverging or turning:
                break

        w = 1. / (self.m + self.t0)
        self.h_bar = ((1 - w) * self.h_bar +
                      w * (self.target_accept - tree.accept_sum * 1. / tree.n_proposals))

        if self.tune:
            self.log_step_size = self.mu - self.h_bar * np.sqrt(self.m) / self.gamma
            mk = self.m ** -self.k
            self.log_step_size_bar = mk * self.log_step_size + (1 - mk) * self.log_step_size_bar

        self.m += 1

        stats = {
            'step_size': step_size,
            'tune': self.tune,
            'h_bar': self.h_bar,
            'step_size_bar': np.exp(self.log_step_size_bar),
            'diverging': diverging,
        }

        stats.update(tree.stats())

        return q, [stats]

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


# A node in the NUTS tree that is at the far right or left of the tree
Edge = namedtuple("Edge", 'q, p, q_grad, energy')

# A proposal for the next position
Proposal = namedtuple("Proposal", "q, energy, p_accept")

# A subtree of the binary tree build by nuts.
Subtree = namedtuple("Subtree", "left, right, proposal, depth, size, accept_sum, n_proposals")


class Tree(object):
    def __init__(self, leapfrog, start, u, step_size, Emax):
        """Binary tree from the NUTS algorithm.

        Parameters
        ----------
        leapfrog : function
            A function that performs a single leapfrog step.
        start : Edge
            The starting point of the trajectory.
        u : float in [0, 1]
            Random slice sampling variable.
        step_size : float
            The step size to use in this tree
        Emax : float
            The maximum energy change to accept before aborting the
            transition as diverging.
        """
        self.leapfrog = leapfrog
        self.start = start
        self.log_u = np.log(u)
        self.step_size = step_size
        self.Emax = Emax
        self.start_energy = np.array(start.energy)

        self.left = self.right = start
        self.proposal = Proposal(start.q, start.energy, 1.0)
        self.depth = 0
        self.size = 1
        # TODO Why not a global accept sum and n_proposals?
        #self.accept_sum = 0
        #self.n_proposals = 0
        self.max_energy_change = 0

    def extend(self, direction):
        """Double the treesize by extending the tree in the given direction.

        If direction is larger than 0, extend it to the right, otherwise
        extend it to the left.

        Return a tuple `(diverging, turning)` of type (bool, bool).
        `diverging` indicates, that the tree extension was aborted because
        the energy change exceeded `self.Emax`. `turning` indicates that
        the tree extension was stopped because the termination criterior
        was reached (the trajectory is turning back).
        """
        if direction > 0:
            tree, diverging, turning = self._build_subtree(
                self.right, self.depth, floatX(np.asarray(self.step_size)))
            self.right = tree.right
        else:
            tree, diverging, turning = self._build_subtree(
                self.left, self.depth, floatX(np.asarray(- self.step_size)))
            self.left = tree.right

        ok = not (diverging or turning)
        if ok and bern(min(1, tree.size / self.size)):
            self.proposal = tree.proposal

        self.depth += 1
        self.size += tree.size
        # TODO why not +=
        #self.accept_sum += tree.accept_sum
        self.accept_sum = tree.accept_sum
        #self.n_proposals += tree.n_proposals
        self.n_proposals = tree.n_proposals

        left, right = self.left, self.right
        span = right.q - left.q
        turning = turning or (span.dot(left.p) < 0) or (span.dot(right.p) < 0)
        return diverging, turning

    def _build_subtree(self, left, depth, epsilon):
        if depth == 0:
            right = self.leapfrog(left.q, left.p, left.q_grad, epsilon)
            right = Edge(*right)
            energy_change = right.energy - self.start_energy
            if np.abs(energy_change) > np.abs(self.max_energy_change):
                self.max_energy_change = energy_change
            p_accept = min(1, np.exp(-energy_change))

            size = int(self.log_u + energy_change <= 0)
            diverging = not (self.log_u + energy_change < self.Emax)

            proposal = Proposal(right.q, right.energy, p_accept)
            tree = Subtree(right, right, proposal, 1, size, p_accept, 1)
            return tree, diverging, False

        tree1, diverging, turning = self._build_subtree(left, depth - 1, epsilon)
        if diverging or turning:
            return tree1, diverging, turning

        tree2, diverging, turning = self._build_subtree(tree1.right, depth - 1, epsilon)

        size = tree1.size + tree2.size
        accept_sum = tree1.accept_sum + tree2.accept_sum
        n_proposals = tree1.n_proposals + tree2.n_proposals

        left, right = tree1.left, tree2.right
        span = np.sign(epsilon) * (right.q - left.q)
        turning = turning or (span.dot(left.p) < 0) or (span.dot(right.p) < 0)

        if bern(tree2.size * 1. / max(size, 1)):
            proposal = tree2.proposal
        else:
            proposal = tree1.proposal

        tree = Subtree(left, right, proposal, depth, size, accept_sum, n_proposals)
        return tree, diverging, turning

    def stats(self):
        return {
            'depth': self.depth,
            'mean_tree_accept': self.accept_sum / self.n_proposals,
            'energy_change': self.proposal.energy - self.start.energy,
            'tree_size': self.n_proposals,
            'max_energy_change': self.max_energy_change,
        }
