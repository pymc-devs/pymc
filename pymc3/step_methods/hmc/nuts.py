from collections import namedtuple

from ..arraystep import Competence
from .base_hmc import BaseHMC
from pymc3.theanof import floatX
from pymc3.vartypes import continuous_types

import numpy as np
import numpy.random as nr

__all__ = ['NUTS']


BinaryTree = namedtuple('BinaryTree',
                        'q, p, q_grad, proposal, leaf_size, is_valid_sample, p_accept, n_proposals')


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

    def __init__(self, vars=None, Emax=1000, target_accept=0.8,
                 gamma=0.05, k=0.75, t0=10, **kwargs):
        """
        Parameters
        ----------
        vars : list of Theano variables, default continuous vars
        Emax : float, default 1000
            Maximum energy change allowed during leapfrog steps. Larger
            deviations will abort the integration.
        target_accept : float (0,1) default .8
            target for avg accept probability between final branch and initial position
        gamma : float, default .05
        k : float (.5,1) default .75
            scaling of speed of adaptation
        t0 : int, default 10
            slows inital adapatation
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

        self.tune = True

    def astep(self, q0):
        p0 = self.potential.random()
        start_energy = self.compute_energy(q0, p0)

        if self.tune:
            step_size = np.exp(self.log_step_size)
        else:
            step_size = np.exp(self.log_step_size_bar)

        u = floatX(nr.uniform())

        q = qn = qp = q0
        qn_grad = qp_grad = self.dlogp(q)
        pn = pp = p0
        tree_size, depth = 1., 0
        keep_sampling = True

        while keep_sampling:
            direction = bern(0.5) * 2 - 1
            q_edge, p_edge, q_edge_grad = {-1: (qn, pn, qn_grad), 1: (qp, pp, qp_grad)}[direction]

            tree = buildtree(self.leapfrog, q_edge, p_edge, q_edge_grad, u, direction,
                             depth, step_size, self.Emax, start_energy)

            if direction == -1:
                qn, pn, qn_grad = tree.q, tree.p, tree.q_grad
            else:
                qp, pp, qp_grad = tree.q, tree.p, tree.q_grad

            if tree.is_valid_sample and bern(min(1, tree.leaf_size / tree_size)):
                q = tree.proposal

            tree_size += tree.leaf_size

            span = qp - qn
            keep_sampling = tree.is_valid_sample and (span.dot(pn) >= 0) and (span.dot(pp) >= 0)
            depth += 1

        w = 1. / (self.m + self.t0)
        self.h_bar = ((1 - w) * self.h_bar +
                      w * (self.target_accept - tree.p_accept * 1. / tree.n_proposals))

        if self.tune:
            self.log_step_size = self.mu - self.h_bar * np.sqrt(self.m) / self.gamma
            mk = self.m ** -self.k
            self.log_step_size_bar = mk * self.log_step_size + (1 - mk) * self.log_step_size_bar

        self.m += 1

        return q

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def buildtree(leapfrog, q, p, q_grad, u, direction, depth, step_size, Emax, start_energy):
    if depth == 0:
        epsilon = floatX(np.asarray(direction * step_size))
        q, p, q_grad, new_energy = leapfrog(q, p, q_grad, epsilon)
        energy_change = new_energy - start_energy
        leaf_size = int(np.log(u) + energy_change <= 0)
        is_valid_sample = (np.log(u) + energy_change < Emax)
        p_accept = min(1, np.exp(-energy_change))
        return BinaryTree(q, p, q_grad, q, leaf_size, is_valid_sample, p_accept, 1)
    else:
        depth -= 1

    tree = buildtree(leapfrog, q, p, q_grad, u, direction, depth, step_size, Emax, start_energy)

    if tree.is_valid_sample:
        subtree = buildtree(leapfrog, tree.q, tree.p, tree.q_grad, u, direction, depth,
                            step_size, Emax, start_energy)
        if bern(subtree.leaf_size * 1. / max(subtree.leaf_size + tree.leaf_size, 1)):
            proposal = subtree.proposal
        else:
            proposal = tree.proposal
        leaf_size = subtree.leaf_size + tree.leaf_size
        p_accept = subtree.p_accept + tree.p_accept
        n_proposals = subtree.n_proposals + tree.n_proposals
        span = direction * (subtree.q - tree.q)
        is_valid_sample = (subtree.is_valid_sample and
                           span.dot(subtree.p) >= 0 and
                           span.dot(tree.p) >= 0)
        q, p, q_grad = subtree.q, subtree.p, subtree.q_grad
        return BinaryTree(q, p, q_grad, proposal, leaf_size, is_valid_sample, p_accept, n_proposals)
    else:
        return tree
