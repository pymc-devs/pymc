from ..arraystep import Competence
from .base_hmc import BaseHMC
from pymc3.vartypes import continuous_types
from pymc3.theanof import floatX
import numpy as np
import numpy.random as nr

__all__ = ['NUTS']


def bern(p):
    return np.random.uniform() < p


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
            maximum energy
        target_accept : float (0,1) default .8
            target for avg accept probability between final branch and initial position
        gamma : float, default .05
        k : float (.5,1) default .75
            scaling of speed of adaptation
        t0 : int, default 10
            slows inital adapatation
        kwargs: passed to BaseHMC
        """
        super(NUTS, self).__init__(vars, use_single_leapfrog=True, **kwargs)

        self.Emax = Emax

        self.target_accept = target_accept
        self.gamma = gamma
        self.k = k
        self.t0 = t0

        self.h_bar = 0
        self.u = np.log(self.step_size * 10)
        self.m = 1

    def astep(self, q0):
        p0 = self.potential.random()
        start_energy = self.compute_energy(q0, p0)

        u = floatX(nr.uniform())
        q = qn = qp = q0
        pn = pp = p0

        tree_size, depth = 1., 0
        keep_sampling = True

        while keep_sampling:
            direction = bern(0.5) * 2 - 1
            q_edge, p_edge = {-1: (qn, pn), 1: (qp, pp)}[direction]

            q_edge, p_edge, proposal, subtree_size, is_valid_sample, a, na = buildtree(
                self.leapfrog, q_edge, p_edge,
                u, direction, depth,
                self.step_size, self.Emax, start_energy)

            if direction == -1:
                qn, pn = q_edge, p_edge
            else:
                qp, pp = q_edge, p_edge

            if is_valid_sample and bern(min(1, subtree_size / tree_size)):
                q = proposal

            tree_size += subtree_size

            span = qp - qn
            keep_sampling = is_valid_sample and (span.dot(pn) >= 0) and (span.dot(pp) >= 0)
            depth += 1

        w = 1. / (self.m + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.target_accept - a * 1. / na)
        self.step_size = np.exp(self.u - (self.m**self.k / self.gamma) * self.h_bar)
        self.m += 1

        return q

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def buildtree(leapfrog, q, p, u, direction, depth, step_size, Emax, start_energy):
    if depth == 0:
        q_edge, p_edge, new_energy = leapfrog(q, p,
                                              floatX(np.asarray(direction * step_size)))
        energy_change = new_energy - start_energy

        leaf_size = int(np.log(u) + energy_change <= 0)
        is_valid_sample = (np.log(u) + energy_change < Emax)
        return q_edge, p_edge, q_edge, leaf_size, is_valid_sample, min(1, np.exp(-energy_change)), 1
    else:
        depth -= 1

    q, p, proposal, tree_size, is_valid_sample, a1, na1 = buildtree(
        leapfrog, q, p, u, direction, depth, step_size, Emax, start_energy)

    if is_valid_sample:
        q_edge, p_edge, new_proposal, subtree_size, is_valid_subsample, a11, na11 = buildtree(
            leapfrog, q, p, u, direction, depth, step_size, Emax, start_energy)

        tree_size += subtree_size
        if bern(subtree_size * 1. / max(tree_size, 1)):
            proposal = new_proposal

        a1 += a11
        na1 += na11
        span = direction * (q_edge - q)
        is_valid_sample = is_valid_subsample and (span.dot(p_edge) >= 0) and (span.dot(p) >= 0)
    else:
        q_edge, p_edge = q, p

    return q_edge, p_edge, proposal, tree_size, is_valid_sample, a1, na1
