from ..arraystep import Competence
from .base_hmc import BaseHMC
from pymc3.vartypes import continuous_types

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

        self.Hbar = 0
        self.u = np.log(self.step_size * 10)
        self.m = 1

    def astep(self, q0):
        Emax = self.Emax
        e = self.step_size

        p0 = self.potential.random()
        E0 = self.compute_energy(q0, p0)

        u = nr.uniform()
        q = qn = qp = q0
        p = pn = pp = p0

        n, s, j = 1, 1, 0

        while s == 1:
            v = bern(0.5) * 2 - 1

            if v == -1:
                qn, pn, _, _, q1, n1, s1, a, na = buildtree(
                    self.leapfrog, qn, pn, u, v, j, e, Emax, E0)
            else:
                _, _, qp, pp, q1, n1, s1, a, na = buildtree(
                    self.leapfrog, qp, pp, u, v, j, e, Emax, E0)

            if s1 == 1 and bern(min(1, n1 * 1. / n)):
                q = q1

            n = n + n1

            span = qp - qn
            s = s1 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            j = j + 1

        p = -p

        w = 1. / (self.m + self.t0)
        self.Hbar = (1 - w) * self.Hbar + w * \
            (self.target_accept - a * 1. / na)

        self.step_size = np.exp(self.u - (self.m**self.k / self.gamma) * self.Hbar)
        self.m += 1

        return q

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def buildtree(leapfrog, q, p, u, v, j, e, Emax, E0):
    if j == 0:
        q1, p1, E = leapfrog(q, p, np.array(v * e))
        dE = E - E0

        n1 = int(np.log(u) + dE <= 0)
        s1 = int(np.log(u) + dE < Emax)
        return q1, p1, q1, p1, q1, n1, s1, min(1, np.exp(-dE)), 1
    qn, pn, qp, pp, q1, n1, s1, a1, na1 = buildtree(leapfrog, q, p, u, v, j - 1, e, Emax, E0)
    if s1 == 1:
        if v == -1:
            qn, pn, _, _, q11, n11, s11, a11, na11 = buildtree(
                leapfrog, qn, pn, u, v, j - 1, e, Emax, E0)
        else:
            _, _, qp, pp, q11, n11, s11, a11, na11 = buildtree(
                leapfrog, qp, pp, u, v, j - 1, e, Emax, E0)

        if bern(n11 * 1. / (max(n1 + n11, 1))):
            q1 = q11

        a1 = a1 + a11
        na1 = na1 + na11

        span = qp - qn
        s1 = s11 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
        n1 = n1 + n11
    return qn, pn, qp, pp, q1, n1, s1, a1, na1
