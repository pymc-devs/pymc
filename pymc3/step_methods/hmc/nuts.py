from collections import namedtuple

from ..arraystep import Competence
from .base_hmc import BaseHMC
from pymc3.theanof import floatX
from pymc3.vartypes import continuous_types

import numpy as np
import numpy.random as nr

__all__ = ['NUTS']


def logbern(log_p):
    if np.isnan(log_p):
        raise FloatingPointError("log_p can't be nan.")
    return np.log(nr.uniform()) < log_p


class NUTS(BaseHMC):
    """
    Automatically tunes step size and adjust number of steps for good performance.

    Implements "Algorithm 6: Efficient No-U-Turn Sampler with Dual Averaging" in:

    Hoffman, Matthew D., & Gelman, Andrew. (2011).
    The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.

    Nuts provides a number of statistics, that can be accessed using
    `trace.get_sampler_stats`:

    - `mean_tree_accept`: The mean acceptance probability for the tree
      that generated this sample. The mean of these values across all
      samples but the burn-in should be approximately `target_accept`
      (the default for this is 0.8).
    - `diverging`: Whether the trajectory for this sample diverged. If
      there are many diverging samples, this usually indicates that a
      region of the posterior has high curvature. Reparametrization can
      often help, but you can also try to increase `target_accept` to
      something like 0.9 or 0.95.
    - `energy`: The energy at the point in phase-space where the sample
      was accepted. This can be used to identify posteriors with
      problematically long tails. See below for an example.
    - `energy_change`: The difference in energy between the start and
      the end of the trajectory. For a perfect integrator this would
      always be zero.
    - `max_energy_change`: The maximum difference in energy along the
      whole trajectory.
    - `depth`: The depth of the tree that was used to generate this sample
    - `tree_size`: The number of leafs of the sampling tree, when the
      sample was accepted. This is usually a bit less than
      $2 ^ \text{depth}$. If the tree size is large, the sampler is using
      a lot of leapfrog steps to find the next sample. This can for
      example happen if there are strong correlations in the posterior,
      if the posterior has long tails, if there are regions of high
      curvature ("funnels"), or if the variance estimates in the mass
      matrix are inaccurate. Reparametrisation of the model or estimating
      the posterior variances from past samples might help.
    - `tune`: This is `True`, if step size adaptation was turned on when
      this sample was generated.
    - `step_size`: The step size used for this sample.
    - `step_size_bar`: The current best known step-size. After the tuning
       samples, the step size is set to this value. This should converge
       during tuning.
    """
    default_blocked = True
    generates_stats = True
    stats_dtypes = [{
        'depth': np.int64,
        'step_size': np.float64,
        'tune': np.bool,
        'mean_tree_accept': np.float64,
        'step_size_bar': np.float64,
        'tree_size': np.float64,
        'diverging': np.bool,
        'energy_error': np.float64,
        'energy': np.float64,
        'max_energy_error': np.float64,
    }]

    def __init__(self, vars=None, Emax=1000, target_accept=0.8,
                 gamma=0.05, k=0.75, t0=10, adapt_step_size=True,
                 max_treedepth=10, **kwargs):
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
        integrator : str, default "leapfrog"
            The integrator to use for the trajectories. One of "leapfrog",
            "two-stage" or "three-stage". The second two can increase
            sampling speed for some high dimensional problems.
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
        self.max_treedepth = max_treedepth

        self.tune = True

    def astep(self, q0):
        p0 = self.potential.random()
        v0 = self.compute_velocity(p0)
        start_energy = self.compute_energy(q0, p0)

        if not self.adapt_step_size:
            step_size = self.step_size
        elif self.tune:
            step_size = np.exp(self.log_step_size)
        else:
            step_size = np.exp(self.log_step_size_bar)

        start = Edge(q0, p0, v0, self.dlogp(q0), start_energy)
        tree = Tree(len(p0), self.leapfrog, start, step_size, self.Emax)

        for _ in range(self.max_treedepth):
            direction = logbern(np.log(0.5)) * 2 - 1
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
Edge = namedtuple("Edge", 'q, p, v, q_grad, energy')

# A proposal for the next position
Proposal = namedtuple("Proposal", "q, energy, p_accept")

# A subtree of the binary tree built by nuts.
Subtree = namedtuple("Subtree", "left, right, p_sum, proposal, log_size, accept_sum, n_proposals")


class Tree(object):
    def __init__(self, ndim, leapfrog, start, step_size, Emax):
        """Binary tree from the NUTS algorithm.

        Parameters
        ----------
        leapfrog : function
            A function that performs a single leapfrog step.
        start : Edge
            The starting point of the trajectory.
        step_size : float
            The step size to use in this tree
        Emax : float
            The maximum energy change to accept before aborting the
            transition as diverging.
        """
        self.ndim = ndim
        self.leapfrog = leapfrog
        self.start = start
        self.step_size = step_size
        self.Emax = Emax
        self.start_energy = np.array(start.energy)

        self.left = self.right = start
        self.proposal = Proposal(start.q, start.energy, 1.0)
        self.depth = 0
        self.log_size = 0
        self.accept_sum = 0
        self.n_proposals = 0
        self.p_sum = start.p.copy()
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
                self.left, self.depth, floatX(np.asarray(-self.step_size)))
            self.left = tree.right

        self.depth += 1
        self.accept_sum += tree.accept_sum
        self.n_proposals += tree.n_proposals

        if diverging or turning:
            return diverging, turning

        size1, size2 = self.log_size, tree.log_size
        if logbern(size2 - size1):
            self.proposal = tree.proposal

        self.log_size = np.logaddexp(self.log_size, tree.log_size)
        self.p_sum[:] += tree.p_sum

        left, right = self.left, self.right
        p_sum = self.p_sum
        turning = (p_sum.dot(left.v) <= 0) or (p_sum.dot(right.v) <= 0)

        return diverging, turning

    def _build_subtree(self, left, depth, epsilon):
        if depth == 0:
            right = self.leapfrog(left.q, left.p, left.q_grad, epsilon)
            right = Edge(*right)
            energy_change = right.energy - self.start_energy
            if np.isnan(energy_change):
                energy_change = np.inf

            if np.abs(energy_change) > np.abs(self.max_energy_change):
                self.max_energy_change = energy_change
            p_accept = min(1, np.exp(-energy_change))

            log_size = -energy_change
            diverging = energy_change > self.Emax

            proposal = Proposal(right.q, right.energy, p_accept)
            tree = Subtree(right, right, right.p, proposal, log_size, p_accept, 1)
            return tree, diverging, False

        tree1, diverging, turning = self._build_subtree(left, depth - 1, epsilon)
        if diverging or turning:
            return tree1, diverging, turning

        tree2, diverging, turning = self._build_subtree(tree1.right, depth - 1, epsilon)

        left, right = tree1.left, tree2.right

        if not (diverging or turning):
            p_sum = tree1.p_sum + tree2.p_sum
            turning = (p_sum.dot(left.v) <= 0) or (p_sum.dot(right.v) <= 0)

            log_size = np.logaddexp(tree1.log_size, tree2.log_size)
            if logbern(tree2.log_size - log_size):
                proposal = tree2.proposal
            else:
                proposal = tree1.proposal
        else:
            p_sum = tree1.p_sum
            log_size = tree1.log_size
            proposal = tree1.proposal

        accept_sum = tree1.accept_sum + tree2.accept_sum
        n_proposals = tree1.n_proposals + tree2.n_proposals

        tree = Subtree(left, right, p_sum, proposal, log_size, accept_sum, n_proposals)
        return tree, diverging, turning

    def stats(self):
        return {
            'depth': self.depth,
            'mean_tree_accept': self.accept_sum / self.n_proposals,
            'energy_error': self.proposal.energy - self.start.energy,
            'energy': self.proposal.energy,
            'tree_size': self.n_proposals,
            'max_energy_error': self.max_energy_change,
        }
