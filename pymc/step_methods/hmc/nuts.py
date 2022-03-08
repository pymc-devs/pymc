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

from collections import namedtuple

import numpy as np

from pymc.aesaraf import floatX
from pymc.backends.report import SamplerWarning, WarningType
from pymc.math import logbern, logdiffexp_numpy
from pymc.step_methods.arraystep import Competence
from pymc.step_methods.hmc.base_hmc import BaseHMC, DivergenceInfo, HMCStepData
from pymc.step_methods.hmc.integration import IntegrationError
from pymc.vartypes import continuous_types

__all__ = ["NUTS"]


class NUTS(BaseHMC):
    r"""A sampler for continuous variables based on Hamiltonian mechanics.

    NUTS automatically tunes the step size and the number of steps per
    sample. A detailed description can be found at [1], "Algorithm 6:
    Efficient No-U-Turn Sampler with Dual Averaging".

    NUTS provides a number of statistics that can be accessed with
    `trace.get_sampler_stats`:

    - `mean_tree_accept`: The mean acceptance probability for the tree
      that generated this sample. The mean of these values across all
      samples but the burn-in should be approximately `target_accept`
      (the default for this is 0.8).
    - `diverging`: Whether the trajectory for this sample diverged. If
      there are any divergences after burnin, this indicates that
      the results might not be reliable. Reparametrization can
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
      `2 ** depth`. If the tree size is large, the sampler is
      using a lot of leapfrog steps to find the next sample. This can for
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
    - `model_logp`: The model log-likelihood for this sample.
    - `process_time_diff`: The time it took to draw the sample, as defined
      by the python standard library `time.process_time`. This counts all
      the CPU time, including worker processes in BLAS and OpenMP.
    - `perf_counter_diff`: The time it took to draw the sample, as defined
      by the python standard library `time.perf_counter` (wall time).
    - `perf_counter_start`: The value of `time.perf_counter` at the beginning
      of the computation of the draw.

    References
    ----------
    .. [1] Hoffman, Matthew D., & Gelman, Andrew. (2011). The No-U-Turn
       Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.
    """

    name = "nuts"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "depth": np.int64,
            "step_size": np.float64,
            "tune": bool,
            "mean_tree_accept": np.float64,
            "step_size_bar": np.float64,
            "tree_size": np.float64,
            "diverging": bool,
            "energy_error": np.float64,
            "energy": np.float64,
            "max_energy_error": np.float64,
            "model_logp": np.float64,
            "process_time_diff": np.float64,
            "perf_counter_diff": np.float64,
            "perf_counter_start": np.float64,
        }
    ]

    def __init__(self, vars=None, max_treedepth=10, early_max_treedepth=8, **kwargs):
        r"""Set up the No-U-Turn sampler.

        Parameters
        ----------
        vars: list, default=None
            List of value variables. If None, all continuous RVs from the
            model are included.
        Emax: float, default 1000
            Maximum energy change allowed during leapfrog steps. Larger
            deviations will abort the integration.
        target_accept: float, default .8
            Adapt the step size such that the average acceptance
            probability across the trajectories are close to target_accept.
            Higher values for target_accept lead to smaller step sizes.
            Setting this to higher values like 0.9 or 0.99 can help
            with sampling from difficult posteriors. Valid values are
            between 0 and 1 (exclusive).
        step_scale: float, default 0.25
            Size of steps to take, automatically scaled down by `1/n**(1/4)`.
            If step size adaptation is switched off, the resulting step size
            is used. If adaptation is enabled, it is used as initial guess.
        gamma: float, default .05
        k: float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0: int, default 10
            Parameter for dual averaging. Higher values slow initial
            adaptation.
        adapt_step_size: bool, default=True
            Whether step size adaptation should be enabled. If this is
            disabled, `k`, `t0`, `gamma` and `target_accept` are ignored.
        max_treedepth: int, default=10
            The maximum tree depth. Trajectories are stopped when this
            depth is reached.
        early_max_treedepth: int, default=8
            The maximum tree depth during the first 200 tuning samples.
        scaling: array_like, ndim = {1,2}
            The inverse mass, or precision matrix. One dimensional arrays are
            interpreted as diagonal matrices. If `is_cov` is set to True,
            this will be interpreded as the mass or covariance matrix.
        is_cov: bool, default=False
            Treat the scaling as mass or covariance matrix.
        potential: Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods. It can be specified instead
            of the scaling matrix.
        model: pymc.Model
            The model
        kwargs: passed to BaseHMC

        Notes
        -----
        The step size adaptation stops when `self.tune` is set to False.
        This is usually achieved by setting the `tune` parameter if
        `pm.sample` to the desired number of tuning steps.
        """
        super().__init__(vars, **kwargs)

        self.max_treedepth = max_treedepth
        self.early_max_treedepth = early_max_treedepth
        self._reached_max_treedepth = 0

    def _hamiltonian_step(self, start, p0, step_size):
        if self.tune and self.iter_count < 200:
            max_treedepth = self.early_max_treedepth
        else:
            max_treedepth = self.max_treedepth

        tree = _Tree(len(p0), self.integrator, start, step_size, self.Emax)

        for _ in range(max_treedepth):
            direction = logbern(np.log(0.5)) * 2 - 1
            divergence_info, turning = tree.extend(direction)

            if divergence_info or turning:
                break
        else:
            if not self.tune:
                self._reached_max_treedepth += 1

        stats = tree.stats()
        accept_stat = stats["mean_tree_accept"]
        return HMCStepData(tree.proposal, accept_stat, divergence_info, stats)

    @staticmethod
    def competence(var, has_grad):
        """Check how appropriate this class is for sampling a random variable."""

        dist = getattr(var.owner, "op", None)
        if var.dtype in continuous_types and has_grad:
            return Competence.PREFERRED
        return Competence.INCOMPATIBLE

    def warnings(self):
        warnings = super().warnings()
        n_samples = self._samples_after_tune
        n_treedepth = self._reached_max_treedepth

        if n_samples > 0 and n_treedepth / float(n_samples) > 0.05:
            msg = (
                "The chain reached the maximum tree depth. Increase "
                "max_treedepth, increase target_accept or reparameterize."
            )
            warn = SamplerWarning(WarningType.TREEDEPTH, msg, "warn")
            warnings.append(warn)
        return warnings


# A proposal for the next position
Proposal = namedtuple("Proposal", "q, q_grad, energy, log_p_accept_weighted, logp")

# A subtree of the binary tree built by nuts.
Subtree = namedtuple(
    "Subtree",
    "left, right, p_sum, proposal, log_size, log_weighted_accept_sum, n_proposals",
)


class _Tree:
    def __init__(self, ndim, integrator, start, step_size, Emax):
        """Binary tree from the NUTS algorithm.

        Parameters
        ----------
        leapfrog: function
            A function that performs a single leapfrog step.
        start: integration.State
            The starting point of the trajectory.
        step_size: float
            The step size to use in this tree
        Emax: float
            The maximum energy change to accept before aborting the
            transition as diverging.
        """
        self.ndim = ndim
        self.integrator = integrator
        self.start = start
        self.step_size = step_size
        self.Emax = Emax
        self.start_energy = np.array(start.energy)

        self.left = self.right = start
        self.proposal = Proposal(start.q.data, start.q_grad, start.energy, 1.0, start.model_logp)
        self.depth = 0
        self.log_size = 0
        self.log_weighted_accept_sum = -np.inf
        self.mean_tree_accept = 0.0
        self.n_proposals = 0
        self.p_sum = start.p.data.copy()
        self.max_energy_change = 0

    def extend(self, direction):
        """Double the treesize by extending the tree in the given direction.

        If direction is larger than 0, extend it to the right, otherwise
        extend it to the left.

        Return a tuple `(diverging, turning)` of type (DivergenceInfo, bool).
        `diverging` indicates, that the tree extension was aborted because
        the energy change exceeded `self.Emax`. `turning` indicates that
        the tree extension was stopped because the termination criterior
        was reached (the trajectory is turning back).
        """
        if direction > 0:
            tree, diverging, turning = self._build_subtree(
                self.right, self.depth, floatX(np.asarray(self.step_size))
            )
            leftmost_begin, leftmost_end = self.left, self.right
            rightmost_begin, rightmost_end = tree.left, tree.right
            leftmost_p_sum = self.p_sum
            rightmost_p_sum = tree.p_sum
            self.right = tree.right
        else:
            tree, diverging, turning = self._build_subtree(
                self.left, self.depth, floatX(np.asarray(-self.step_size))
            )
            leftmost_begin, leftmost_end = tree.right, tree.left
            rightmost_begin, rightmost_end = self.left, self.right
            leftmost_p_sum = tree.p_sum
            rightmost_p_sum = self.p_sum
            self.left = tree.right

        self.depth += 1
        self.n_proposals += tree.n_proposals

        if diverging or turning:
            return diverging, turning

        size1, size2 = self.log_size, tree.log_size
        if logbern(size2 - size1):
            self.proposal = tree.proposal

        self.log_size = np.logaddexp(self.log_size, tree.log_size)
        self.log_weighted_accept_sum = np.logaddexp(
            self.log_weighted_accept_sum, tree.log_weighted_accept_sum
        )
        self.p_sum[:] += tree.p_sum

        # Additional turning check only when tree depth > 0 to avoid redundant work
        if self.depth > 0:
            left, right = self.left, self.right
            p_sum = self.p_sum
            turning = (p_sum.dot(left.v) <= 0) or (p_sum.dot(right.v) <= 0)
            p_sum1 = leftmost_p_sum + rightmost_begin.p.data
            turning1 = (p_sum1.dot(leftmost_begin.v) <= 0) or (p_sum1.dot(rightmost_begin.v) <= 0)
            p_sum2 = leftmost_end.p.data + rightmost_p_sum
            turning2 = (p_sum2.dot(leftmost_end.v) <= 0) or (p_sum2.dot(rightmost_end.v) <= 0)
            turning = turning | turning1 | turning2

        return diverging, turning

    def _single_step(self, left, epsilon):
        """Perform a leapfrog step and handle error cases."""
        try:
            # `State` type
            right = self.integrator.step(epsilon, left)
        except IntegrationError as err:
            error_msg = str(err)
            error = err
            right = None
        else:
            # h - H0
            energy_change = right.energy - self.start_energy
            if np.isnan(energy_change):
                energy_change = np.inf

            if np.abs(energy_change) > np.abs(self.max_energy_change):
                self.max_energy_change = energy_change
            if np.abs(energy_change) < self.Emax:
                # Acceptance statistic
                # e^{H(q_0, p_0) - H(q_n, p_n)} max(1, e^{H(q_0, p_0) - H(q_n, p_n)})
                # Saturated Metropolis accept probability with Boltzmann weight
                # if h - H0 < 0
                log_p_accept_weighted = -energy_change + min(0.0, -energy_change)
                log_size = -energy_change
                proposal = Proposal(
                    right.q.data,
                    right.q_grad,
                    right.energy,
                    log_p_accept_weighted,
                    right.model_logp,
                )
                tree = Subtree(
                    right, right, right.p.data, proposal, log_size, log_p_accept_weighted, 1
                )
                return tree, None, False
            else:
                error_msg = f"Energy change in leapfrog step is too large: {energy_change}."
                error = None
        tree = Subtree(None, None, None, None, -np.inf, -np.inf, 1)
        divergance_info = DivergenceInfo(error_msg, error, left, right)
        return tree, divergance_info, False

    def _build_subtree(self, left, depth, epsilon):
        if depth == 0:
            return self._single_step(left, epsilon)

        tree1, diverging, turning = self._build_subtree(left, depth - 1, epsilon)
        if diverging or turning:
            return tree1, diverging, turning

        tree2, diverging, turning = self._build_subtree(tree1.right, depth - 1, epsilon)

        left, right = tree1.left, tree2.right

        if not (diverging or turning):
            p_sum = tree1.p_sum + tree2.p_sum
            turning = (p_sum.dot(left.v) <= 0) or (p_sum.dot(right.v) <= 0)
            # Additional U turn check only when depth > 1 to avoid redundant work.
            if depth - 1 > 0:
                p_sum1 = tree1.p_sum + tree2.left.p.data
                turning1 = (p_sum1.dot(tree1.left.v) <= 0) or (p_sum1.dot(tree2.left.v) <= 0)
                p_sum2 = tree1.right.p.data + tree2.p_sum
                turning2 = (p_sum2.dot(tree1.right.v) <= 0) or (p_sum2.dot(tree2.right.v) <= 0)
                turning = turning | turning1 | turning2

            log_size = np.logaddexp(tree1.log_size, tree2.log_size)
            log_weighted_accept_sum = np.logaddexp(
                tree1.log_weighted_accept_sum, tree2.log_weighted_accept_sum
            )
            if logbern(tree2.log_size - log_size):
                proposal = tree2.proposal
            else:
                proposal = tree1.proposal
        else:
            p_sum = tree1.p_sum
            log_size = tree1.log_size
            log_weighted_accept_sum = tree1.log_weighted_accept_sum
            proposal = tree1.proposal

        n_proposals = tree1.n_proposals + tree2.n_proposals

        tree = Subtree(left, right, p_sum, proposal, log_size, log_weighted_accept_sum, n_proposals)
        return tree, diverging, turning

    def stats(self):
        # Update accept stat if any subtrees were accepted
        if self.log_size > 0:
            # Remove contribution from initial state which is always a perfect
            # accept
            log_sum_weight = logdiffexp_numpy(self.log_size, 0.0)
            self.mean_tree_accept = np.exp(self.log_weighted_accept_sum - log_sum_weight)
        return {
            "depth": self.depth,
            "mean_tree_accept": self.mean_tree_accept,
            "energy_error": self.proposal.energy - self.start.energy,
            "energy": self.proposal.energy,
            "tree_size": self.n_proposals,
            "max_energy_error": self.max_energy_change,
            "model_logp": self.proposal.logp,
        }
