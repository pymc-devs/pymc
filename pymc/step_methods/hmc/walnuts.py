#   Copyright 2024 - present The PyMC Developers
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

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from pymc.stats.convergence import SamplerWarning
from pymc.step_methods.compound import Competence
from pymc.step_methods.hmc.base_hmc import BaseHMC, DivergenceInfo, HMCStepData
from pymc.step_methods.hmc.integration import IntegrationError, State
from pymc.vartypes import continuous_types

__all__ = ["WALNUTS"]


class WalnutsStepData(NamedTuple):
    """State during adaptive integration."""

    state: State
    n_steps: int
    energy_error: float


class WalnutsTree:
    """Binary tree for WALNUTS algorithm.

    Similar to NUTS tree but with adaptive step size within orbits.
    """

    def __init__(
        self,
        integrator,
        start: State,
        step_size: float,
        Emax: float,
        max_error: float,
        rng: np.random.Generator,
    ):
        self.integrator = integrator
        self.start = start
        self.step_size = step_size
        self.Emax = Emax
        self.max_error = max_error
        self.rng = rng

        self.left = self.right = start
        self.depth = 0
        self.n_proposals = 0
        self.max_energy_error = 0.0
        self.sum_accept_stat = 0.0

        # WALNUTS-specific
        self.n_steps_total = 0
        self.n_stable_steps = 0

    def _find_stable_steps(self, state: State, direction: int) -> tuple[bool, int]:
        """Find minimum number of steps for stable integration."""
        initial_energy = state.energy

        # Try powers of 2: 1, 2, 4, 8
        for n in range(4):  # Simplified range
            n_steps = 2**n
            test_state = state
            max_error = 0.0

            try:
                for _ in range(n_steps):
                    test_state = self.integrator.step(
                        direction * self.step_size / n_steps, test_state
                    )
                    energy_error = abs(test_state.energy - initial_energy)
                    max_error = max(max_error, energy_error)

                    if max_error > self.max_error:
                        break

                if max_error <= self.max_error:
                    return True, n_steps

            except IntegrationError:
                continue

        return False, 1

    def _extend_adaptive(self, state: State, direction: int) -> tuple[State | None, bool, bool]:
        """Extend tree with adaptive step size."""
        # Find stable number of steps
        is_stable, n_steps = self._find_stable_steps(state, direction)

        if not is_stable:
            return None, True, False  # diverged

        # Perform integration with adaptive steps
        current_state = state
        actual_step = direction * self.step_size / n_steps

        try:
            for _ in range(n_steps):
                current_state = self.integrator.step(actual_step, current_state)

                energy_error = abs(current_state.energy - self.start.energy)
                if energy_error > self.Emax:
                    return None, True, False  # diverged

                self.max_energy_error = max(self.max_energy_error, energy_error)

            self.n_proposals += 1
            self.n_steps_total += n_steps
            self.n_stable_steps += n_steps

            # Acceptance statistic
            accept_stat = min(1.0, np.exp(self.start.energy - current_state.energy))
            self.sum_accept_stat += accept_stat

            return current_state, False, False

        except IntegrationError:
            return None, True, False

    def extend(self, direction: int) -> tuple[DivergenceInfo | None, bool]:
        """Extend the tree in given direction."""
        if direction > 0:
            new_state, diverged, turning = self._extend_adaptive(self.right, direction)
            if not diverged and new_state is not None:
                self.right = new_state
        else:
            new_state, diverged, turning = self._extend_adaptive(self.left, direction)
            if not diverged and new_state is not None:
                self.left = new_state

        self.depth += 1

        divergence_info = None
        if diverged:
            divergence_info = DivergenceInfo(
                "Energy error exceeded threshold in WALNUTS",
                None,
                self.left if direction < 0 else self.right,
                None,
            )
            return divergence_info, False

        # Check for U-turn
        turning = False
        if new_state is not None:
            delta_q = self.right.q.data - self.left.q.data
            turning = np.dot(self.left.p, delta_q) <= 0 or np.dot(self.right.p, delta_q) <= 0

        return divergence_info, turning

    def get_proposal(self) -> State:
        """Get proposal state (currently just the right endpoint)."""
        return self.right if self.rng.random() < 0.5 else self.left

    def stats(self) -> dict[str, Any]:
        """Get tree statistics."""
        mean_accept = self.sum_accept_stat / max(1, self.n_proposals)
        return {
            "depth": self.depth,
            "mean_tree_accept": mean_accept,
            "energy_error": self.right.energy - self.start.energy,
            "energy": self.right.energy,
            "tree_size": self.n_proposals,
            "max_energy_error": self.max_energy_error,
            "model_logp": self.right.model_logp,
            "index_in_trajectory": self.right.index_in_trajectory,
            "n_steps_total": self.n_steps_total,
            "avg_steps_per_proposal": self.n_steps_total / max(1, self.n_proposals),
            "largest_eigval": np.nan,
            "smallest_eigval": np.nan,
        }


class WALNUTS(BaseHMC):
    """Within-orbit Adaptive Step-length No-U-Turn Sampler.

    WALNUTS (Bou-Rabee et al., 2025) extends NUTS by adapting the integration step size within
    each trajectory. This can improve numerical stability in models
    with varying curvature.

    Parameters
    ----------
    vars : list, optional
        Variables to sample. If None, all continuous variables in the model.
    max_error : float, default=1.0
        Maximum allowed Hamiltonian error for adaptive steps.
    max_treedepth : int, default=10
        Maximum depth of the binary tree.
    early_max_treedepth : int, default=8
        Maximum depth during tuning phase.
    **kwargs
        Additional arguments passed to BaseHMC.

    References
    ----------
    .. [1] Bou-Rabee, N., Carpenter, B., Kleppe, T. S., & Liu, S. (2025).
       The Within-Orbit Adaptive Leapfrog No-U-Turn Sampler.
       arXiv preprint arXiv:2506.18746.
       https://arxiv.org/abs/2506.18746v1
    """

    name = "walnuts"

    default_blocked = True

    stats_dtypes_shapes = {
        "depth": (np.int64, []),
        "step_size": (np.float64, []),
        "tune": (bool, []),
        "mean_tree_accept": (np.float64, []),
        "step_size_bar": (np.float64, []),
        "tree_size": (np.float64, []),
        "diverging": (bool, []),
        "energy_error": (np.float64, []),
        "energy": (np.float64, []),
        "max_energy_error": (np.float64, []),
        "model_logp": (np.float64, []),
        "process_time_diff": (np.float64, []),
        "perf_counter_diff": (np.float64, []),
        "perf_counter_start": (np.float64, []),
        "largest_eigval": (np.float64, []),
        "smallest_eigval": (np.float64, []),
        "index_in_trajectory": (np.int64, []),
        "reached_max_treedepth": (bool, []),
        "warning": (SamplerWarning, None),
        "n_steps_total": (np.int64, []),
        "avg_steps_per_proposal": (np.float64, []),
    }

    def __init__(
        self,
        vars=None,
        max_error=1.0,
        max_treedepth=10,
        early_max_treedepth=8,
        **kwargs,
    ):
        """Initialize WALNUTS sampler."""
        self.max_error = max_error
        self.max_treedepth = max_treedepth
        self.early_max_treedepth = early_max_treedepth

        super().__init__(vars, **kwargs)

    def _hamiltonian_step(self, start, p0, step_size) -> HMCStepData:
        """Perform a single WALNUTS iteration."""
        if self.tune and self.iter_count < 200:
            max_treedepth = self.early_max_treedepth
        else:
            max_treedepth = self.max_treedepth

        tree = WalnutsTree(self.integrator, start, step_size, self.Emax, self.max_error, self.rng)

        reached_max_treedepth = False
        divergence_info = None
        for _ in range(max_treedepth):
            direction = (self.rng.random() < 0.5) * 2 - 1
            divergence_info, turning = tree.extend(direction)

            if divergence_info or turning:
                break
        else:  # no-break
            reached_max_treedepth = not self.tune

        stats = tree.stats()
        stats["reached_max_treedepth"] = reached_max_treedepth

        # Get proposal from tree
        proposal = tree.get_proposal()
        mean_accept = stats["mean_tree_accept"]

        return HMCStepData(proposal, mean_accept, divergence_info, stats)

    @staticmethod
    def competence(var, has_grad):
        """Check if WALNUTS can sample this variable."""
        if var.dtype in continuous_types and has_grad:
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE
