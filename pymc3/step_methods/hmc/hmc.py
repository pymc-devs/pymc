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

import numpy as np

from pymc3.step_methods.arraystep import Competence
from pymc3.step_methods.hmc.base_hmc import BaseHMC, DivergenceInfo, HMCStepData
from pymc3.step_methods.hmc.integration import IntegrationError
from pymc3.vartypes import discrete_types

__all__ = ["HamiltonianMC"]


def unif(step_size, elow=0.85, ehigh=1.15):
    return np.random.uniform(elow, ehigh) * step_size


class HamiltonianMC(BaseHMC):
    R"""A sampler for continuous variables based on Hamiltonian mechanics.

    See NUTS sampler for automatically tuned stopping time and step size scaling.
    """

    name = "hmc"
    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "step_size": np.float64,
            "n_steps": np.int64,
            "tune": bool,
            "step_size_bar": np.float64,
            "accept": np.float64,
            "diverging": bool,
            "energy_error": np.float64,
            "energy": np.float64,
            "path_length": np.float64,
            "accepted": bool,
            "model_logp": np.float64,
            "process_time_diff": np.float64,
            "perf_counter_diff": np.float64,
            "perf_counter_start": np.float64,
        }
    ]

    def __init__(self, vars=None, path_length=2.0, max_steps=1024, **kwargs):
        """Set up the Hamiltonian Monte Carlo sampler.

        Parameters
        ----------
        vars: list of theano variables
        path_length: float, default=2
            total length to travel
        step_rand: function float -> float, default=unif
            A function which takes the step size and returns an new one used to
            randomize the step size at each iteration.
        step_scale: float, default=0.25
            Initial size of steps to take, automatically scaled down
            by 1/n**(1/4).
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
        target_accept: float, default 0.65
            Adapt the step size such that the average acceptance
            probability across the trajectories are close to target_accept.
            Higher values for target_accept lead to smaller step sizes.
            Setting this to higher values like 0.9 or 0.99 can help
            with sampling from difficult posteriors. Valid values are
            between 0 and 1 (exclusive). Default of 0.65 is from (Beskos et.
            al. 2010, Neal 2011). See Hoffman and Gelman's "The No-U-Turn
            Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
            Carlo" section 3.2 for details.
        gamma: float, default .05
        k: float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0: float > 0, default 10
            Parameter for dual averaging. Higher values slow initial
            adaptation.
        adapt_step_size: bool, default=True
            Whether step size adaptation should be enabled. If this is
            disabled, `k`, `t0`, `gamma` and `target_accept` are ignored.
        max_steps: int
            The maximum number of leapfrog steps.
        model: pymc3.Model
            The model
        **kwargs: passed to BaseHMC
        """
        kwargs.setdefault("step_rand", unif)
        kwargs.setdefault("target_accept", 0.65)
        super().__init__(vars, **kwargs)
        self.path_length = path_length
        self.max_steps = max_steps

    def _hamiltonian_step(self, start, p0, step_size):
        n_steps = max(1, int(self.path_length / step_size))
        n_steps = min(self.max_steps, n_steps)

        energy_change = -np.inf
        state = start
        last = state
        div_info = None
        try:
            for _ in range(n_steps):
                last = state
                state = self.integrator.step(step_size, state)
        except IntegrationError as e:
            div_info = DivergenceInfo("Integration failed.", e, last, None)
        else:
            if not np.isfinite(state.energy):
                div_info = DivergenceInfo("Divergence encountered, bad energy.", None, last, state)
            energy_change = start.energy - state.energy
            if np.isnan(energy_change):
                energy_change = -np.inf
            if np.abs(energy_change) > self.Emax:
                div_info = DivergenceInfo(
                    "Divergence encountered, large integration error.", None, last, state
                )

        accept_stat = min(1, np.exp(energy_change))

        if div_info is not None or np.random.rand() >= accept_stat:
            end = start
            accepted = False
        else:
            end = state
            accepted = True

        stats = {
            "path_length": self.path_length,
            "n_steps": n_steps,
            "accept": accept_stat,
            "energy_error": energy_change,
            "energy": state.energy,
            "accepted": accepted,
            "model_logp": state.model_logp,
        }
        return HMCStepData(end, accept_stat, div_info, stats)

    @staticmethod
    def competence(var, has_grad):
        """Check how appropriate this class is for sampling a random variable."""
        if var.dtype in discrete_types or not has_grad:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE
