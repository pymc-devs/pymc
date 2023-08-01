#   Copyright 2023 The PyMC Developers
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

import logging
import time

from abc import abstractmethod
from typing import Any, NamedTuple

import numpy as np

from pymc.blocking import DictToArrayBijection, RaveledVars, StatsType
from pymc.exceptions import SamplingError
from pymc.model import Point, modelcontext
from pymc.pytensorf import floatX
from pymc.stats.convergence import SamplerWarning, WarningType
from pymc.step_methods import step_sizes
from pymc.step_methods.arraystep import GradientSharedStep
from pymc.step_methods.hmc import integration
from pymc.step_methods.hmc.integration import IntegrationError, State
from pymc.step_methods.hmc.quadpotential import QuadPotentialDiagAdapt, quad_potential
from pymc.tuning import guess_scaling
from pymc.util import get_value_vars_from_user_vars

logger = logging.getLogger(__name__)


class DivergenceInfo(NamedTuple):
    message: str
    exec_info: IntegrationError | None
    state: State
    state_div: State | None


class HMCStepData(NamedTuple):
    end: State
    accept_stat: int
    divergence_info: DivergenceInfo | None
    stats: dict[str, Any]


class BaseHMC(GradientSharedStep):
    """Superclass to implement Hamiltonian/hybrid monte carlo."""

    integrator: integration.CpuLeapfrogIntegrator
    default_blocked = True

    def __init__(
        self,
        vars=None,
        scaling=None,
        step_scale=0.25,
        is_cov=False,
        model=None,
        blocked=True,
        potential=None,
        dtype=None,
        Emax=1000,
        target_accept=0.8,
        gamma=0.05,
        k=0.75,
        t0=10,
        adapt_step_size=True,
        step_rand=None,
        **pytensor_kwargs,
    ):
        """Set up Hamiltonian samplers with common structures.

        Parameters
        ----------
        vars: list, default=None
            List of PyTensor variables. If None, all continuous RVs from the
            model are included.
        scaling: array_like, ndim={1,2}
            Scaling for momentum distribution. 1d arrays interpreted matrix
            diagonal.
        step_scale: float, default=0.25
            Size of steps to take, automatically scaled down by 1/n**(1/4),
            where n is the dimensionality of the parameter space
        is_cov: bool, default=False
            Treat scaling as a covariance matrix/vector if True, else treat
            it as a precision matrix/vector
        model: pymc.Model
        blocked: bool, default=True
        potential: Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        **pytensor_kwargs: passed to PyTensor functions
        """
        self._model = modelcontext(model)

        if vars is None:
            vars = self._model.continuous_value_vars
        else:
            vars = get_value_vars_from_user_vars(vars, self._model)
        super().__init__(vars, blocked=blocked, model=self._model, dtype=dtype, **pytensor_kwargs)

        self.adapt_step_size = adapt_step_size
        self.Emax = Emax
        self.iter_count = 0

        # We're using the initial/test point to determine the (initial) step
        # size.
        # XXX: If the dimensions of these terms change, the step size
        # dimension-scaling should change as well, no?
        test_point = self._model.initial_point()

        nuts_vars = [test_point[v.name] for v in vars]
        size = sum(v.size for v in nuts_vars)

        self.step_size = step_scale / (size**0.25)
        self.step_adapt = step_sizes.DualAverageAdaptation(
            self.step_size, target_accept, gamma, k, t0
        )
        self.target_accept = target_accept
        self.tune = True

        if scaling is None and potential is None:
            mean = floatX(np.zeros(size))
            var = floatX(np.ones(size))
            potential = QuadPotentialDiagAdapt(size, mean, var, 10)

        if isinstance(scaling, dict):
            point = Point(scaling, model=self._model)
            scaling = guess_scaling(point, model=self._model, vars=vars)

        if scaling is not None and potential is not None:
            raise ValueError("Can not specify both potential and scaling.")

        if potential is not None:
            self.potential = potential
        else:
            self.potential = quad_potential(scaling, is_cov)

        self.integrator = integration.CpuLeapfrogIntegrator(self.potential, self._logp_dlogp_func)

        self._step_rand = step_rand
        self._num_divs_sample = 0

    @abstractmethod
    def _hamiltonian_step(self, start, p0, step_size) -> HMCStepData:
        """Compute one Hamiltonian trajectory and return the next state.

        Subclasses must overwrite this abstract method and return an `HMCStepData` object.
        """

    def astep(self, q0: RaveledVars) -> tuple[RaveledVars, StatsType]:
        """Perform a single HMC iteration."""
        perf_start = time.perf_counter()
        process_start = time.process_time()

        p0 = self.potential.random()
        p0 = RaveledVars(p0, q0.point_map_info)

        start = self.integrator.compute_state(q0, p0)

        warning: SamplerWarning | None = None
        if not np.isfinite(start.energy):
            model = self._model
            check_test_point_dict = model.point_logps()
            check_test_point = np.asarray(list(check_test_point_dict.values()))
            error_logp = check_test_point[
                (np.abs(check_test_point) >= 1e20) | np.isnan(check_test_point)
            ]
            self.potential.raise_ok(q0.point_map_info)
            message_energy = (
                "Bad initial energy, check any log probabilities that "
                f"are inf or -inf, nan or very small:\n{error_logp}"
            )
            warning = SamplerWarning(
                WarningType.BAD_ENERGY,
                message_energy,
                "critical",
                self.iter_count,
            )
            raise SamplingError(f"Bad initial energy: {warning}")

        adapt_step = self.tune and self.adapt_step_size
        step_size = self.step_adapt.current(adapt_step)
        self.step_size = step_size

        if self._step_rand is not None:
            step_size = self._step_rand(step_size)

        hmc_step = self._hamiltonian_step(start, p0.data, step_size)

        perf_end = time.perf_counter()
        process_end = time.process_time()

        self.step_adapt.update(hmc_step.accept_stat, adapt_step)
        self.potential.update(hmc_step.end.q, hmc_step.end.q_grad, self.tune)
        if hmc_step.divergence_info:
            info = hmc_step.divergence_info
            point = None
            point_dest = None
            info_store = None
            if self.tune:
                kind = WarningType.TUNING_DIVERGENCE
            else:
                kind = WarningType.DIVERGENCE
                self._num_divs_sample += 1
                # We don't want to fill up all memory with divergence info
                if self._num_divs_sample < 100 and info.state is not None:
                    point = DictToArrayBijection.rmap(info.state.q)

                if self._num_divs_sample < 100 and info.state_div is not None:
                    point_dest = DictToArrayBijection.rmap(info.state_div.q)

                if self._num_divs_sample < 100:
                    info_store = info
            warning = SamplerWarning(
                kind,
                info.message,
                "debug",
                self.iter_count,
                info.exec_info,
                divergence_point_source=point,
                divergence_point_dest=point_dest,
                divergence_info=info_store,
            )

        self.iter_count += 1

        stats: dict[str, Any] = {
            "tune": self.tune,
            "diverging": bool(hmc_step.divergence_info),
            "perf_counter_diff": perf_end - perf_start,
            "process_time_diff": process_end - process_start,
            "perf_counter_start": perf_start,
            "warning": warning,
        }

        stats.update(hmc_step.stats)
        stats.update(self.step_adapt.stats())
        stats.update(self.potential.stats())

        return hmc_step.end.q, [stats]

    def reset_tuning(self, start=None):
        self.step_adapt.reset()
        self.reset(start=None)

    def reset(self, start=None):
        self.tune = True
        self.potential.reset()
