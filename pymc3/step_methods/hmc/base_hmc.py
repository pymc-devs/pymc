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

import logging
import time

from collections import namedtuple

import numpy as np

from pymc3.backends.report import SamplerWarning, WarningType
from pymc3.exceptions import SamplingError
from pymc3.model import Point, modelcontext
from pymc3.step_methods import arraystep, step_sizes
from pymc3.step_methods.hmc import integration
from pymc3.step_methods.hmc.quadpotential import QuadPotentialDiagAdapt, quad_potential
from pymc3.theanof import floatX, inputvars
from pymc3.tuning import guess_scaling

logger = logging.getLogger("pymc3")

HMCStepData = namedtuple("HMCStepData", "end, accept_stat, divergence_info, stats")

DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state, state_div")


class BaseHMC(arraystep.GradientSharedStep):
    """Superclass to implement Hamiltonian/hybrid monte carlo."""

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
        **theano_kwargs
    ):
        """Set up Hamiltonian samplers with common structures.

        Parameters
        ----------
        vars: list of theano variables
        scaling: array_like, ndim = {1,2}
            Scaling for momentum distribution. 1d arrays interpreted matrix
            diagonal.
        step_scale: float, default=0.25
            Size of steps to take, automatically scaled down by 1/n**(1/4)
        is_cov: bool, default=False
            Treat scaling as a covariance matrix/vector if True, else treat
            it as a precision matrix/vector
        model: pymc3 Model instance
        blocked: bool, default=True
        potential: Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        **theano_kwargs: passed to theano functions
        """
        self._model = modelcontext(model)

        if vars is None:
            vars = self._model.cont_vars
        vars = inputvars(vars)

        super().__init__(vars, blocked=blocked, model=model, dtype=dtype, **theano_kwargs)

        self.adapt_step_size = adapt_step_size
        self.Emax = Emax
        self.iter_count = 0
        size = self._logp_dlogp_func.size

        self.step_size = step_scale / (size ** 0.25)
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
            point = Point(scaling, model=model)
            scaling = guess_scaling(point, model=model, vars=vars)

        if scaling is not None and potential is not None:
            raise ValueError("Can not specify both potential and scaling.")

        if potential is not None:
            self.potential = potential
        else:
            self.potential = quad_potential(scaling, is_cov)

        self.integrator = integration.CpuLeapfrogIntegrator(self.potential, self._logp_dlogp_func)

        self._step_rand = step_rand
        self._warnings = []
        self._samples_after_tune = 0
        self._num_divs_sample = 0

    def _hamiltonian_step(self, start, p0, step_size):
        """Compute one hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.
        """
        raise NotImplementedError("Abstract method")

    def astep(self, q0):
        """Perform a single HMC iteration."""
        perf_start = time.perf_counter()
        process_start = time.process_time()

        p0 = self.potential.random()
        start = self.integrator.compute_state(q0, p0)

        if not np.isfinite(start.energy):
            model = self._model
            check_test_point = model.check_test_point()
            error_logp = check_test_point.loc[
                (np.abs(check_test_point) >= 1e20) | np.isnan(check_test_point)
            ]
            self.potential.raise_ok(self._logp_dlogp_func._ordering.vmap)
            message_energy = (
                "Bad initial energy, check any log probabilities that "
                "are inf or -inf, nan or very small:\n{}".format(error_logp.to_string())
            )
            warning = SamplerWarning(
                WarningType.BAD_ENERGY,
                message_energy,
                "critical",
                self.iter_count,
            )
            self._warnings.append(warning)
            raise SamplingError("Bad initial energy")

        adapt_step = self.tune and self.adapt_step_size
        step_size = self.step_adapt.current(adapt_step)
        self.step_size = step_size

        if self._step_rand is not None:
            step_size = self._step_rand(step_size)

        hmc_step = self._hamiltonian_step(start, p0, step_size)

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
                    point = self._logp_dlogp_func.array_to_dict(info.state.q)
                if self._num_divs_sample < 100 and info.state_div is not None:
                    point_dest = self._logp_dlogp_func.array_to_dict(info.state_div.q)
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

            self._warnings.append(warning)

        self.iter_count += 1
        if not self.tune:
            self._samples_after_tune += 1

        stats = {
            "tune": self.tune,
            "diverging": bool(hmc_step.divergence_info),
            "perf_counter_diff": perf_end - perf_start,
            "process_time_diff": process_end - process_start,
            "perf_counter_start": perf_start,
        }

        stats.update(hmc_step.stats)
        stats.update(self.step_adapt.stats())

        return hmc_step.end.q, [stats]

    def reset_tuning(self, start=None):
        self.step_adapt.reset()
        self.reset(start=None)

    def reset(self, start=None):
        self.tune = True
        self.potential.reset()

    def warnings(self):
        # list.copy() is not available in python2
        warnings = self._warnings[:]

        # Generate a global warning for divergences
        message = ""
        n_divs = self._num_divs_sample
        if n_divs and self._samples_after_tune == n_divs:
            message = (
                "The chain contains only diverging samples. The model " "is probably misspecified."
            )
        elif n_divs == 1:
            message = (
                "There was 1 divergence after tuning. Increase "
                "`target_accept` or reparameterize."
            )
        elif n_divs > 1:
            message = (
                "There were %s divergences after tuning. Increase "
                "`target_accept` or reparameterize." % n_divs
            )

        if message:
            warning = SamplerWarning(WarningType.DIVERGENCES, message, "error")
            warnings.append(warning)

        warnings.extend(self.step_adapt.warnings())
        return warnings
