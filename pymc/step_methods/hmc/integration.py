#   Copyright 2024 The PyMC Developers
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

from typing import NamedTuple

import numpy as np

from scipy import linalg

from pymc.blocking import RaveledVars
from pymc.step_methods.hmc.quadpotential import QuadPotential


class State(NamedTuple):
    q: RaveledVars
    p: RaveledVars
    v: np.ndarray
    q_grad: np.ndarray
    energy: float
    model_logp: float
    index_in_trajectory: int


class IntegrationError(RuntimeError):
    pass


class CpuLeapfrogIntegrator:
    def __init__(self, potential: QuadPotential, logp_dlogp_func):
        """Leapfrog integrator using CPU."""
        self._potential = potential
        self._logp_dlogp_func = logp_dlogp_func
        self._dtype = self._logp_dlogp_func.dtype
        if self._potential.dtype != self._dtype:
            raise ValueError(
                f"dtypes of potential ({self._potential.dtype}) and logp function ({self._dtype})"
                "don't match."
            )

    def compute_state(self, q: RaveledVars, p: RaveledVars):
        """Compute Hamiltonian functions using a position and momentum."""
        if q.data.dtype != self._dtype or p.data.dtype != self._dtype:
            raise ValueError(f"Invalid dtype. Must be {self._dtype}")

        logp, dlogp = self._logp_dlogp_func(q)

        v = self._potential.velocity(p.data, out=None)
        kinetic = self._potential.energy(p.data, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, dlogp, energy, logp, 0)

    def step(self, epsilon, state):
        """Leapfrog integrator step.

        Half a momentum update, full position update, half momentum update.

        Parameters
        ----------
        epsilon: float, > 0
            step scale
        state: State namedtuple,
            current position data
        out: (optional) State namedtuple,
            preallocated arrays to write to in place

        Returns
        -------
        None if `out` is provided, else a State namedtuple
        """
        try:
            return self._step(epsilon, state)
        except linalg.LinAlgError as err:
            msg = "LinAlgError during leapfrog step."
            raise IntegrationError(msg) from err
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                msg = "Infs or nans in scipy.linalg during leapfrog step."
                raise IntegrationError(msg)
            else:
                raise

    def _step(self, epsilon, state):
        axpy = linalg.blas.get_blas_funcs("axpy", dtype=self._dtype)
        pot = self._potential

        q_new = state.q.data.copy()
        p_new = state.p.data.copy()
        v_new = np.empty_like(q_new)
        q_new_grad = np.empty_like(q_new)

        dt = 0.5 * epsilon

        # p is already stored in p_new
        # p_new = p + dt * q_grad
        axpy(state.q_grad, p_new, a=dt)

        pot.velocity(p_new, out=v_new)
        # q is already stored in q_new
        # q_new = q + epsilon * v_new
        axpy(v_new, q_new, a=epsilon)

        p_new = RaveledVars(p_new, state.p.point_map_info)
        q_new = RaveledVars(q_new, state.q.point_map_info)

        logp = self._logp_dlogp_func(q_new, grad_out=q_new_grad)

        # p_new = p_new + dt * q_new_grad
        axpy(q_new_grad, p_new.data, a=dt)

        kinetic = pot.velocity_energy(p_new.data, v_new)
        energy = kinetic - logp

        return State(
            q_new,
            p_new,
            v_new,
            q_new_grad,
            energy,
            logp,
            state.index_in_trajectory + int(np.sign(epsilon)),
        )
