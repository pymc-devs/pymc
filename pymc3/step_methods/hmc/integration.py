from collections import namedtuple

import numpy as np
from scipy import linalg


State = namedtuple("State", 'q, p, v, q_grad, energy, model_logp')


class IntegrationError(RuntimeError):
    pass


class CpuLeapfrogIntegrator:
    def __init__(self, potential, logp_dlogp_func):
        """Leapfrog integrator using CPU."""
        self._potential = potential
        self._logp_dlogp_func = logp_dlogp_func
        self._dtype = self._logp_dlogp_func.dtype
        if self._potential.dtype != self._dtype:
            raise ValueError("dtypes of potential (%s) and logp function (%s)"
                             "don't match."
                             % (self._potential.dtype, self._dtype))

    def compute_state(self, q, p):
        """Compute Hamiltonian functions using a position and momentum."""
        if q.dtype != self._dtype or p.dtype != self._dtype:
            raise ValueError('Invalid dtype. Must be %s' % self._dtype)
        logp, dlogp = self._logp_dlogp_func(q)
        v = self._potential.velocity(p)
        kinetic = self._potential.energy(p, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, dlogp, energy, logp)

    def step(self, epsilon, state, out=None):
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
            return self._step(epsilon, state, out=None)
        except linalg.LinAlgError as err:
            msg = "LinAlgError during leapfrog step."
            raise IntegrationError(msg)
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                msg = "Infs or nans in scipy.linalg during leapfrog step."
                raise IntegrationError(msg)
            else:
                raise

    def _step(self, epsilon, state, out=None):
        pot = self._potential
        axpy = linalg.blas.get_blas_funcs('axpy', dtype=self._dtype)

        q, p, v, q_grad, energy, logp = state
        if out is None:
            q_new = q.copy()
            p_new = p.copy()
            v_new = np.empty_like(q)
            q_new_grad = np.empty_like(q)
        else:
            q_new, p_new, v_new, q_new_grad, energy = out
            q_new[:] = q
            p_new[:] = p

        dt = 0.5 * epsilon

        # p is already stored in p_new
        # p_new = p + dt * q_grad
        axpy(q_grad, p_new, a=dt)

        pot.velocity(p_new, out=v_new)
        # q is already stored in q_new
        # q_new = q + epsilon * v_new
        axpy(v_new, q_new, a=epsilon)

        logp = self._logp_dlogp_func(q_new, q_new_grad)

        # p_new = p_new + dt * q_new_grad
        axpy(q_new_grad, p_new, a=dt)

        kinetic = pot.velocity_energy(p_new, v_new)
        energy = kinetic - logp

        if out is not None:
            out.energy = energy
            return
        else:
            return State(q_new, p_new, v_new, q_new_grad, energy, logp)
