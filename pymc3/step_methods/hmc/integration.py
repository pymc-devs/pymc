from collections import namedtuple

import numpy as np
from scipy import linalg


State = namedtuple("State", 'q, p, v, q_grad, energy')


class CpuLeapfrogIntegrator(object):
    def __init__(self, ndim, potential, logp_dlogp_func):
        self._ndim = ndim
        self._potential = potential
        self._logp_dlogp_func = logp_dlogp_func
        self._dtype = self._logp_dlogp_func.dtype
        if self._potential.dtype != self._dtype:
            raise ValueError("dtypes of potential and logp function "
                             "don't match."
                             % (self._potential.dtype, self._dtype))

    def compute_state(self, q, p):
        if q.dtype != self._dtype or p.dtype != self._dtype:
            raise ValueError('Invalid dtype. Must be %s' % self._dtype)
        logp, dlogp = self._logp_dlogp_func(q)
        v = self._potential.velocity(p)
        kinetic = self._potential.energy(p, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, dlogp, energy)

    def step(self, epsilon, state, out=None):
        pot = self._potential
        axpy = linalg.blas.get_blas_funcs('axpy', dtype=self._dtype)

        q, p, v, q_grad, energy = state
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
            return State(q_new, p_new, v_new, q_new_grad, energy)
