import numpy as np
import math

from ..arraystep import metrop_select, Competence
from .base_hmc import BaseHMC
from pymc3.vartypes import discrete_types
from pymc3.theanof import floatX
from pymc3.step_methods import step_size
from scipy import linalg


__all__ = ['HamiltonianMC']

# TODO:
# add constraint handling via page 37 of Radford's
# http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html


def unif(step_size, elow=.85, ehigh=1.15):
    return np.random.uniform(elow, ehigh) * step_size


class HamiltonianMC(BaseHMC):
    R"""A sampler for continuous variables based on Hamiltonian mechanics.

    See NUTS sampler for automatically tuned stopping time and step size scaling.
    """

    name = 'hmc'
    default_blocked = True

    def __init__(self, vars=None, path_length=2., step_rand=unif,
                 adapt_step_size=True, gamma=0.05, k=0.75, t0=10,
                 target_accept=0.8, **kwargs):
        """Set up the Hamiltonian Monte Carlo sampler.

        Parameters
        ----------
        vars : list of theano variables
        path_length : float, default=2
            total length to travel
        step_rand : function float -> float, default=unif
            A function which takes the step size and returns an new one used to
            randomize the step size at each iteration.
        step_scale : float, default=0.25
            Initial size of steps to take, automatically scaled down
            by 1/n**(1/4).
        scaling : array_like, ndim = {1,2}
            The inverse mass, or precision matrix. One dimensional arrays are
            interpreted as diagonal matrices. If `is_cov` is set to True,
            this will be interpreded as the mass or covariance matrix.
        is_cov : bool, default=False
            Treat the scaling as mass or covariance matrix.
        potential : Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods. It can be specified instead
            of the scaling matrix.
        target_accept : float (0,1), default .8
            Try to find a step size such that the average acceptance
            probability across the trajectories are close to target_accept.
            Higher values for target_accept lead to smaller step sizes.
        gamma : float, default .05
        k : float (.5,1) default .75
            scaling of speed of adaptation
        t0 : int, default 10
            slows initial adaptation
        adapt_step_size : bool, default=True
            Whether step size adaptation should be enabled. If this is
            disabled, `k`, `t0`, `gamma` and `target_accept` are ignored.
        model : pymc3.Model
            The model
        **kwargs : passed to BaseHMC
        """
        super(HamiltonianMC, self).__init__(vars, **kwargs)
        self.path_length = path_length
        self.step_rand = step_rand
        self.step_adapt = step_size.DualAverageAdaptation(
            self.step_size, target_accept, gamma, k, t0)
        self.adapt_step_size = adapt_step_size
        self.tune = True

    def astep(self, q0):
        """Perform a single HMC iteration."""
        if self.adapt_step_size:
            step_size = self.step_adapt.current(self.tune)
        else:
            step_size = self.step_size
        if self.step_rand is not None:
            step_size = self.step_rand(step_size)
        path_length = np.random.rand() * self.path_length
        n_steps = max(1, int(path_length / step_size))

        p0 = self.potential.random()
        start = self.integrator.compute_state(q0, p0)

        if not np.isfinite(start.energy):
            raise ValueError('Bad initial energy: %s. The model '
                             'might be misspecified.' % start.energy)

        energy_change = -np.inf
        state = start
        try:
            for _ in range(n_steps):
                state = self.integrator.step(step_size, state)
        except linalg.LinAlgError as err:
            error_msg = "LinAlgError during leapfrog step."
            error = err
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                error_msg = "Infs or nans in scipy.linalg during leapfrog step."
                error = err
            else:
                raise
        else:
            energy_change = start.energy - state.energy

        if self.tune and self.adapt_step_size:
            self.step_adapt.update(min(1, math.exp(energy_change)))
        return metrop_select(energy_change, state.q, start.q)[0]

    @staticmethod
    def competence(var, has_grad):
        """Check how appropriate this class is for sampling a random variable."""
        if var.dtype in discrete_types or not has_grad:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE
