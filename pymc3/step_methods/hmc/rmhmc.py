from ..arraystep import metrop_select, Competence
from .base_hmc import BaseHMC
from pymc3.vartypes import discrete_types
from pymc3.theanof import floatX
import numpy as np


__all__ = ['RiemannianManifoldHMC']

def unif(step_size, elow=.85, ehigh=1.15):
    return np.random.uniform(elow, ehigh) * step_size

class RiemannianManifoldHMC(BaseHMC):
    name = 'rmhmc'
    default_blocked = True

    def __init__(self, vars=None, path_length=2., step_rand=unif, **kwargs):
        """
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
        model : pymc3.Model
            The model
        **kwargs : passed to BaseHMC
        """
        super(RiemannianManifoldHMC, self).__init__(vars, manifold=True, **kwargs)
        self.path_length = path_length
        self.step_rand = step_rand

    def astep(self, q0):
        e = floatX(self.step_rand(self.step_size))
        n_steps = np.array(self.path_length / e, dtype='int32')
        q = q0
        p = self.H.pot.random()  # initialize momentum
        initial_energy = self.compute_energy(q, p)
        q, p, current_energy = self.leapfrog(q, p, e, n_steps)
        energy_change = initial_energy - current_energy
        return metrop_select(energy_change, q, q0)[0]

    @staticmethod
    def competence(var):
        if var.dtype in discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE
        