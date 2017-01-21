'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from ..arraystep import metrop_select, Competence
from .base_hmc import BaseHMC
from pymc3.vartypes import discrete_types
from pymc3.theanof import floatX
import numpy as np


__all__ = ['HamiltonianMC']

# TODO:
# add constraint handling via page 37 of Radford's
# http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html


def unif(step_size, elow=.85, ehigh=1.15):
    return np.random.uniform(elow, ehigh) * step_size


class HamiltonianMC(BaseHMC):
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
        **kwargs : passed to BaseHMC
        """
        super(HamiltonianMC, self).__init__(vars, **kwargs)
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
        return metrop_select(energy_change, q, q0)

    @staticmethod
    def competence(var):
        if var.dtype in discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE
