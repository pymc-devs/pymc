'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from numpy import floor
from .quadpotential import quad_potential
from .arraystep import ArrayStep, SamplerHist, metrop_select, Competence
from ..tuning import guess_scaling
from ..model import modelcontext, Point
from ..theanof import inputvars
from ..vartypes import discrete_types

import numpy as np
from scipy.sparse import issparse

from collections import namedtuple

__all__ = ['HamiltonianMC']

# TODO:
# add constraint handling via page 37 of Radford's
# http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html


def unif(step_size, elow=.85, ehigh=1.15):
    return np.random.uniform(elow, ehigh) * step_size


class HamiltonianMC(ArrayStep):
    default_blocked = True

    def __init__(self, vars=None, scaling=None, step_scale=.25, path_length=2., is_cov=False, step_rand=unif, state=None, model=None, **kwargs):
        """
        Parameters
        ----------
            vars : list of theano variables
            scaling : array_like, ndim = {1,2}
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=.25
                Size of steps to take, automatically scaled down by 1/n**(1/4) (defaults to .25)
            path_length : float, default=2
                total length to travel
            is_cov : bool, default=False
                Treat scaling as a covariance matrix/vector if True, else treat it as a precision matrix/vector
            step_rand : function float -> float, default=unif
                A function which takes the step size and returns an new one used to randomize the step size at each iteration.
            state
                State object
            model : Model
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        if scaling is None:
            scaling = model.test_point

        if isinstance(scaling, dict):
            scaling = guess_scaling(Point(scaling, model=model), model=model)

        n = scaling.shape[0]

        self.step_size = step_scale / n ** (1 / 4.)

        self.potential = quad_potential(scaling, is_cov, as_cov=False)

        self.path_length = path_length
        self.step_rand = step_rand

        if state is None:
            state = SamplerHist()
        self.state = state

        super(HamiltonianMC, self).__init__(
            vars, [model.fastlogp, model.fastdlogp(vars)], **kwargs)

    def astep(self, q0, logp, dlogp):
        H = Hamiltonian(logp, dlogp, self.potential)

        e = self.step_rand(self.step_size)
        nstep = int(self.path_length / e)

        p0 = H.pot.random()

        q, p = leapfrog(H, q0, p0, nstep, e)
        p = -p

        mr = energy(H, q0, p0) - energy(H, q, p)

        self.state.metrops.append(mr)

        return metrop_select(mr, q, q0)

    @staticmethod
    def competence(var):
        if var.dtype in discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


def bern(p):
    return np.random.uniform() < p

Hamiltonian = namedtuple("Hamiltonian", "logp, dlogp, pot")


def energy(H, q, p):
    return -(H.logp(q) - H.pot.energy(p))


def leapfrog(H, q, p, n, e):
    _, dlogp, pot = H

    p = p - (e / 2) * -dlogp(q)  # half momentum update

    for i in range(n):
        # alternate full variable and momentum updates
        q = q + e * pot.velocity(p)

        if i != n - 1:
            p = p - e * -dlogp(q)

    p = p - (e / 2) * -dlogp(q)  # do a half step momentum update to finish off
    return q, p
