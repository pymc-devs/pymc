'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from numpy import floor
from quadpotential import *
from arraystep import *
from ..core import *
import numpy as np
from scipy.sparse import issparse

__all__ = ['HamiltonianMC']

#TODO:
#add constraint handling via page 37 of Radford's http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html

def unif(step_size, elow = .85, ehigh = 1.15):
    return np.random.uniform(elow, ehigh) * step_size



class HamiltonianMC(ArrayStep):
    def __init__(self, vars, C, step_scale = .25, path_length = 2., is_cov = False, step_rand = unif, state = None, model = None):
        """
        Parameters
        ----------
            vars : list of theano variables
            C : array_like, ndim = {1,2}
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=.25
                Size of steps to take, automatically scaled down by 1/n**(1/4) (defaults to .25)
            path_length : float, default=2
                total length to travel
            is_cov : bool, default=False
                Treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
            step_rand : function float -> float, default=unif
                A function which takes the step size and returns an new one used to randomize the step size at each iteration.
            state
                State object
            model : Model
        """
        model = modelcontext(model)
        n = C.shape[0]

        self.step_size = step_scale / n**(1/4.)

        if issparse(C):
            raise ValueError("Cannot use sparse matrix for scaling without scikits.sparse installed")
        self.potential = quad_potential(C, is_cov)

        self.path_length = path_length
        self.step_rand = step_rand

        if state is None:
            state = SamplerHist()
        self.state = state

        ArrayStep.__init__(self,
                vars, [model.logpc, model.dlogpc(vars)]
                )

    def astep(self, q0, logp, dlogp):


        #randomize step size
        e = self.step_rand(self.step_size)
        nstep = int(floor(self.path_length / self.step_size))

        q = q0
        p = p0 = self.potential.random()

        #use the leapfrog method
        p = p - (e/2) * -dlogp(q) # half momentum update

        for i in range(nstep):
            #alternate full variable and momentum updates
            q = q + e * self.potential.velocity(p)
            if i != nstep - 1:
                p = p - e * -dlogp(q)

        p = p - (e/2) * -dlogp(q)  # do a half step momentum update to finish off

        p = -p

        # - H(q*, p*) + H(q, p) = -H(q, p) + H(q0, p0) = -(- logp(q) + K(p)) + (-logp(q0) + K(p0))
        mr = (-logp(q0)) + self.potential.energy(p0) - ((-logp(q))  + self.potential.energy(p))

        self.state.metrops.append(mr)

        return metrop_select(mr, q, q0)

