'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from numpy.linalg import cholesky

from ..core import *
from quadpotential import quad_potential

from arraystep import *

__all__ = ['Metropolis']

# TODO Implement tuning for Metropolis step
class Metropolis(ArrayStep):
    def __init__(self, model, vars, C, scaling=.25):

        self.potential = quad_potential(C*scaling, False)
        super(Metropolis,self).__init__(vars, [model.logpc])
        
    def astep(self, state, q0, logp):

        delta = self.potential.random()
        
        q = q0 + delta  
        
        return state, metrop_select(logp(q) - logp(q0),
                                    q, q0)
