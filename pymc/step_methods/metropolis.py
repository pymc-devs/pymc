'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from numpy.linalg import cholesky

from ..core import *

from utils import *

# TODO Implement tuning for Metropolis step
class metropolis_step(array_step):
    def __init__(self, model, vars, C, scaling=.25):
        logp_d = model_logp(model)
        
        self.cholC = cholesky(scaling * C)
        super(metropolis_step,self).__init__(vars, [logp_d])
        
    def astep(self, state, q0, logp):

        delta = cholesky_normal(q0.shape, self.cholC)
        
        q = q0 + delta  
        
        return state, metrop_select(logp(q) - logp(q0),
                                    q, q0)
