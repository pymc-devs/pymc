from numpy.linalg import cholesky

from ..core import *

from utils import *

# TODO Implement tuning for Metropolis step
def metropolis_step(model, vars, C, scaling=0.25):
    logp_d = model_logp(model)
        
    cholC = cholesky(scaling * C)
        
    def step(logp, state, q0):

        delta = cholesky_normal(q0.shape, cholC)
        
        q = q0 + delta  
        
        return state, metrop_select(logp(q) - logp(q0),
                                    q, q0)
        
    return array_step(step, logp_d, vars)
        
