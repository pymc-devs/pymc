'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import fmin_bfgs, fmin_ncg
import numpy as np 

def find_MAP( model, chain_state, disp = False):
    """
    moves the chain to the local maximum a posteriori point given a model
    """
    def logp(x):
        
        return np.nan_to_num(-(model.evaluate_as_vector(model.project(chain_state, x)))[0])
        
    def grad_logp(x):
        return np.nan_to_num(-(model.evaluate_as_vector(model.project(chain_state, x)))[1])

    
    x = fmin_ncg(logp, model.subspace(chain_state), grad_logp, disp = disp)
    return model.project(chain_state, x)

    