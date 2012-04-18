'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import fmin_bfgs, fmin_ncg
import numpy as np 

def find_MAP( model, chain_state, disp = False, retall = False):
    """
    moves the chain to the local maximum a posteriori point given a model
    """
    def logp(x):
        return nan_to_high(-(model.evaluate_as_vector(model.project(chain_state, x)))[0])
        
    def grad_logp(x):
        return np.nan_to_num(-(model.evaluate_as_vector(model.project(chain_state, x)))[1])

    x = fmin_bfgs(logp, model.subspace(chain_state), grad_logp, disp = disp, retall = retall, full_output = True)
    
    chain_state = model.project(chain_state, x[0])
    
    if retall:
        return chain_state, x[-1]
    else:
        return chain_state

    
    
def nan_to_high(x):
    x = x.copy() 
    if (x.ndim >= 1):
        x[np.logical_not(np.isfinite(x))] = 1.79769313e+308
        return x
    else: 
        if np.logical_not(np.isfinite(x)): 
            return 1.79769313e+308
        else:
            return x
    