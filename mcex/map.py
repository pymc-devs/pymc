'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import fmin_bfgs, fmin_ncg
import numpy as np 
from core import *

def find_MAP( model, chain_state, vars = None, minalg = fmin_bfgs, disp = False, retall = False):
    """
    Moves the chain to the local maximum a posteriori point given a model.
    Current default of fmin_bfgs does not deal well with optimizing close to sharp edges, especially if they are the minimum.
    """
    if vars is None: 
        vars = model.vars 
        
    mapping = DASpaceMap(vars)
    
    logp = model_logp(model)
    dlogp = model_dlogp(model, vars)
    
    def logp_o(x):
        return nan_to_high(-logp(mapping.rproject(x, chain_state)))
        
    def grad_logp_o(x):
        return np.nan_to_num(-dlogp(mapping.rproject(x, chain_state)))

    x = minalg(logp_o, mapping.project(chain_state), grad_logp_o, disp = disp, retall = retall, full_output = True)
    
    chain_state = mapping.rproject(x[0], chain_state)
    
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
            return 1.79769313e+200
        else:
            return x
    