'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import fmin_bfgs, fmin_ncg
import numpy as np 
from core import *

def find_MAP(model, chain_state, vars=None, min_alg=fmin_bfgs, disp=False, return_all=False):
    """
    Sets state to the local maximum a posteriori point given a model.
    Current default of fmin_Hessian does not deal well with optimizing close 
    to sharp edges, especially if they are the minimum.
    
    Parameters
    ----------
    model : Model
    chain_state : array
    vars : list or array
        List of variables to set to MAP point (Defaults to all).
    min_alg : function
        Optimization algorithm (Defaults to `fmin_bfgs`).
    disp : bool
        Print convergence message if True (Defaults to False).
    return_all : bool
        Return a list of results at each iteration if True (Defaults 
        to False).
    """
    if vars is None: 
        vars = continuous_vars(model)
        
    mapping = DASpaceMap(vars)
    
    logp = model_logp(model)
    dlogp = model_dlogp(model, vars)
    
    def logp_o(x):
        return nan_to_high(-logp(mapping.rproject(x, chain_state)))
        
    def grad_logp_o(x):
        return np.nan_to_num(-dlogp(mapping.rproject(x, chain_state)))

    x = min_alg(logp_o, mapping.project(chain_state), grad_logp_o, disp=disp, retall=return_all, full_output=True)
    
    chain_state = mapping.rproject(x[0], chain_state)
    
    if return_all:
        return chain_state, x[-1]
    else:
        return chain_state

    
    
def nan_to_high(x):
    # Converts NaN values to largest integer
    x = x.copy() 
    if (x.ndim >= 1):
        x[np.logical_not(np.isfinite(x))] = 1.79769313e+308
        return x
    else: 
        if np.logical_not(np.isfinite(x)): 
            return 1.79769313e+100
        else:
            return x
    