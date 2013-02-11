'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy.optimize import fmin_bfgs, fmin_ncg
import numpy as np 
from numpy import isfinite, nan_to_num
from core import *

def find_MAP(model, start, vars=None, min_alg=fmin_bfgs, disp=False, return_all=False):
    """
    Sets state to the local maximum a posteriori point given a model.
    Current default of fmin_Hessian does not deal well with optimizing close 
    to sharp edges, especially if they are the minimum.
    
    Parameters
    ----------
    model : Model
    start : array
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
        
    start = clean_point(start)
    bij = DictArrBij(IdxMap(vars), start)
    
    logp = bij.mapf(model_logp(model))
    dlogp = bij.mapf(model_dlogp(model, vars))
    
    def logp_o(point):
        return nan_to_high(-logp(point))
        
    def grad_logp_o(point):
        return nan_to_num(-dlogp(point))

    results = min_alg(logp_o, bij.map(start), grad_logp_o, disp=disp, retall=return_all, full_output=True)
    mx = results[0]
    log = results[-1]

    if not np.all(isfinite(mx)):
        raise ValueError("Optimization error, found optima has some bad values " + repr(mx)) 

    if not isfinite(logp(mx)) or not np.all(isfinite(dlogp(mx))):
        raise ValueError("Optimization error, logp or dlogp at max have bad values. logp: " + repr(logp(mx)) + " dlogp: " + repr(dlogp(mx)))

    
    max_point = bij.rmap(mx)
    
    if return_all:
        return max_point, log
    else:
        return max_point

    
def nan_to_high(x):
    return np.where(isfinite(x), x, 1.0e100)  
