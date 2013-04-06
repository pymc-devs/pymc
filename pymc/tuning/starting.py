'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy import optimize
import numpy as np 
from numpy import isfinite, nan_to_num
from ..core import *

__all__ = ['find_MAP', 'scipyminimize']


@withmodel
def find_MAP(model, start = None, vars=None, fmin = optimize.fmin_bfgs, return_raw = False, disp = False, *args, **kwargs):
    """
    Sets state to the local maximum a posteriori point given a model.
    Current default of fmin_Hessian does not deal well with optimizing close 
    to sharp edges, especially if they are the minimum.
    
    Parameters
    ----------
    model : Model
    start : dict of parameter values (Defaults to model.test_point)
    vars : list or array
        List of variables to set to MAP point (Defaults to all continuous).
    fmin : function
        Optimization algorithm (Defaults to `scipy.optimize.fmin_l_bfgs_b`).
    return_raw : Bool
        Whether to return extra value returned by fmin (Defaults to False)
    *args, **kwargs 
        Extra args passed to fmin
    """
    if start is None:
        start = model.test_point

    if vars is None: 
        vars = model.cont_vars
        
    start = Point(start)
    bij = DictToArrayBijection(ArrayOrdering(vars), start)
    
    logp = bij.mapf(model.logpc)
    dlogp = bij.mapf(model.dlogpc(vars))
    
    def logp_o(point):
        return nan_to_high(-logp(point))
        
    def grad_logp_o(point):
        return nan_to_num(-dlogp(point))

    r = fmin(logp_o, bij.map(start), fprime = grad_logp_o, disp = disp, *args, **kwargs)
    if isinstance(r, tuple):
        mx = r[0]
    else: 
        mx = r    
   

    if (not allfinite(mx) or
       not allfinite(logp(mx)) or
       not allfinite(dlogp(mx))) :
            raise ValueError("Optimization error: max, logp or dlogp at max have bad values. max: " + repr(mx) + " logp: " + repr(logp(mx)) + " dlogp: " + repr(dlogp(mx)))
    
    mx = bij.rmap(mx)
    if return_raw: 
        return mx, r
    else:
        return mx

def allfinite(x):
    return np.all(isfinite(x))

def nan_to_high(x):
    return np.where(isfinite(x), x, 1.0e100)  


def scipyminimize(f, x0, fprime, *args, **kwargs):
    r = scipy.optimize.minimize(f, x0, jac = fprime, *args, **kwargs)
    return r.x, r
    

    

