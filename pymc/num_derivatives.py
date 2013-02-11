'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numdifftools as nd
import numpy as np 
from core import *

def approx_hess(model, start, vars=None):
    """
    Returns an approximation of the Hessian at the current chain location.
    
    Parameters
    ----------
    model : Model
    start : dict
    vars : list or array
        Variables for which Hessian is to be calculated.
    """
    if vars is None :
        vars = continuous_vars(model)

    bij = DictArrBij(IdxMap(vars), start)
    dlogp = bij.mapf(model_dlogp(model, vars))

    
    def grad_logp(point): 
        return np.nan_to_num(-dlogp(point))
    
    '''
    Find the jacobian of the gradient function at the current position
    this should be the Hessian; invert it to find the approximate 
    covariance matrix.
    '''
    return nd.Jacobian(grad_logp)(bij.map(start))
