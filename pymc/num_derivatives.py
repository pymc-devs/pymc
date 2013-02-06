'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numdifftools as nd
import numpy as np 
from core import *

def approx_hess(model, chain_state, vars=None):
    """
    Returns an approximation of the Hessian at the current chain location.
    
    Parameters
    ----------
    model : Model
    chain_state : array
    vars : list or array
        Variables for which Hessian is to be calculated.
    """
    if vars is None :
        vars = continuous_vars(model)
    dlogp = model_dlogp(model, vars)
    mapping = DASpaceMap(vars)
    
    def grad_logp(x): 
        return np.nan_to_num(-dlogp(mapping.rproject(x, chain_state)))
    
    '''
    Find the jacobian of the gradient function at the current position
    this should be the Hessian; invert it to find the approximate 
    covariance matrix.
    '''
    return nd.Jacobian(grad_logp)(mapping.project(chain_state))