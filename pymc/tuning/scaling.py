'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numdifftools as nd
import numpy as np 
from ..core import *

__all__ = ['approx_hess']
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
        vars = model.cont_vars

    start = clean_point(start)

    bij = DictToArrayBijection(ArrayOrdering(vars), start)
    dlogp = bij.mapf(model.dlogpc(vars))

    
    def grad_logp(point): 
        return np.nan_to_num(dlogp(point))
    
    '''
    Find the jacobian of the gradient function at the current position
    this should be the Hessian; invert it to find the approximate 
    covariance matrix.
    '''
    return -nd.Jacobian(grad_logp)(bij.map(start))


def trace_cov(trace, vars = None):
    """
    Calculate the flattened covariance matrix using a sample trace

    Useful if you want to base your covariance on some initial samples.
    """

    if vars is None: 
        vars = trace.samples.keys

    def flat_t(var):
        x = trace[str(var)]
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))
    
    return np.cov(np.concatenate(map(flat_t, vars), 1).T)
