'''
Created on Mar 12, 2011

from __future__ import division
@author: johnsalvatier
'''
import numdifftools as nd
import numpy as np
from numpy import exp, log, sqrt
from ..core import *

__all__ = ['approx_hess', 'find_hessian', 'trace_cov', 'guess_scaling']


def approx_hess(point, vars=None, model=None):
    """
    Returns an approximation of the Hessian at the current chain location.

    Parameters
    ----------
    model : Model (optional if in `with` context)
    point : dict
    vars : list
        Variables for which Hessian is to be calculated.
    """
    model = modelcontext(model)
    if vars is None:
        vars = model.cont_vars

    point = Point(point, model=model)

    bij = DictToArrayBijection(ArrayOrdering(vars), point)
    dlogp = bij.mapf(model.dlogpc(vars))

    def grad_logp(point):
        return np.nan_to_num(dlogp(point))

    '''
    Find the jacobian of the gradient function at the current position
    this should be the Hessian; invert it to find the approximate
    covariance matrix.
    '''
    return -nd.Jacobian(grad_logp)(bij.map(point))


def find_hessian(point, vars=None, model=None):
    """
    Returns Hessian of logp at the point passed.

    Parameters
    ----------
    model : Model (optional if in `with` context)
    point : dict
    vars : list
        Variables for which Hessian is to be calculated.
    """
    H = model.d2logpc(vars)
    return H(Point(point, model=model))


def guess_scaling(point=None, vars = None, model=None):
    model = modelcontext(model)
    H = find_hessian(point,vars,model)

    val, vec = np.linalg.eigh(H)

    mag = sqrt(abs(val))

    bounded = bound(log(mag), log(1e-10), log(1e10))
    val = exp(bounded)**2

    return eig_recompose(val,vec)

def bound(a, l, u):
    return np.maximum(np.minimum(a,u), l)
def soft_bound(a, l, u):
    d = u - l
    x = (a-l)/d + .5

    return d*(1./(1+exp(-x))) + l


def eig_recompose(val, vec):
    return vec.dot(np.diag(val)).dot(vec.T)


def trace_cov(trace, vars=None):
    """
    Calculate the flattened covariance matrix using a sample trace

    Useful if you want to base your covariance matrix for further sampling on some initial samples.

    Parameters
    ----------
    trace : Trace
    vars : list
        variables for which to calculate covariance matrix

    Returns
    -------
    r : array (n,n)
        covariance matrix
    """

    if vars is None:
        vars = trace.samples.keys

    def flat_t(var):
        x = trace[str(var)]
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))

    return np.cov(np.concatenate(map(flat_t, vars), 1).T)
