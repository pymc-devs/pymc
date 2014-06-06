'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division
import theano.tensor as t
from theano.tensor import (
    sum, switch, log, exp, sqrt,
    eq, neq, lt, gt, le, ge, all, any,
    cast, round, arange, max, min,
    maximum, minimum, floor, ceil,
    zeros_like, ones, ones_like,
    concatenate, constant, argmax)

import theano


from numpy import pi, inf, nan, float64, array
from .special import gammaln, multigammaln
from theano.ifelse import ifelse
from theano.printing import Print
from .distribution import *

def guess_is_scalar(c):
    bcast = None
    if (isinstance(c, t.TensorType)):
        return c.broadcastable == ()
    
    if (isinstance(c, t.Variable) or isinstance(c, t.Constant)):
        return c.type.broadcastable == ()
    
    tt = t.as_tensor_variable(c)
    return tt.type.broadcastable == ()

impossible = constant(-inf, dtype=float64) # We re-use that constant

def bound_scalar(logp, *conditions):
    """
    Bounds a log probability density with several conditions

    Parameters
    ----------
    logp : float
    *conditions : booleans

    Returns
    -------
    logp if all conditions are true
    -inf if some are false
    """        
    cond = alltrue(conditions)
    return ifelse(cond, logp, impossible)

  
def bound(logp, *conditions):
    """
    Bounds a log probability density with several conditions

    Parameters
    ----------
    logp : float
    *conditionss : booleans

    Returns
    -------
    logp if all conditions are true
    -inf if some are false
    """
    if (guess_is_scalar(logp)):
        for c in conditions:
            if (not guess_is_scalar(c)):
                break
            return bound_scalar(logp, *conditions)
    return switch(alltrue(conditions), logp, impossible)

def alltrue(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret


def logpow(x, m):
    """
    Calculates log(x**m) since m*log(x) will fail when m, x = 0.
    """
    return switch(eq(x, 0) & eq(m, 0), 0, m * log(x))

def logpow_relaxed(x, m):
    """
    Calculates log(x**m), unless m==0 or x==0 in which case it will return 0 
    """
    return switch(eq(x, 0) | eq(m, 0), 0, m * log(x))

def factln(n):
    return gammaln(n + 1)


def idfn(x):
    return x
