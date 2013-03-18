'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division
import theano.tensor as t
from theano.tensor import sum, switch, log, eq, neq, lt, gt, le, ge, zeros_like, cast,arange, round, max, min

from numpy import pi, inf, nan
from special import gammaln

from theano.printing import Print





from functools import wraps

def quickclass(fn): 
    class Distribution(object):
        __doc__ = fn.__doc__

        @wraps(fn)
        def __init__(self, *args, **kwargs):  #still need to figure out how to give it the right argument names
                properties = fn(*args, **kwargs) 
                self.__dict__.update(properties)


    Distribution.__name__ = fn.__name__
    return Distribution

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

    return switch(alltrue(conditions), logp, -inf)

def alltrue(vals):
    ret = 1
    for c in vals:
        ret = ret & (1*c)
    return ret
    

def logpow(x, m):
    """
    Calculates log(x**m) since m*log(x) will fail when m, x = 0.
    """
    return switch( eq(x,0) & eq(m,0), 0, m*log(x))

def factln(n):
    return gammaln(n +1)
