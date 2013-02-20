'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import theano.tensor as t
from theano.tensor import switch, log, eq, neq, lt, gt, le, ge, zeros_like, cast, round

from numpy import pi, inf, nan
from special import gammaln, factln

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

def bound(logp, *conds):
    cond = 1
    for c in conds:
        cond = cond & (1*c)

    return switch(cond, logp, -inf)
