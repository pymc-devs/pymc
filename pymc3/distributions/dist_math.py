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
    concatenate, constant, argmax,
    erf, gamma)

from theano.tensor import as_tensor_variable


from numpy import pi, inf, nan
import numpy as np
from .special import gammaln, multigammaln

from theano.printing import Print
from .distribution import *

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
        ret = ret * (1 * c)
    return ret


def logpow(x, m):
    """
    Calculates log(x**m) since m*log(x) will fail when m, x = 0.
    """
    # return m * log(x)
    return switch(eq(m, 0), 0, m * log(x))


def factln(n):
    return gammaln(n + 1)


def idfn(x):
    return x


def std_cdf(x):
    """
    Calculates the standard normal cumulative distribution function.
    """
    return 0.5 + 0.5*erf(x / sqrt(2.))

