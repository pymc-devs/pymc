'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division

import numpy as np
import theano.tensor as T

from .special import gammaln, multigammaln


def bound(logp, *conditions):
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
    return T.switch(alltrue(conditions), logp, -np.inf)


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
    return T.switch(T.any(T.eq(x, 0)), -np.inf, m * T.log(x))


def factln(n):
    return gammaln(n + 1)


def binomln(n, k):
    return factln(n) - factln(k) - factln(n - k)


def betaln(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def std_cdf(x):
    """
    Calculates the standard normal cumulative distribution function.
    """
    return 0.5 + 0.5*T.erf(x / T.sqrt(2.))
