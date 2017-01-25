'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division

import numpy as np
import theano.tensor as tt

from .special import gammaln


def bound(logp, *conditions, **kwargs):
    """
    Bounds a log probability density with several conditions.

    Parameters
    ----------
    logp : float
    *conditions : booleans
    broadcast_conditions : bool (optional, default=True)
        If True, broadcasts logp to match the largest shape of the conditions.
        This is used e.g. in DiscreteUniform where logp is a scalar constant and the shape
        is specified via the conditions.
        If False, will return the same shape as logp.
        This is used e.g. in Multinomial where broadcasting can lead to differences in the logp.

    Returns
    -------
    logp with elements set to -inf where any condition is False
    """
    broadcast_conditions = kwargs.get('broadcast_conditions', True)

    if broadcast_conditions:
        alltrue = alltrue_elemwise
    else:
        alltrue = alltrue_scalar

    return tt.switch(alltrue(conditions), logp, -np.inf)


def alltrue_elemwise(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret


def alltrue_scalar(vals):
    return tt.all([tt.all(1 * val) for val in vals])


def logpow(x, m):
    """
    Calculates log(x**m) since m*log(x) will fail when m, x = 0.
    """
    # return m * log(x)
    return tt.switch(tt.eq(x, 0), -np.inf, m * tt.log(x))


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
    return 0.5 + 0.5 * tt.erf(x / tt.sqrt(2.))


def i0(x):
    """
    Calculates the 0 order modified Bessel function of the first kind""
    """
    return tt.switch(tt.lt(x, 5), 1 + x**2 / 4 + x**4 / 64 + x**6 / 2304 + x**8 / 147456
                     + x**10 / 14745600 + x**12 / 2123366400,
                     np.e**x / (2 * np.pi * x)**0.5 * (1 + 1 / (8 * x) + 9 / (128 * x**2) + 225 / (3072 * x**3)
                                                       + 11025 / (98304 * x**4)))


def i1(x):
    """
    Calculates the 1 order modified Bessel function of the first kind""
    """
    return tt.switch(tt.lt(x, 5), x / 2 + x**3 / 16 + x**5 / 384 + x**7 / 18432 +
                     x**9 / 1474560 + x**11 / 176947200 + x**13 / 29727129600,
                     np.e**x / (2 * np.pi * x)**0.5 * (1 - 3 / (8 * x) + 15 / (128 * x**2) + 315 / (3072 * x**3)
                                                       + 14175 / (98304 * x**4)))
