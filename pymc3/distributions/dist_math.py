'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division

import numpy as np
import theano.tensor as tt

from .special import gammaln
from ..math import logdet as _logdet

c = - 0.5 * np.log(2 * np.pi)


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


def sd2rho(sd):
    """
    `sd -> rho` theano converter
    :math:`mu + sd*e = mu + log(1+exp(rho))*e`"""
    return tt.log(tt.exp(sd) - 1)


def rho2sd(rho):
    """
    `rho -> sd` theano converter
    :math:`mu + sd*e = mu + log(1+exp(rho))*e`"""
    return tt.log1p(tt.exp(rho))


def log_normal(x, mean, **kwargs):
    """
    Calculate logarithm of normal distribution at point `x`
    with given `mean` and `std`
    Parameters
    ----------
    x : Tensor
        point of evaluation
    mean : Tensor
        mean of normal distribution
    kwargs : one of parameters `{sd, tau, w, rho}`
    Notes
    -----
    There are four variants for density parametrization.
    They are:
        1) standard deviation - `std`
        2) `w`, logarithm of `std` :math:`w = log(std)`
        3) `rho` that follows this equation :math:`rho = log(exp(std) - 1)`
        4) `tau` that follows this equation :math:`tau = std^{-1}`
    ----
    """
    sd = kwargs.get('sd')
    w = kwargs.get('w')
    rho = kwargs.get('rho')
    tau = kwargs.get('tau')
    eps = kwargs.get('eps', 0.0)
    check = sum(map(lambda a: a is not None, [sd, w, rho, tau]))
    if check > 1:
        raise ValueError('more than one required kwarg is passed')
    if check == 0:
        raise ValueError('none of required kwarg is passed')
    if sd is not None:
        std = sd
    elif w is not None:
        std = tt.exp(w)
    elif rho is not None:
        std = rho2sd(rho)
    else:
        std = tau**(-1)
    std += eps
    return c - tt.log(tt.abs_(std)) - (x - mean) ** 2 / (2 * std ** 2)


def log_normal_mv(x, mean, gpu_compat=False, **kwargs):
    """
    Calculate logarithm of normal distribution at point `x`
    with given `mean` and `sigma` matrix
    Parameters
    ----------
    x : Tensor
        point of evaluation
    mean : Tensor
        mean of normal distribution
    kwargs : one of parameters `{cov, tau, chol}`

    Flags
    ----------
    gpu_compat : False, because LogDet is not GPU compatible yet.
                 If this is set as true, the GPU compatible (but numerically unstable) log(det) is used.

    Notes
    -----
    There are three variants for density parametrization.
    They are:
        1) covariance matrix - `cov`
        2) precision matrix - `tau`,
        3) cholesky decomposition matrix  - `chol`
    ----
    """
    if gpu_compat:
        def logdet(m):
            return tt.log(tt.abs_(tt.nlinalg.det(m)))
    else:
        logdet = _logdet

    T = kwargs.get('tau')
    S = kwargs.get('cov')
    L = kwargs.get('chol')
    check = sum(map(lambda a: a is not None, [T, S, L]))
    if check > 1:
        raise ValueError('more than one required kwarg is passed')
    if check == 0:
        raise ValueError('none of required kwarg is passed')
    # avoid unnecessary computations
    if L is not None:
        S = L.dot(L.T)
        T = tt.nlinalg.matrix_inverse(S)
        log_det = -logdet(S)
    elif T is not None:
        log_det = logdet(T)
    else:
        T = tt.nlinalg.matrix_inverse(S)
        log_det = -logdet(S)
    delta = x - mean
    k = S.shape[0]
    result = k * tt.log(2 * np.pi) - log_det
    result += delta.dot(T).dot(delta)
    return -1 / 2. * result
