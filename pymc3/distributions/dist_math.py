'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division

import numpy as np
import theano.tensor as tt
import theano
from theano.ifelse import ifelse
from theano.tensor import slinalg

from .special import gammaln
from pymc3.theanof import floatX

f = floatX
c = - .5 * np.log(2. * np.pi)


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
    return .5 + .5 * tt.erf(x / tt.sqrt(2.))


def sd2rho(sd):
    """
    `sd -> rho` theano converter
    :math:`mu + sd*e = mu + log(1+exp(rho))*e`"""
    return tt.log(tt.exp(tt.abs_(sd)) - 1.)


def rho2sd(rho):
    """
    `rho -> sd` theano converter
    :math:`mu + sd*e = mu + log(1+exp(rho))*e`"""
    return tt.nnet.softplus(rho)


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
    eps = kwargs.get('eps', 0.)
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
    std += f(eps)
    return f(c) - tt.log(tt.abs_(std)) - (x - mean) ** 2 / (2. * std ** 2)


def MvNormalLogp(chol_cov=False):
    """Compute the log pdf of a multivariate normal distribution.

    Parameters
    ----------
    cov : tt.matrix
        The covariance matrix or its Cholesky decompositon (the latter if
        `chol_cov` is set to True when instantiating the Op).
    delta : tt.matrix
        Array of deviations from the mean.
    """
    cov = tt.matrix('cov')
    cov.tag.test_value = floatX(np.eye(3))
    delta = tt.matrix('delta')
    delta.tag.test_value = floatX(np.zeros((2, 3)))

    solve_lower = slinalg.Solve(A_structure='lower_triangular', overwrite_b=True)
    solve_upper = slinalg.Solve(A_structure='upper_triangular', overwrite_b=True)

    n, k = delta.shape
    n = f(n)

    if not chol_cov:
        # add inplace=True when/if impletemented by Theano
        cholesky = slinalg.Cholesky(lower=True, on_error="nan")
        cov = cholesky(cov)
        # The Cholesky op will return NaNs if the cov is not positive definite
        # checking the first one is sufficient
        ok = ~tt.isnan(cov[0,0])
        # will all be NaN if the Cholesky was no-go, which is fine
        diag = tt.ExtractDiag(view=True)(cov)
    else:
        diag = tt.ExtractDiag(view=True)(cov)
        # Here we must check if the Cholesky is positive definite
        ok = tt.all(diag>0)

    # `solve_lower` throws errors with NaNs hence we replace the cov with
    # identity and return -Inf later
    chol_cov = ifelse(ok, cov, tt.eye(k))
    delta_trans = solve_lower(chol_cov, delta.T).T

    result = n * f(k) * tt.log(f(2) * np.pi)
    result += f(2) * n * tt.sum(tt.log(diag))
    result += (delta_trans ** f(2)).sum()
    result = f(-.5) * result

    logp = ifelse(ok, result, -np.inf * tt.zeros_like(delta))

    def dlogp(inputs, gradients):
        g_logp, = gradients
        cov, delta = inputs

        g_logp.tag.test_value = floatX(1.)
        n, k = delta.shape

        if not chol_cov:
            cov = cholesky(cov)
            ok = ~tt.isnan(chol_cov[0,0])
        else:
            diag = tt.ExtractDiag(view=True)(cov)
            ok = tt.all(diag>0)

        chol_cov = ifelse(ok, cov, tt.eye(k))
        delta_trans = solve_lower(chol_cov, delta.T).T

        inner = n * tt.eye(k) - tt.dot(delta_trans.T, delta_trans)
        g_cov = solve_upper(chol_cov.T, inner)
        g_cov = solve_upper(chol_cov.T, g_cov.T)

        tau_delta = solve_upper(chol_cov.T, delta_trans.T)
        g_delta = tau_delta.T

        g_cov = ifelse(ok, g_cov, -np.nan)
        g_delta = ifelse(ok, g_delta, -np.nan)

        return [-0.5 * g_cov * g_logp, -g_delta * g_logp]

    return theano.OpFromGraph(
        [cov, delta], [logp], grad_overrides=dlogp, inline=True)




class SplineWrapper(theano.Op):
    """
    Creates a theano operation from scipy.interpolate.UnivariateSpline
    """

    __props__ = ('spline',)

    def __init__(self, spline):
        self.spline = spline

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        return tt.Apply(self, [x], [x.type()])

    @property
    def grad_op(self):
        if not hasattr(self, '_grad_op'):
            try:
                self._grad_op = SplineWrapper(self.spline.derivative())
            except ValueError:
                self._grad_op = None

        if self._grad_op is None:
            raise NotImplementedError('Spline of order 0 is not differentiable')
        return self._grad_op

    def perform(self, node, inputs, output_storage):
        x, = inputs
        output_storage[0][0] = np.asarray(self.spline(x))

    def grad(self, inputs, grads):
        x, = inputs
        x_grad, = grads

        return [x_grad * self.grad_op(x)]
