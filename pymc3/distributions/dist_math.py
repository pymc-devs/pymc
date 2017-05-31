'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from __future__ import division

import numpy as np
import scipy.linalg
import theano.tensor as tt
import theano

from .special import gammaln
from ..math import logdet as _logdet
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


def i0(x):
    """
    Calculates the 0 order modified Bessel function of the first kind""
    """
    return tt.switch(tt.lt(x, 5), 1. + x**2 / 4. + x**4 / 64. + x**6 / 2304. + x**8 / 147456.
                     + x**10 / 14745600. + x**12 / 2123366400.,
                     np.e**x / (2. * np.pi * x)**0.5 * (1. + 1. / (8. * x) + 9. / (128. * x**2) + 225. / (3072 * x**3)
                                                       + 11025. / (98304. * x**4)))


def i1(x):
    """
    Calculates the 1 order modified Bessel function of the first kind""
    """
    return tt.switch(tt.lt(x, 5), x / 2. + x**3 / 16. + x**5 / 384. + x**7 / 18432. +
                     x**9 / 1474560. + x**11 / 176947200. + x**13 / 29727129600.,
                     np.e**x / (2. * np.pi * x)**0.5 * (1. - 3. / (8. * x) + 15. / (128. * x**2) + 315. / (3072. * x**3)
                                                        + 14175. / (98304. * x**4)))


def sd2rho(sd):
    """
    `sd -> rho` theano converter
    :math:`mu + sd*e = mu + log(1+exp(rho))*e`"""
    return tt.log(tt.exp(sd) - 1.)


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

    Other Parameters
    ----------------
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
    k = f(S.shape[0])
    result = k * tt.log(2. * np.pi) - log_det
    result += delta.dot(T).dot(delta)
    return -.5 * result


def MvNormalLogp():
    """Compute the log pdf of a multivariate normal distribution.

    This should be used in MvNormal.logp once Theano#5908 is released.

    Parameters
    ----------
    cov : tt.matrix
        The covariance matrix.
    delta : tt.matrix
        Array of deviations from the mean.
    """
    cov = tt.matrix('cov')
    cov.tag.test_value = floatX(np.eye(3))
    delta = tt.matrix('delta')
    delta.tag.test_value = floatX(np.zeros((2, 3)))

    solve_lower = tt.slinalg.Solve(A_structure='lower_triangular')
    solve_upper = tt.slinalg.Solve(A_structure='upper_triangular')
    cholesky = Cholesky(nofail=True, lower=True)

    n, k = delta.shape
    n, k = f(n), f(k)
    chol_cov = cholesky(cov)
    diag = tt.nlinalg.diag(chol_cov)
    ok = tt.all(diag > 0)

    chol_cov = tt.switch(ok, chol_cov, tt.fill(chol_cov, 1))
    delta_trans = solve_lower(chol_cov, delta.T).T

    result = n * k * tt.log(f(2) * np.pi)
    result += f(2) * n * tt.sum(tt.log(diag))
    result += (delta_trans ** f(2)).sum()
    result = f(-.5) * result
    logp = tt.switch(ok, result, -np.inf)

    def dlogp(inputs, gradients):
        g_logp, = gradients
        cov, delta = inputs

        g_logp.tag.test_value = floatX(1.)
        n, k = delta.shape

        chol_cov = cholesky(cov)
        diag = tt.nlinalg.diag(chol_cov)
        ok = tt.all(diag > 0)

        chol_cov = tt.switch(ok, chol_cov, tt.fill(chol_cov, 1))
        delta_trans = solve_lower(chol_cov, delta.T).T

        inner = n * tt.eye(k) - tt.dot(delta_trans.T, delta_trans)
        g_cov = solve_upper(chol_cov.T, inner)
        g_cov = solve_upper(chol_cov.T, g_cov.T)

        tau_delta = solve_upper(chol_cov.T, delta_trans.T)
        g_delta = tau_delta.T

        g_cov = tt.switch(ok, g_cov, -np.nan)
        g_delta = tt.switch(ok, g_delta, -np.nan)

        return [-0.5 * g_cov * g_logp, -g_delta * g_logp]

    return theano.OpFromGraph(
        [cov, delta], [logp], grad_overrides=dlogp, inline=True)


class Cholesky(theano.Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`.

    This is a copy of the cholesky op in theano, that doesn't throw an
    error if the matrix is not positive definite, but instead returns
    nan.

    This has been merged upstream and we should switch to that
    version after the next theano release.

    L = cholesky(X, lower=True) implies dot(L, L.T) == X.
    """
    __props__ = ('lower', 'destructive', 'nofail')

    def __init__(self, lower=True, nofail=False):
        self.lower = lower
        self.destructive = False
        self.nofail = nofail

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        if x.ndim != 2:
            raise ValueError('Matrix must me two dimensional.')
        return tt.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)
        except (ValueError, scipy.linalg.LinAlgError):
            if self.nofail:
                z[0] = np.eye(x.shape[-1])
                z[0][0, 0] = np.nan
            else:
                raise

    def grad(self, inputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.

        Symbolic expression for reverse-mode Cholesky gradient taken from [0]_

        References
        ----------
        .. [0] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527

        """

        x = inputs[0]
        dz = gradients[0]
        chol_x = self(x)
        ok = tt.all(tt.nlinalg.diag(chol_x) > 0)
        chol_x = tt.switch(ok, chol_x, tt.fill_diagonal(chol_x, 1))
        dz = tt.switch(ok, dz, floatX(1))

        # deal with upper triangular by converting to lower triangular
        if not self.lower:
            chol_x = chol_x.T
            dz = dz.T

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return tt.tril(mtx) - tt.diag(tt.diagonal(mtx) / 2.)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            solve = tt.slinalg.Solve(A_structure="upper_triangular")
            return solve(outer.T, solve(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(chol_x.T.dot(dz)))

        if self.lower:
            grad = tt.tril(s + s.T) - tt.diag(tt.diagonal(s))
        else:
            grad = tt.triu(s + s.T) - tt.diag(tt.diagonal(s))
        return [tt.switch(ok, grad, floatX(np.nan))]


class SplineWrapper(theano.Op):
    """
    Creates a theano operation from scipy.interpolate.UnivariateSpline
    """

    __props__ = ('spline',)
    itypes = [tt.dscalar]
    otypes = [tt.dscalar]

    def __init__(self, spline):
        self.spline = spline

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
