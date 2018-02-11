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


def CholeskyCheck(mode='cov', return_ldet=True, replacement=None):
    """Checks if the given matrix/cholesky is positive definite. Returns a dummy
       Cholesky replacement if it is not, along with a boolean to assert whether
       replacement was needed and, optionally, the log of the determinant of
       either the real Cholesky or its replacement."""
    is_cholesky = (mode == 'chol')
    w_ldet = return_ldet
    cholesky = slinalg.Cholesky(lower=True, on_error="nan")

    # Check if a given Cholesky is positive definite
    def check_chol(cov):
        diag = tt.ExtractDiag(view=True)(cov)
        ldet = tt.sum(diag.log()) if w_ldet else None
        return tt.all(diag>0), ldet

    # Check if the Cholesky decomposition worked (ie, the cov or tau
    # was positive definite)
    def check_nonchol(cov):
        ldet = None
        if w_ldet :
            # will all be NaN if the Cholesky was no-go, which is fine
            diag = tt.ExtractDiag(view=True)(cov)
            ldet = tt.sum(diag.log())
        return ~tt.isnan(cov[0,0]), ldet

    check = check_chol if is_cholesky else check_nonchol
    repl = lambda ncov: replacement if replacement else tt.identity_like(ncov)

    def func(cov):
        if not is_cholesky:
            # add inplace=True when/if impletemented by Theano
            cov = cholesky(cov)
        ok, ldet = check(cov)
        chol_cov = ifelse(ok, cov, repl(cov))
        return [chol_cov, ldet, ok] if w_ldet else [chol_cov, ok]

    return func


def MvNormalLogp(mode='cov'):
    """Concstructor for the elementwise log pdf of a multivariate normal distribution.

    The returned function will have parameters:
    ----------
    cov : tt.matrix
        The covariance matrix or its Cholesky decompositon (the latter if
        `chol_cov` is set to True when instantiating the Op).
    delta : tt.matrix
        Array of deviations from the mean.
    """
    solve_lower = slinalg.Solve(A_structure='lower_triangular', overwrite_b=True)
    check_chol_wldet = CholeskyCheck(mode, return_ldet=True)
    def logpf(cov, delta):
        chol, logdet, ok = check_chol_wldet(cov)

        if mode == 'tau':
            delta_trans = tt.dot(delta, chol)
        else:
            delta_trans = solve_lower(chol, delta.T).T
        _, k = delta.shape
        quaddist = (delta_trans ** floatX(2)).sum(axis=-1)
        result = floatX(-.5) * floatX(k) * tt.log(floatX(2 * np.pi))
        result += floatX(-.5) * quaddist - logdet
        return ifelse(ok, floatX(result), floatX(-np.inf * tt.ones_like(result)))

    return logpf

def MvNormalLogpSum(mode='cov'):
    """Compute the sum of log pdf of a multivariate normal distribution.

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
    check_chol = CholeskyCheck(mode, return_ldet=False)
    check_chol_wldet = CholeskyCheck(mode, return_ldet=True)

    chol, logdet, ok = check_chol_wldet(cov)

    if mode == 'tau':
        delta_trans = tt.dot(delta, chol)
    else:
        delta_trans = solve_lower(chol, delta.T).T
    quaddist = (delta_trans ** floatX(2)).sum()

    n, k = delta.shape
    result = n * floatX(k) * tt.log(floatX(2 * np.pi))
    result += floatX(2) * n * logdet
    result += quaddist
    result = floatX(-.5) * result

    logp = ifelse(ok, floatX(result), floatX(-np.inf * tt.ones_like(result)))

    def dlogp(inputs, gradients):
        g_logp, = gradients
        g_logp.tag.test_value = floatX(1.)
        cov, delta = inputs
        if (mode == 'tau'):
            warnings.warn("For now, gradient of MvNormalLogp only works "
                          "for cov or chol parameters, not tau.")
            return [grad_not_implemented(self, 0, cov)] * 2

        n, k = delta.shape
        I_k = tt.eye(k, dtype=theano.config.floatX)

        chol_cov, ok = check_chol(cov, replacement=I_k)

        delta_trans = solve_lower(chol_cov, delta.T).T

        inner =  n * I_k - tt.dot(delta_trans.T, delta_trans)
        g_cov = solve_upper(chol_cov.T, inner)
        g_cov = solve_upper(chol_cov.T, g_cov.T)

        tau_delta = solve_upper(chol_cov.T, delta_trans.T)
        g_delta = tau_delta.T

        g_cov = ifelse(ok, f(g_cov), f(-np.nan * tt.zeros_like(g_cov)))
        g_delta = ifelse(ok, f(g_delta), f(-np.nan * tt.zeros_like(g_delta)))

        return [-0.5 * g_cov * g_logp, -g_delta * g_logp]

    return theano.OpFromGraph(
        [cov, delta], [logp], grad_overrides=dlogp, inline=True)

def MvTLogp(nu):
    """Concstructor for the elementwise log pdf of a multivariate normal distribution.

    The returned function will have parameters:
    ----------
    cov : tt.matrix
        The covariance matrix or its Cholesky decompositon (the latter if
        `chol_cov` is set to True when instantiating the Op).
    delta : tt.matrix
        Array of deviations from the mean.
    """
    solve_lower = slinalg.Solve(A_structure='lower_triangular', overwrite_b=True)
    check_chol_wldet = CholeskyCheck(mode, return_ldet=True)
    nu = tt.as_tensor_variable(nu)

    def constructor(mode='cov'):
        def logpf(cov, delta):
            chol, logdet, ok = check_chol_wldet(cov)

            if mode == 'tau':
                delta_trans = tt.dot(delta, chol)
            else:
                delta_trans = solve_lower(chol, delta.T).T
            _, k = delta.shape
            k = floatX(k)

            quaddist = (delta_trans ** floatX(2)).sum()

            result = gammaln((nu + k) / 2.)
            result -= gammaln(nu / 2.)
            result -= .5 * k * tt.log(nu * floatX(np.pi))
            result -= (nu + k) / 2. * tt.log1p(quaddist / nu)
            result -= logdet
            return ifelse(ok, floatX(result), floatX(-np.inf * tt.ones_like(result)))

        return logpf
    return constructor

def MvTLogpSum(nu):
    """Concstructor for the sum of log pdf of a multivariate t distribution.
    WIP (not sure if this is at all possible)
    The returned function will have parameters:
    ----------
    cov : tt.matrix
        The covariance matrix or its Cholesky decompositon (the latter if
        `chol_cov` is set to True when instantiating the Op).
    delta : tt.matrix
        Array of deviations from the mean.
    """
    solve_lower = slinalg.Solve(A_structure='lower_triangular', overwrite_b=True)
    check_chol_wldet = CholeskyCheck(mode, return_ldet=True)
    nu = tt.as_tensor_variable(nu)
    def constuctor(mode='cov'):
        def logpf(cov, delta):
            chol, logdet, ok = check_chol_wldet(cov)

            if mode == 'tau':
                delta_trans = tt.dot(delta, chol)
            else:
                delta_trans = solve_lower(chol, delta.T).T
            n, k = delta.shape
            n, k = floatX(n), floatX(k)

            quaddist = (delta_trans ** floatX(2)).sum(axis=-1)
            ## TODO haven't done the full math yet
            result = n * (gammaln((nu + k) / 2.) - gammaln(nu / 2.))
            result -= n * .5 * k * tt.log(nu * floatX(np.pi))
            result -= (nu + k) / 2. * tt.log1p(quaddist / nu)
            result -= logdet
            return ifelse(ok, floatX(result), floatX(-np.inf * tt.ones_like(result)))
        return logpf
    return constructor

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
