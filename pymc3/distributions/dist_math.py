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
from pymc3.theanof import floatX

from six.moves import xrange
from functools import partial

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


# Custom Eigh, EighGrad, and eigh are required until
# https://github.com/Theano/Theano/pull/6557 is handled, since lambda's
# cannot be used with pickling.
class Eigh(tt.nlinalg.Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    This is a copy of Eigh from theano that calls an EighGrad which uses
    partial instead of lambda. Once this has been merged with theano this
    should be removed.
    """

    _numop = staticmethod(np.linalg.eigh)
    __props__ = ('UPLO',)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[np.dtype(x.dtype).type()]])[0].dtype.name
        w = theano.tensor.vector(dtype=w_dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return theano.gof.Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        w[0], v[0] = self._numop(x, self.UPLO)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),
        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.
        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:
           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}
           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}
        """
        x, = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        gw, gv = tt.nlinalg._zero_disconnected([w, v], g_outputs)
        return [EighGrad(self.UPLO)(x, w, v, gw, gv)]


class EighGrad(theano.Op):
    """
    Gradient of an eigensystem of a Hermitian matrix.

    This is a copy of EighGrad from theano that uses partial instead of lambda.
    Once this has been merged with theano this should be removed.
    """

    __props__ = ('UPLO',)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO
        if UPLO == 'L':
            self.tri0 = np.tril
            self.tri1 = partial(np.triu, k=1)
        else:
            self.tri0 = np.triu
            self.tri1 = partial(np.tril, k=-1)

    def make_node(self, x, w, v, gw, gv):
        x, w, v, gw, gv = map(tt.as_tensor_variable, (x, w, v, gw, gv))
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out_dtype = theano.scalar.upcast(x.dtype, w.dtype, v.dtype,
                                         gw.dtype, gv.dtype)
        out = theano.tensor.matrix(dtype=out_dtype)
        return theano.gof.Apply(self, [x, w, v, gw, gv], [out])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the eigensystem of
        a square matrix.
        """
        x, w, v, W, V = inputs
        N = x.shape[0]
        outer = np.outer

        def G(n):
            return sum(v[:, m] * V.T[n].dot(v[:, m]) / (w[n] - w[m])
                       for m in xrange(N) if m != n)

        g = sum(outer(v[:, n], v[:, n] * W[n] + G(n))
                for n in xrange(N))

        # Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
        # (triu(a)) only.  This means that partial derivative of
        # eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
        # for i < j (i > j).  At the same time, non-zero components of
        # the gradient must account for the fact that variation of the
        # opposite triangle contributes to variation of two elements
        # of Hermitian (symmetric) matrix. The following line
        # implements the necessary logic.
        out = self.tri0(g) + self.tri1(g).T

        # Make sure we return the right dtype even if NumPy performed
        # upcasting in self.tri0.
        outputs[0][0] = np.asarray(out, dtype=node.outputs[0].dtype)

    def infer_shape(self, node, shapes):
        return [shapes[0]]


def eigh(a, UPLO='L'):
    """A copy, remove with Eigh and EighGrad when possible"""
    return Eigh(UPLO)(a)
