#   Copyright 2021 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Created on Mar 7, 2011

@author: johnsalvatier
"""
import aesara
import aesara.scalar as aes
import aesara.tensor as at
import numpy as np
import scipy.linalg
import scipy.stats

from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.scalar import ScalarOp, UnaryScalarOp, upgrade_to_float_no_complex
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.slinalg import Cholesky, Solve

from pymc3.aesaraf import floatX
from pymc3.distributions.shape_utils import to_tuple
from pymc3.distributions.special import gammaln

f = floatX
c = -0.5 * np.log(2.0 * np.pi)
_beta_clip_values = {
    dtype: (np.nextafter(0, 1, dtype=dtype), np.nextafter(1, 0, dtype=dtype))
    for dtype in ["float16", "float32", "float64"]
}


def bound(logp, *conditions, **kwargs):
    """
    Bounds a log probability density with several conditions.
    When conditions are not met, the logp values are replaced by -inf.

    Note that bound should not be used to enforce the logic of the logp under the normal
    support as it can be disabled by the user via check_bounds = False in pm.Model()

    Parameters
    ----------
    logp: float
    *conditions: booleans
    broadcast_conditions: bool (optional, default=True)
        If True, conditions are broadcasted and applied element-wise to each value in logp.
        If False, conditions are collapsed via at.all(). As a consequence the entire logp
        array is either replaced by -inf or unchanged.

        Setting broadcasts_conditions to False is necessary for most (all?) multivariate
        distributions where the dimensions of the conditions do not unambigously match
        that of the logp.

    Returns
    -------
    logp with elements set to -inf where any condition is False
    """

    # If called inside a model context, see if bounds check is disabled
    try:
        from pymc3.model import modelcontext

        model = modelcontext(kwargs.get("model"))
        if not model.check_bounds:
            return logp
    except TypeError:  # No model found
        pass

    broadcast_conditions = kwargs.get("broadcast_conditions", True)

    if broadcast_conditions:
        alltrue = alltrue_elemwise
    else:
        alltrue = alltrue_scalar

    return at.switch(alltrue(conditions), logp, -np.inf)


def alltrue_elemwise(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret


def alltrue_scalar(vals):
    return at.all([at.all(1 * val) for val in vals])


def logpow(x, m):
    """
    Calculates log(x**m) since m*log(x) will fail when m, x = 0.
    """
    # return m * log(x)
    return at.switch(at.eq(x, 0), at.switch(at.eq(m, 0), 0.0, -np.inf), m * at.log(x))


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
    return 0.5 + 0.5 * at.erf(x / at.sqrt(2.0))


def normal_lcdf(mu, sigma, x):
    """Compute the log of the cumulative density function of the normal."""
    z = (x - mu) / sigma
    return at.switch(
        at.lt(z, -1.0),
        at.log(at.erfcx(-z / at.sqrt(2.0)) / 2.0) - at.sqr(z) / 2.0,
        at.log1p(-at.erfc(z / at.sqrt(2.0)) / 2.0),
    )


def normal_lccdf(mu, sigma, x):
    z = (x - mu) / sigma
    return at.switch(
        at.gt(z, 1.0),
        at.log(at.erfcx(z / at.sqrt(2.0)) / 2.0) - at.sqr(z) / 2.0,
        at.log1p(-at.erfc(-z / at.sqrt(2.0)) / 2.0),
    )


def log_diff_normal_cdf(mu, sigma, x, y):
    """
    Compute :math:`\\log(\\Phi(\frac{x - \\mu}{\\sigma}) - \\Phi(\frac{y - \\mu}{\\sigma}))` safely in log space.

    Parameters
    ----------
    mu: float
        mean
    sigma: float
        std

    x: float

    y: float
        must be strictly less than x.

    Returns
    -------
    log (\\Phi(x) - \\Phi(y))

    """
    x = (x - mu) / sigma / at.sqrt(2.0)
    y = (y - mu) / sigma / at.sqrt(2.0)

    # To stabilize the computation, consider these three regions:
    # 1) x > y > 0 => Use erf(x) = 1 - e^{-x^2} erfcx(x) and erf(y) =1 - e^{-y^2} erfcx(y)
    # 2) 0 > x > y => Use erf(x) = e^{-x^2} erfcx(-x) and erf(y) = e^{-y^2} erfcx(-y)
    # 3) x > 0 > y => Naive formula log( (erf(x) - erf(y)) / 2 ) works fine.
    return at.log(0.5) + at.switch(
        at.gt(y, 0),
        -at.square(y) + at.log(at.erfcx(y) - at.exp(at.square(y) - at.square(x)) * at.erfcx(x)),
        at.switch(
            at.lt(x, 0),  # 0 > x > y
            -at.square(x)
            + at.log(at.erfcx(-x) - at.exp(at.square(x) - at.square(y)) * at.erfcx(-y)),
            at.log(at.erf(x) - at.erf(y)),  # x >0 > y
        ),
    )


def sigma2rho(sigma):
    """
    `sigma -> rho` Aesara converter
    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`"""
    return at.log(at.exp(at.abs_(sigma)) - 1.0)


def rho2sigma(rho):
    """
    `rho -> sigma` Aesara converter
    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`"""
    return at.softplus(rho)


rho2sd = rho2sigma
sd2rho = sigma2rho


def log_normal(x, mean, **kwargs):
    """
    Calculate logarithm of normal distribution at point `x`
    with given `mean` and `std`

    Parameters
    ----------
    x: Tensor
        point of evaluation
    mean: Tensor
        mean of normal distribution
    kwargs: one of parameters `{sigma, tau, w, rho}`

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
    sigma = kwargs.get("sigma")
    w = kwargs.get("w")
    rho = kwargs.get("rho")
    tau = kwargs.get("tau")
    eps = kwargs.get("eps", 0.0)
    check = sum(map(lambda a: a is not None, [sigma, w, rho, tau]))
    if check > 1:
        raise ValueError("more than one required kwarg is passed")
    if check == 0:
        raise ValueError("none of required kwarg is passed")
    if sigma is not None:
        std = sigma
    elif w is not None:
        std = at.exp(w)
    elif rho is not None:
        std = rho2sigma(rho)
    else:
        std = tau ** (-1)
    std += f(eps)
    return f(c) - at.log(at.abs_(std)) - (x - mean) ** 2 / (2.0 * std ** 2)


def MvNormalLogp():
    """Compute the log pdf of a multivariate normal distribution.

    This should be used in MvNormal.logp once Theano#5908 is released.

    Parameters
    ----------
    cov: at.matrix
        The covariance matrix.
    delta: at.matrix
        Array of deviations from the mean.
    """
    cov = at.matrix("cov")
    cov.tag.test_value = floatX(np.eye(3))
    delta = at.matrix("delta")
    delta.tag.test_value = floatX(np.zeros((2, 3)))

    solve_lower = Solve(A_structure="lower_triangular")
    solve_upper = Solve(A_structure="upper_triangular")
    cholesky = Cholesky(lower=True, on_error="nan")

    n, k = delta.shape
    n, k = f(n), f(k)
    chol_cov = cholesky(cov)
    diag = at.diag(chol_cov)
    ok = at.all(diag > 0)

    chol_cov = at.switch(ok, chol_cov, at.fill(chol_cov, 1))
    delta_trans = solve_lower(chol_cov, delta.T).T

    result = n * k * at.log(f(2) * np.pi)
    result += f(2) * n * at.sum(at.log(diag))
    result += (delta_trans ** f(2)).sum()
    result = f(-0.5) * result
    logp = at.switch(ok, result, -np.inf)

    def dlogp(inputs, gradients):
        (g_logp,) = gradients
        cov, delta = inputs

        g_logp.tag.test_value = floatX(1.0)
        n, k = delta.shape

        chol_cov = cholesky(cov)
        diag = at.diag(chol_cov)
        ok = at.all(diag > 0)

        chol_cov = at.switch(ok, chol_cov, at.fill(chol_cov, 1))
        delta_trans = solve_lower(chol_cov, delta.T).T

        inner = n * at.eye(k) - at.dot(delta_trans.T, delta_trans)
        g_cov = solve_upper(chol_cov.T, inner)
        g_cov = solve_upper(chol_cov.T, g_cov.T)

        tau_delta = solve_upper(chol_cov.T, delta_trans.T)
        g_delta = tau_delta.T

        g_cov = at.switch(ok, g_cov, -np.nan)
        g_delta = at.switch(ok, g_delta, -np.nan)

        return [-0.5 * g_cov * g_logp, -g_delta * g_logp]

    return OpFromGraph([cov, delta], [logp], grad_overrides=dlogp, inline=True)


class SplineWrapper(Op):
    """
    Creates an Aesara operation from scipy.interpolate.UnivariateSpline
    """

    __props__ = ("spline",)

    def __init__(self, spline):
        self.spline = spline

    def make_node(self, x):
        x = at.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    @property
    def grad_op(self):
        if not hasattr(self, "_grad_op"):
            try:
                self._grad_op = SplineWrapper(self.spline.derivative())
            except ValueError:
                self._grad_op = None

        if self._grad_op is None:
            raise NotImplementedError("Spline of order 0 is not differentiable")
        return self._grad_op

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(self.spline(x))

    def grad(self, inputs, grads):
        (x,) = inputs
        (x_grad,) = grads

        return [x_grad * self.grad_op(x)]


class I1e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 1, exponentially scaled.
    """

    nfunc_spec = ("scipy.special.i1e", 1, 1)

    def impl(self, x):
        return scipy.special.i1e(x)


i1e_scalar = I1e(upgrade_to_float_no_complex, name="i1e")
i1e = Elemwise(i1e_scalar, name="Elemwise{i1e,no_inplace}")


class I0e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 0, exponentially scaled.
    """

    nfunc_spec = ("scipy.special.i0e", 1, 1)

    def impl(self, x):
        return scipy.special.i0e(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return (gz * (i1e_scalar(x) - aesara.scalar.sgn(x) * i0e_scalar(x)),)


i0e_scalar = I0e(upgrade_to_float_no_complex, name="i0e")
i0e = Elemwise(i0e_scalar, name="Elemwise{i0e,no_inplace}")


def random_choice(*args, **kwargs):
    """Return draws from a categorial probability functions

    Args:
        p: array
           Probability of each class. If p.ndim > 1, the last axis is
           interpreted as the probability of each class, and numpy.random.choice
           is iterated for every other axis element.
        size: int or tuple
            Shape of the desired output array. If p is multidimensional, size
            should broadcast with p.shape[:-1].

    Returns:
        random sample: array

    """
    p = kwargs.pop("p")
    size = kwargs.pop("size")
    k = p.shape[-1]

    if p.ndim > 1:
        # If p is an nd-array, the last axis is interpreted as the class
        # probability. We must iterate over the elements of all the other
        # dimensions.
        # We first ensure that p is broadcasted to the output's shape
        size = to_tuple(size) + (1,)
        p = np.broadcast_arrays(p, np.empty(size))[0]
        out_shape = p.shape[:-1]
        # np.random.choice accepts 1D p arrays, so we semiflatten p to
        # iterate calls using the last axis as the category probabilities
        p = np.reshape(p, (-1, p.shape[-1]))
        samples = np.array([np.random.choice(k, p=p_) for p_ in p])
        # We reshape to the desired output shape
        samples = np.reshape(samples, out_shape)
    else:
        samples = np.random.choice(k, p=p, size=size)
    return samples


def zvalue(value, sigma, mu):
    """
    Calculate the z-value for a normal distribution.
    """
    return (value - mu) / sigma


def clipped_beta_rvs(a, b, size=None, random_state=None, dtype="float64"):
    """Draw beta distributed random samples in the open :math:`(0, 1)` interval.

    The samples are generated with ``scipy.stats.beta.rvs``, but any value that
    is equal to 0 or 1 will be shifted towards the next floating point in the
    interval :math:`[0, 1]`, depending on the floating point precision that is
    given by ``dtype``.

    Parameters
    ----------
    a : float or array_like of floats
        Alpha, strictly positive (>0).
    b : float or array_like of floats
        Beta, strictly positive (>0).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` and ``b`` are both scalars.
        Otherwise, ``np.broadcast(a, b).size`` samples are drawn.
    dtype : str or dtype instance
        The floating point precision that the samples should have. This also
        determines the value that will be used to shift any samples returned
        by the numpy random number generator that are zero or one.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized beta distribution. The scipy
        implementation can yield values that are equal to zero or one. We
        assume the support of the Beta distribution to be in the open interval
        :math:`(0, 1)`, so we shift any sample that is equal to 0 to
        ``np.nextafter(0, 1, dtype=dtype)`` and any sample that is equal to 1
        is shifted to ``np.nextafter(1, 0, dtype=dtype)``.

    """
    out = scipy.stats.beta.rvs(a, b, size=size, random_state=random_state).astype(dtype)
    lower, upper = _beta_clip_values[dtype]
    return np.maximum(np.minimum(out, upper), lower)


def _betainc_a_n(f, p, q, n):
    """
    Numerator (a_n) of the nth approximant of the continued fraction
    representation of the regularized incomplete beta function
    """

    if n == 1:
        return p * f * (q - 1) / (q * (p + 1))

    p2n = p + 2 * n
    F1 = p ** 2 * f ** 2 * (n - 1) / (q ** 2)
    F2 = (p + q + n - 2) * (p + n - 1) * (q - n) / ((p2n - 3) * (p2n - 2) ** 2 * (p2n - 1))

    return F1 * F2


def _betainc_b_n(f, p, q, n):
    """
    Offset (b_n) of the nth approximant of the continued fraction
    representation of the regularized incomplete beta function
    """
    pf = p * f
    p2n = p + 2 * n

    N1 = 2 * (pf + 2 * q) * n * (n + p - 1) + p * q * (p - 2 - pf)
    D1 = q * (p2n - 2) * p2n

    return N1 / D1


def _betainc_da_n_dp(f, p, q, n):
    """
    Derivative of a_n wrt p
    """

    if n == 1:
        return -p * f * (q - 1) / (q * (p + 1) ** 2)

    pp = p ** 2
    ppp = pp * p
    p2n = p + 2 * n

    N1 = -(n - 1) * f ** 2 * pp * (q - n)
    N2a = (-8 + 8 * p + 8 * q) * n ** 3
    N2b = (16 * pp + (-44 + 20 * q) * p + 26 - 24 * q) * n ** 2
    N2c = (10 * ppp + (14 * q - 46) * pp + (-40 * q + 66) * p - 28 + 24 * q) * n
    N2d = 2 * pp ** 2 + (-13 + 3 * q) * ppp + (-14 * q + 30) * pp
    N2e = (-29 + 19 * q) * p + 10 - 8 * q

    D1 = q ** 2 * (p2n - 3) ** 2
    D2 = (p2n - 2) ** 3 * (p2n - 1) ** 2

    return (N1 / D1) * (N2a + N2b + N2c + N2d + N2e) / D2


def _betainc_da_n_dq(f, p, q, n):
    """
    Derivative of a_n wrt q
    """
    if n == 1:
        return p * f / (q * (p + 1))

    p2n = p + 2 * n
    F1 = (p ** 2 * f ** 2 / (q ** 2)) * (n - 1) * (p + n - 1) * (2 * q + p - 2)
    D1 = (p2n - 3) * (p2n - 2) ** 2 * (p2n - 1)

    return F1 / D1


def _betainc_db_n_dp(f, p, q, n):
    """
    Derivative of b_n wrt p
    """
    p2n = p + 2 * n
    pp = p ** 2
    q4 = 4 * q
    p4 = 4 * p

    F1 = (p * f / q) * ((-p4 - q4 + 4) * n ** 2 + (p4 - 4 + q4 - 2 * pp) * n + pp * q)
    D1 = (p2n - 2) ** 2 * p2n ** 2

    return F1 / D1


def _betainc_db_n_dq(f, p, q, n):
    """
    Derivative of b_n wrt to q
    """
    p2n = p + 2 * n
    return -(p ** 2 * f) / (q * (p2n - 2) * p2n)


def _betainc_derivative(x, p, q, wrtp=True):
    """
    Compute the derivative of regularized incomplete beta function wrt to p (alpha) or q (beta)

    Reference: Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
    Journal of Statistical Software, 3(1), 1-20.
    """

    # Input validation
    if not (0 <= x <= 1) or p < 0 or q < 0:
        return np.nan

    if x > (p / (p + q)):
        return -_betainc_derivative(1 - x, q, p, not wrtp)

    min_iters = 3
    max_iters = 200
    err_threshold = 1e-12

    derivative_old = 0

    Am2, Am1 = 1, 1
    Bm2, Bm1 = 0, 1
    dAm2, dAm1 = 0, 0
    dBm2, dBm1 = 0, 0

    f = (q * x) / (p * (1 - x))
    K = np.exp(p * np.log(x) + (q - 1) * np.log1p(-x) - np.log(p) - scipy.special.betaln(p, q))
    if wrtp:
        dK = np.log(x) - 1 / p + scipy.special.digamma(p + q) - scipy.special.digamma(p)
    else:
        dK = np.log1p(-x) + scipy.special.digamma(p + q) - scipy.special.digamma(q)

    for n in range(1, max_iters + 1):
        a_n_ = _betainc_a_n(f, p, q, n)
        b_n_ = _betainc_b_n(f, p, q, n)
        if wrtp:
            da_n = _betainc_da_n_dp(f, p, q, n)
            db_n = _betainc_db_n_dp(f, p, q, n)
        else:
            da_n = _betainc_da_n_dq(f, p, q, n)
            db_n = _betainc_db_n_dq(f, p, q, n)

        A = a_n_ * Am2 + b_n_ * Am1
        B = a_n_ * Bm2 + b_n_ * Bm1
        dA = da_n * Am2 + a_n_ * dAm2 + db_n * Am1 + b_n_ * dAm1
        dB = da_n * Bm2 + a_n_ * dBm2 + db_n * Bm1 + b_n_ * dBm1

        Am2, Am1 = Am1, A
        Bm2, Bm1 = Bm1, B
        dAm2, dAm1 = dAm1, dA
        dBm2, dBm1 = dBm1, dB

        if n < min_iters - 1:
            continue

        F1 = A / B
        F2 = (dA - F1 * dB) / B
        derivative = K * (F1 * dK + F2)

        errapx = abs(derivative_old - derivative)
        d_errapx = errapx / max(err_threshold, abs(derivative))
        derivative_old = derivative

        if d_errapx <= err_threshold:
            break

        if n >= max_iters:
            return np.nan

    return derivative


class TernaryScalarOp(ScalarOp):
    nin = 3


class BetaIncDda(TernaryScalarOp):
    """
    Gradient of the regularized incomplete beta function wrt to the first argument (a)
    """

    def impl(self, a, b, z):
        return _betainc_derivative(z, a, b, wrtp=True)


class BetaIncDdb(TernaryScalarOp):
    """
    Gradient of the regularized incomplete beta function wrt to the second argument (b)
    """

    def impl(self, a, b, z):
        return _betainc_derivative(z, a, b, wrtp=False)


betainc_dda_scalar = BetaIncDda(upgrade_to_float_no_complex, name="betainc_dda")
betainc_ddb_scalar = BetaIncDdb(upgrade_to_float_no_complex, name="betainc_ddb")


class BetaInc(TernaryScalarOp):
    """
    Regularized incomplete beta function
    """

    nfunc_spec = ("scipy.special.betainc", 3, 1)

    def impl(self, a, b, x):
        return scipy.special.betainc(a, b, x)

    def grad(self, inp, grads):
        a, b, z = inp
        (gz,) = grads

        return [
            gz * betainc_dda_scalar(a, b, z),
            gz * betainc_ddb_scalar(a, b, z),
            gz
            * aes.exp(
                aes.log1p(-z) * (b - 1)
                + aes.log(z) * (a - 1)
                - (aes.gammaln(a) + aes.gammaln(b) - aes.gammaln(a + b))
            ),
        ]


betainc_scalar = BetaInc(upgrade_to_float_no_complex, "betainc")
betainc = Elemwise(betainc_scalar, name="Elemwise{betainc,no_inplace}")
