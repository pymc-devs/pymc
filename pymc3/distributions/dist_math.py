'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np
import scipy.linalg
import theano.tensor as tt
import theano
from theano.scalar import UnaryScalarOp, upgrade_to_float_no_complex
from theano.tensor.slinalg import Cholesky
from theano.scan_module import until
from theano import scan

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
    return tt.switch(tt.eq(x, 0), tt.switch(tt.eq(m, 0), 0.0, -np.inf), m * tt.log(x))


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


def normal_lcdf(mu, sigma, x):
    """Compute the log of the cumulative density function of the normal."""
    z = (x - mu) / sigma
    return tt.switch(
        tt.lt(z, -1.0),
        tt.log(tt.erfcx(-z / tt.sqrt(2.)) / 2.) - tt.sqr(z) / 2.,
        tt.log1p(-tt.erfc(z / tt.sqrt(2.)) / 2.)
    )


def normal_lccdf(mu, sigma, x):
    z = (x - mu) / sigma
    return tt.switch(
        tt.gt(z, 1.0),
        tt.log(tt.erfcx(z / tt.sqrt(2.)) / 2.) - tt.sqr(z) / 2.,
        tt.log1p(-tt.erfc(-z / tt.sqrt(2.)) / 2.)
    )


def sigma2rho(sigma):
    """
    `sigma -> rho` theano converter
    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`"""
    return tt.log(tt.exp(tt.abs_(sigma)) - 1.)


def rho2sigma(rho):
    """
    `rho -> sigma` theano converter
    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`"""
    return tt.nnet.softplus(rho)

rho2sd = rho2sigma
sd2rho = sigma2rho

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
    kwargs : one of parameters `{sigma, tau, w, rho}`

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
    sigma = kwargs.get('sigma')
    w = kwargs.get('w')
    rho = kwargs.get('rho')
    tau = kwargs.get('tau')
    eps = kwargs.get('eps', 0.)
    check = sum(map(lambda a: a is not None, [sigma, w, rho, tau]))
    if check > 1:
        raise ValueError('more than one required kwarg is passed')
    if check == 0:
        raise ValueError('none of required kwarg is passed')
    if sigma is not None:
        std = sigma
    elif w is not None:
        std = tt.exp(w)
    elif rho is not None:
        std = rho2sigma(rho)
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
    cholesky = Cholesky(lower=True, on_error='nan')

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


class I1e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 1, exponentially scaled.
    """
    nfunc_spec = ('scipy.special.i1e', 1, 1)

    def impl(self, x):
        return scipy.special.i1e(x)


i1e_scalar = I1e(upgrade_to_float_no_complex, name="i1e")
i1e = tt.Elemwise(i1e_scalar, name="Elemwise{i1e,no_inplace}")


class I0e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 0, exponentially scaled.
    """
    nfunc_spec = ('scipy.special.i0e', 1, 1)

    def impl(self, x):
        return scipy.special.i0e(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return gz * (i1e_scalar(x) - theano.scalar.sgn(x) * i0e_scalar(x)),


i0e_scalar = I0e(upgrade_to_float_no_complex, name="i0e")
i0e = tt.Elemwise(i0e_scalar, name="Elemwise{i0e,no_inplace}")


def random_choice(*args, **kwargs):
    """Return draws from a categorial probability functions

    Args:
        p: array
           Probability of each class
        size: int
            Number of draws to return
        k: int
            Number of bins

    Returns:
        random sample: array

    """
    p = kwargs.pop('p')
    size = kwargs.pop('size')
    k = p.shape[-1]

    if p.ndim > 1:
        # If a 2d vector of probabilities is passed return a sample for each row of categorical probability
        samples = np.array([np.random.choice(k, p=p_) for p_ in p])
    else:
        samples = np.random.choice(k, p=p, size=size)
    return samples


def zvalue(value, sigma, mu):
    """
    Calculate the z-value for a normal distribution.
    """
    return (value - mu) / sigma


def incomplete_beta_cfe(a, b, x, small):
    '''Incomplete beta continued fraction expansions
    based on Cephes library by Steve Moshier (incbet.c).
    small: Choose element-wise which continued fraction expansion to use.
    '''
    BIG = tt.constant(4.503599627370496e15, dtype='float64')
    BIGINV = tt.constant(2.22044604925031308085e-16, dtype='float64')
    THRESH = tt.constant(3. * np.MachAr().eps, dtype='float64')

    zero = tt.constant(0., dtype='float64')
    one = tt.constant(1., dtype='float64')
    two = tt.constant(2., dtype='float64')

    r = one
    k1 = a
    k3 = a
    k4 = a + one
    k5 = one
    k8 = a + two

    k2 = tt.switch(small, a + b, b - one)
    k6 = tt.switch(small, b - one, a + b)
    k7 = tt.switch(small, k4, a + one)
    k26update = tt.switch(small, one, -one)
    x = tt.switch(small, x, x / (one - x))

    pkm2 = zero
    qkm2 = one
    pkm1 = one
    qkm1 = one
    r = one

    def _step(
            i,
            pkm1, pkm2, qkm1, qkm2,
            k1, k2, k3, k4, k5, k6, k7, k8, r
    ):
        xk = -(x * k1 * k2) / (k3 * k4)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk

        xk = (x * k5 * k6) / (k7 * k8)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk

        old_r = r
        r = tt.switch(tt.eq(qk, zero), r, pk/qk)

        k1 += one
        k2 += k26update
        k3 += two
        k4 += two
        k5 += one
        k6 -= k26update
        k7 += two
        k8 += two

        big_cond = tt.gt(tt.abs_(qk) + tt.abs_(pk), BIG)
        biginv_cond = tt.or_(
            tt.lt(tt.abs_(qk), BIGINV),
            tt.lt(tt.abs_(pk), BIGINV)
        )

        pkm2 = tt.switch(big_cond, pkm2 * BIGINV, pkm2)
        pkm1 = tt.switch(big_cond, pkm1 * BIGINV, pkm1)
        qkm2 = tt.switch(big_cond, qkm2 * BIGINV, qkm2)
        qkm1 = tt.switch(big_cond, qkm1 * BIGINV, qkm1)

        pkm2 = tt.switch(biginv_cond, pkm2 * BIG, pkm2)
        pkm1 = tt.switch(biginv_cond, pkm1 * BIG, pkm1)
        qkm2 = tt.switch(biginv_cond, qkm2 * BIG, qkm2)
        qkm1 = tt.switch(biginv_cond, qkm1 * BIG, qkm1)

        return ((pkm1, pkm2, qkm1, qkm2,
                 k1, k2, k3, k4, k5, k6, k7, k8, r),
                until(tt.abs_(old_r - r) < (THRESH * tt.abs_(r))))

    (pkm1, pkm2, qkm1, qkm2,
     k1, k2, k3, k4, k5, k6, k7, k8, r), _ = scan(
        _step,
        sequences=[tt.arange(0, 300)],
        outputs_info=[
            e for e in
            tt.cast((pkm1, pkm2, qkm1, qkm2,
                     k1, k2, k3, k4, k5, k6, k7, k8, r),
                    'float64')
        ]
    )

    return r[-1]


def incomplete_beta_ps(a, b, value):
    '''Power series for incomplete beta
    Use when b*x is small and value not too close to 1.
    Based on Cephes library by Steve Moshier (incbet.c)
    '''
    one = tt.constant(1, dtype='float64')
    ai = one / a
    u = (one - b) * value
    t1 = u / (a + one)
    t = u
    threshold = np.MachAr().eps * ai
    s = tt.constant(0, dtype='float64')

    def _step(i, t, s):
        t *= (i - b) * value / i
        step = t / (a + i)
        s += step
        return ((t, s), until(tt.abs_(step) < threshold))

    (t, s), _ = scan(
        _step,
        sequences=[tt.arange(2, 302)],
        outputs_info=[
            e for e in
            tt.cast((t, s),
                    'float64')
        ]
    )

    s = s[-1] + t1 + ai

    t = (
        gammaln(a + b) - gammaln(a) - gammaln(b) +
        a * tt.log(value) +
        tt.log(s)
    )
    return tt.exp(t)


def incomplete_beta(a, b, value):
    '''Incomplete beta implementation
    Power series and continued fraction expansions chosen for best numerical
    convergence across the board based on inputs.
    '''
    machep = tt.constant(np.MachAr().eps, dtype='float64')
    one = tt.constant(1, dtype='float64')
    w = one - value

    ps = incomplete_beta_ps(a, b, value)

    flip = tt.gt(value, (a / (a + b)))
    aa, bb = a, b
    a = tt.switch(flip, bb, aa)
    b = tt.switch(flip, aa, bb)
    xc = tt.switch(flip, value, w)
    x = tt.switch(flip, w, value)

    tps = incomplete_beta_ps(a, b, x)
    tps = tt.switch(tt.le(tps, machep), one - machep, one - tps)

    # Choose which continued fraction expansion for best convergence.
    small = tt.lt(x * (a + b - 2.0) - (a - one), 0.0)
    cfe = incomplete_beta_cfe(a, b, x, small)
    w = tt.switch(small, cfe, cfe / xc)

    # Direct incomplete beta accounting for flipped a, b.
    t = tt.exp(
        a * tt.log(x) + b * tt.log(xc) +
        gammaln(a + b) - gammaln(a) - gammaln(b) +
        tt.log(w / a)
    )

    t = tt.switch(
        flip,
        tt.switch(tt.le(t, machep), one - machep, one - t),
        t
    )
    return tt.switch(
        tt.and_(flip, tt.and_(tt.le((b * x), one), tt.le(x, 0.95))),
        tps,
        tt.switch(
            tt.and_(tt.le(b * value, one), tt.le(value, 0.95)),
            ps,
            t))
