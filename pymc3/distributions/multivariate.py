#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import scipy
import theano
import theano.tensor as tt

from scipy import stats
from theano.tensor.nlinalg import det, matrix_inverse, trace

import pymc3 as pm
from . import transforms
from .distribution import Continuous, Discrete, draw_values, generate_samples
from ..model import Deterministic
from .continuous import ChiSquared, Normal
from .special import gammaln, multigammaln
from .dist_math import bound, logpow, factln

__all__ = ['MvNormal', 'MvStudentT', 'Dirichlet',
           'Multinomial', 'Wishart', 'WishartBartlett', 'LKJCorr']


def get_tau_cov(mu, tau=None, cov=None):
    """
    Find precision and standard deviation

    .. math::
        \Tau = \Sigma^-1

    Parameters
    ----------
    mu : array-like
    tau : array-like, optional
    cov : array-like, optional

    Results
    -------
    Returns tuple (tau, sd)

    Notes
    -----
    If neither tau nor cov is provided, returns an identity matrix.
    """
    if tau is None:
        if cov is None:
            cov = np.eye(len(mu))
            tau = np.eye(len(mu))
        else:
            tau = tt.nlinalg.matrix_inverse(cov)

    else:
        if cov is not None:
            raise ValueError("Can't pass both tau and sd")
        else:
            cov = tt.nlinalg.matrix_inverse(tau)

    return (tau, cov)

class MvNormal(Continuous):
    R"""
    Multivariate normal log-likelihood.

    .. math::

       f(x \mid \pi, T) =
           \frac{|T|^{1/2}}{(2\pi)^{1/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T (x-\mu) \right\}

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`T^{-1}`
    ========  ==========================

    Parameters
    ----------
    mu : array
        Vector of means.
    cov : array, optional
        Covariance matrix.
    tau : array, optional
        Precision matrix.
    """

    def __init__(self, mu, cov=None, tau=None, *args, **kwargs):
        super(MvNormal, self).__init__(*args, **kwargs)
        warnings.warn(('The calling signature of MvNormal() has changed. The new signature is:\n'
                       'MvNormal(name, mu, cov) instead of MvNormal(name, mu, tau).'
                       'You do not need to explicitly invert the covariance matrix anymore.'),
                      DeprecationWarning)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau, self.cov = get_tau_cov(mu, tau=tau, cov=cov)

    def random(self, point=None, size=None):
        mu, cov = draw_values([self.mu, self.cov], point=point)

        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(
                mean, cov, None if size == mean.shape else size)

        samples = generate_samples(_random,
                                   mean=mu, cov=cov,
                                   dist_shape=self.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        mu = self.mu
        tau = self.tau

        delta = value - mu
        k = tau.shape[0]

        result = k * tt.log(2 * np.pi) + tt.log(1. / det(tau))
        result += (delta.dot(tau) * delta).sum(axis=delta.ndim - 1)
        return -1 / 2. * result


class MvStudentT(Continuous):
    R"""
    Multivariate Student-T log-likelihood.

    .. math::
        f(\mathbf{x}| \nu,\mu,\Sigma) =
        \frac{\Gamma\left[(\nu+p)/2\right]}{\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}\left|{\Sigma}\right|^{1/2}\left[1+\frac{1}{\nu}({\mathbf x}-{\mu})^T{\Sigma}^{-1}({\mathbf x}-{\mu})\right]^{(\nu+p)/2}}


    ========  =============================================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu` if :math:`\nu > 1` else undefined
    Variance  :math:`\frac{\nu}{\mu-2}\Sigma`
                  if :math:`\nu>2` else undefined
    ========  =============================================


    Parameters
    ----------
    nu : int
        Degrees of freedom.
    Sigma : matrix
        Covariance matrix.
    mu : array
        Vector of means.
    """

    def __init__(self, nu, Sigma, mu=None, *args, **kwargs):
        super(MvStudentT, self).__init__(*args, **kwargs)
        self.nu = nu
        self.mu = np.zeros(Sigma.shape[0]) if mu is None else mu
        self.Sigma = Sigma

        self.mean = self.median = self.mode = self.mu = mu

    def random(self, point=None, size=None):
        chi2 = np.random.chisquare
        mvn = np.random.multivariate_normal

        nu, S, mu = draw_values([self.nu, self.Sigma, self.mu], point=point)

        return (np.sqrt(nu) * (mvn(np.zeros(len(S)), S, size).T
                               / chi2(nu, size))).T + mu

    def logp(self, value):

        S = self.Sigma
        nu = self.nu
        mu = self.mu

        d = S.shape[0]

        X = value - mu

        Q = X.dot(matrix_inverse(S)).dot(X.T).sum()
        log_det = tt.log(det(S))
        log_pdf = gammaln((nu + d) / 2.) - 0.5 * \
            (d * tt.log(np.pi * nu) + log_det) - gammaln(nu / 2.)
        log_pdf -= 0.5 * (nu + d) * tt.log(1 + Q / nu)

        return log_pdf


class Dirichlet(Continuous):
    R"""
    Dirichlet log-likelihood.

    .. math::

       f(\mathbf{x}) =
           \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)}
           \prod_{i=1}^{k-1} x_i^{\theta_i - 1}
           \left(1-\sum_{i=1}^{k-1}x_i\right)^\theta_k

    ========  ===============================================
    Support   :math:`x_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum x_i = 1`
    Mean      :math:`\dfrac{a_i}{\sum a_i}`
    Variance  :math:`\dfrac{a_i - \sum a_0}{a_0^2 (a_0 + 1)}`
              where :math:`a_0 = \sum a_i`
    ========  ===============================================

    Parameters
    ----------
    a : array
        Concentration parameters (a > 0).

    Notes
    -----
    Only the first `k-1` elements of `x` are expected. Can be used
    as a parent of Multinomial and Categorical nevertheless.
    """

    def __init__(self, a, transform=transforms.stick_breaking,
                 *args, **kwargs):
        shape = a.shape[-1]
        kwargs.setdefault("shape", shape)
        super(Dirichlet, self).__init__(transform=transform, *args, **kwargs)

        self.k = shape
        self.a = a
        self.mean = a / tt.sum(a)

        self.mode = tt.switch(tt.all(a > 1),
                              (a - 1) / tt.sum(a - 1),
                              np.nan)

    def random(self, point=None, size=None):
        a = draw_values([self.a], point=point)

        def _random(a, size=None):
            return stats.dirichlet.rvs(a, None if size == a.shape else size)

        samples = generate_samples(_random, a,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, value):
        k = self.k
        a = self.a

        # only defined for sum(value) == 1
        return bound(tt.sum(logpow(value, a - 1) - gammaln(a), axis=-1)
                     + gammaln(tt.sum(a, axis=-1)),
                     tt.all(value >= 0), tt.all(value <= 1),
                     k > 1, tt.all(a > 0),
                     broadcast_conditions=False)


class Multinomial(Discrete):
    R"""
    Multinomial log-likelihood.

    Generalizes binomial distribution, but instead of each trial resulting
    in "success" or "failure", each one results in exactly one of some
    fixed finite number k of possible outcomes over n independent trials.
    'x[i]' indicates the number of times outcome number i was observed
    over the n trials.

    .. math::

       f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    ==========  ===========================================
    Support     :math:`x \in \{0, 1, \ldots, n\}` such that
                :math:`\sum x_i = n`
    Mean        :math:`n p_i`
    Variance    :math:`n p_i (1 - p_i)`
    Covariance  :math:`-n p_i p_j` for :math:`i \ne j`
    ==========  ===========================================

    Parameters
    ----------
    n : int or array
        Number of trials (n > 0).
    p : one- or two-dimensional array
        Probability of each one of the different outcomes. Elements must
        be non-negative and sum to 1 along the last axis. They will be automatically
        rescaled otherwise.
    """

    def __init__(self, n, p, *args, **kwargs):
        super(Multinomial, self).__init__(*args, **kwargs)

        p = p / tt.sum(p, axis=-1, keepdims=True)

        if len(self.shape) == 2:
            try:
                assert n.shape == (self.shape[0],)
            except AttributeError:
                # this occurs when n is a scalar Python int or float
                n *= tt.ones(self.shape[0])

            self.n = tt.shape_padright(n)
            self.p = p if p.ndim == 2 else tt.shape_padleft(p)
        else:
            self.n = n
            self.p = p

        self.mean = self.n * self.p
        self.mode = tt.cast(tt.round(self.mean), 'int32')

    def _random(self, n, p, size=None):
        if size == p.shape:
            size = None
        return np.random.multinomial(n, p, size=size)

    def random(self, point=None, size=None):
        n, p = draw_values([self.n, self.p], point=point)
        samples = generate_samples(self._random, n, p,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, x):
        n = self.n
        p = self.p

        return bound(
            tt.sum(factln(n)) - tt.sum(factln(x)) + tt.sum(x * tt.log(p)),
            tt.all(x >= 0),
            tt.all(tt.eq(tt.sum(x, axis=-1, keepdims=True), n)),
            tt.all(p <= 1),
            tt.all(tt.eq(tt.sum(p, axis=-1), 1)),
            tt.all(tt.ge(n, 0)),
            broadcast_conditions=False
        )


def posdef(AA):
    try:
        np.linalg.cholesky(AA)
        return 1
    except np.linalg.LinAlgError:
        return 0


class PosDefMatrix(theano.Op):
    """
    Check if input is positive definite. Input should be a square matrix.

    """

    # Properties attribute
    __props__ = ()

    # Compulsory if itypes and otypes are not defined

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        o = tt.TensorType(dtype='int8', broadcastable=[])()
        return theano.Apply(self, [x], [o])

    # Python implementation:
    def perform(self, node, inputs, outputs):

        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.array(posdef(x), dtype='int8')
        except Exception:
            pm._log.exception('Failed to check if positive definite', x)
            raise

    def infer_shape(self, node, shapes):
        return [[]]

    def grad(self, inp, grads):
        x, = inp
        return [x.zeros_like(theano.config.floatX)]

    def __str__(self):
        return "MatrixIsPositiveDefinite"

matrix_pos_def = PosDefMatrix()


class Wishart(Continuous):
    R"""
    Wishart log-likelihood.

    The Wishart distribution is the probability distribution of the
    maximum-likelihood estimator (MLE) of the precision matrix of a
    multivariate normal distribution.  If V=1, the distribution is
    identical to the chi-square distribution with n degrees of
    freedom.

    .. math::

       f(X \mid n, T) =
           \frac{{\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2}}{2^{nk/2}
           \Gamma_p(n/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of :math:`X`.

    ========  =========================================
    Support   :math:`X(p x p)` positive definite matrix
    Mean      :math:`n V`
    Variance  :math:`n (v_{ij}^2 + v_{ii} v_{jj})`
    ========  =========================================

    Parameters
    ----------
    n : int
        Degrees of freedom, > 0.
    V : array
        p x p positive definite matrix.

    Note
    ----
    This distribution is unusable in a PyMC3 model. You should instead
    use WishartBartlett or LKJCorr.
    """

    def __init__(self, n, V, *args, **kwargs):
        super(Wishart, self).__init__(*args, **kwargs)
        warnings.warn('The Wishart distribution can currently not be used '
                      'for MCMC sampling. The probability of sampling a '
                      'symmetric matrix is basically zero. Instead, please '
                      'use WishartBartlett or better yet, LKJCorr.'
                      'For more information on the issues surrounding the '
                      'Wishart see here: https://github.com/pymc-devs/pymc3/issues/538.',
                      UserWarning)
        self.n = n
        self.p = p = V.shape[0]
        self.V = V
        self.mean = n * V
        self.mode = tt.switch(1 * (n >= p + 1),
                              (n - p - 1) * V,
                              np.nan)

    def logp(self, X):
        n = self.n
        p = self.p
        V = self.V

        IVI = det(V)
        IXI = det(X)

        return bound(((n - p - 1) * tt.log(IXI)
                      - trace(matrix_inverse(V).dot(X))
                      - n * p * tt.log(2) - n * tt.log(IVI)
                      - 2 * multigammaln(n / 2., p)) / 2,
                     matrix_pos_def(X),
                     tt.eq(X, X.T),
                     n > (p - 1),
                     broadcast_conditions=False
        )


def WishartBartlett(name, S, nu, is_cholesky=False, return_cholesky=False, testval=None):
    R"""
    Bartlett decomposition of the Wishart distribution. As the Wishart
    distribution requires the matrix to be symmetric positive semi-definite
    it is impossible for MCMC to ever propose acceptable matrices.

    Instead, we can use the Barlett decomposition which samples a lower
    diagonal matrix. Specifically:

    .. math::
        \text{If} L \sim \begin{pmatrix}
        \sqrt{c_1} & 0 & 0 \\
        z_{21} & \sqrt{c_2} & 0 \\
        z_{31} & z_{32} & \sqrt{c_3}
        \end{pmatrix}

        \text{with} c_i \sim \chi^2(n-i+1) \text{ and } n_{ij} \sim \mathcal{N}(0, 1), \text{then} \\
        L \times A \times A.T \times L.T \sim \text{Wishart}(L \times L.T, \nu)

    See http://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
    for more information.

    Parameters
    ----------
    S : ndarray
        p x p positive definite matrix
        Or:
        p x p lower-triangular matrix that is the Cholesky factor
        of the covariance matrix.
    nu : int
        Degrees of freedom, > dim(S).
    is_cholesky : bool (default=False)
        Input matrix S is already Cholesky decomposed as S.T * S
    return_cholesky : bool (default=False)
        Only return the Cholesky decomposed matrix.
    testval : ndarray
        p x p positive definite matrix used to initialize

    Note
    ----
    This is not a standard Distribution class but follows a similar
    interface. Besides the Wishart distribution, it will add RVs
    c and z to your model which make up the matrix.
    """

    L = S if is_cholesky else scipy.linalg.cholesky(S)
    diag_idx = np.diag_indices_from(S)
    tril_idx = np.tril_indices_from(S, k=-1)
    n_diag = len(diag_idx[0])
    n_tril = len(tril_idx[0])

    if testval is not None:
        # Inverse transform
        testval = np.dot(np.dot(np.linalg.inv(L), testval), np.linalg.inv(L.T))
        testval = scipy.linalg.cholesky(testval, lower=True)
        diag_testval = testval[diag_idx]**2
        tril_testval = testval[tril_idx]
    else:
        diag_testval = None
        tril_testval = None

    c = tt.sqrt(ChiSquared('c', nu - np.arange(2, 2 + n_diag), shape=n_diag,
                           testval=diag_testval))
    pm._log.info('Added new variable c to model diagonal of Wishart.')
    z = Normal('z', 0, 1, shape=n_tril, testval=tril_testval)
    pm._log.info('Added new variable z to model off-diagonals of Wishart.')
    # Construct A matrix
    A = tt.zeros(S.shape, dtype=np.float32)
    A = tt.set_subtensor(A[diag_idx], c)
    A = tt.set_subtensor(A[tril_idx], z)

    # L * A * A.T * L.T ~ Wishart(L*L.T, nu)
    if return_cholesky:
        return Deterministic(name, tt.dot(L, A))
    else:
        return Deterministic(name, tt.dot(tt.dot(tt.dot(L, A), A.T), L.T))


class LKJCorr(Continuous):
    R"""
    The LKJ (Lewandowski, Kurowicka and Joe) log-likelihood.

    The LKJ distribution is a prior distribution for correlation matrices.
    If n = 1 this corresponds to the uniform distribution over correlation
    matrices. For n -> oo the LKJ prior approaches the identity matrix.

    ========  ==============================================
    Support   Upper triangular matrix with values in [-1, 1]
    ========  ==============================================

    Parameters
    ----------
    n : float
        Shape parameter (n > 0). Uniform distribution at n = 1.
    p : int
        Dimension of correlation matrix (p > 0).

    Notes
    -----
    This implementation only returns the values of the upper triangular
    matrix excluding the diagonal. Here is a schematic for p = 5, showing
    the indexes of the elements::

        [[- 0 1 2 3]
         [- - 4 5 6]
         [- - - 7 8]
         [- - - - 9]
         [- - - - -]]


    References
    ----------
    .. [LKJ2009] Lewandowski, D., Kurowicka, D. and Joe, H. (2009).
        "Generating random correlation matrices based on vines and
        extended onion method." Journal of multivariate analysis,
        100(9), pp.1989-2001.
    """

    def __init__(self, n, p, *args, **kwargs):
        self.n = n
        self.p = p
        n_elem = int(p * (p - 1) / 2)
        self.mean = np.zeros(n_elem)
        super(LKJCorr, self).__init__(shape=n_elem, *args, **kwargs)

        self.tri_index = np.zeros([p, p], dtype=int)
        self.tri_index[np.triu_indices(p, k=1)] = np.arange(n_elem)
        self.tri_index[np.triu_indices(p, k=1)[::-1]] = np.arange(n_elem)

    def _normalizing_constant(self, n, p):
        if n == 1:
            result = gammaln(2. * tt.arange(1, int((p - 1) / 2) + 1)).sum()
            if p % 2 == 1:
                result += (0.25 * (p ** 2 - 1) * tt.log(np.pi)
                           - 0.25 * (p - 1) ** 2 * tt.log(2.)
                           - (p - 1) * gammaln(int((p + 1) / 2)))
            else:
                result += (0.25 * p * (p - 2) * tt.log(np.pi)
                           + 0.25 * (3 * p ** 2 - 4 * p) * tt.log(2.)
                           + p * gammaln(p / 2) - (p - 1) * gammaln(p))
        else:
            result = -(p - 1) * gammaln(n + 0.5 * (p - 1))
            k = tt.arange(1, p)
            result += (0.5 * k * tt.log(np.pi)
                       + gammaln(n + 0.5 * (p - 1 - k))).sum()
        return result

    def logp(self, x):
        n = self.n
        p = self.p

        X = x[self.tri_index]
        X = tt.fill_diagonal(X, 1)

        result = self._normalizing_constant(n, p)
        result += (n - 1.) * tt.log(det(X))
        return bound(result,
                     tt.all(X <= 1), tt.all(X >= -1),
                     matrix_pos_def(X),
                     n > 0,
                     broadcast_conditions=False
        )
