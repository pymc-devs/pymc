import warnings

import numpy as np
import theano.tensor as T
import theano

from scipy import stats
from theano.tensor.nlinalg import det, matrix_inverse, trace, eigh

from . import transforms
from .distribution import Continuous, Discrete, draw_values, generate_samples
from .special import gammaln, multigammaln
from .dist_math import bound, logpow, factln

__all__ = ['MvNormal', 'Dirichlet', 'Multinomial', 'Wishart', 'LKJCorr']


class MvNormal(Continuous):
    r"""
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
    tau : array
        Precision matrix.
    """
    def __init__(self, mu, tau, *args, **kwargs):
        super(MvNormal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau = tau

    def random(self, point=None, size=None):
        mu, tau = draw_values([self.mu, self.tau], point=point)

        def _random(mean, cov, size=None):
            # FIXME: cov here is actually precision?
            return stats.multivariate_normal.rvs(
                mean, cov, None if size == mean.shape else size)

        samples = generate_samples(_random,
                                   mean=mu, cov=tau,
                                   dist_shape=self.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        mu = self.mu
        tau = self.tau

        delta = value - mu
        k = tau.shape[0]

        result = k * T.log(2 * np.pi) + T.log(1./det(tau))
        result += (delta.dot(tau) * delta).sum(axis=delta.ndim - 1)
        return -1/2. * result


class Dirichlet(Continuous):
    r"""
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
        shape = a.shape[0]
        kwargs.setdefault("shape", shape)
        super(Dirichlet, self).__init__(transform=transform, *args, **kwargs)

        self.k = shape
        self.a = a
        self.mean = a / T.sum(a)

        self.mode = T.switch(T.all(a > 1),
                             (a - 1) / T.sum(a - 1),
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
        return bound(T.sum(logpow(value, a - 1) - gammaln(a), axis=0)
                     + gammaln(T.sum(a, axis=0)),
                     T.all(value >= 0), T.all(value <= 1),
                     k > 1, T.all(a > 0))


class Multinomial(Discrete):
    r"""
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
    n : int
        Number of trials (n > 0).
    p : array
        Probability of each one of the different outcomes. Elements must
        be non-negative and sum to 1.
    """
    def __init__(self, n, p, *args, **kwargs):
        super(Multinomial, self).__init__(*args, **kwargs)
        self.n = n
        self.p = p
        self.mean = n * p
        self.mode = T.cast(T.round(n * p), 'int32')

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
        # only defined for sum(p) == 1
        return bound(
            factln(n) + T.sum(x * T.log(p) - factln(x)),
            T.all(x >= 0), T.all(x <= n), T.eq(T.sum(x), n),
            n >= 0)


class Wishart(Continuous):
    r"""
    Wishart log-likelihood.

    The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the precision
    matrix of a multivariate normal distribution.  If V=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`C=T{-1}`
    (Not yet implemented)

    .. math::

       f(X \mid n, T) =
           \frac{{\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2}}{2^{nk/2}
           \Gamma_p(n/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of X.

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
    """
    def __init__(self, n, V, *args, **kwargs):
        super(Wishart, self).__init__(*args, **kwargs)
        warnings.warn('The Wishart distribution can currently not be used '
                      'for MCMC sampling. The probability of sampling a '
                      'symmetric matrix is basically zero. Instead, please '
                      'use the LKJCorr prior. For more information on the '
                      'issues surrounding the Wishart see here: '
                      'https://github.com/pymc-devs/pymc3/issues/538.',
                      UserWarning)
        self.n = n
        self.p = p = V.shape[0]
        self.V = V
        self.mean = n * V
        self.mode = T.switch(1*(n >= p + 1),
                             (n - p - 1) * V,
                             np.nan)

    def logp(self, X):
        n = self.n
        p = self.p
        V = self.V

        IVI = det(V)
        IXI = det(X)

        return bound(((n - p - 1) * T.log(IXI)
                     - trace(matrix_inverse(V).dot(X))
                     - n * p * T.log(2) - n * T.log(IVI)
                     - 2 * multigammaln(n / 2., p)) / 2,
                     T.all(eigh(X)[0] > 0), T.eq(X, X.T),
                     n > (p - 1))


def posdef(AA):
    try:
        fct = np.linalg.cholesky(AA)
        return 1
    except np.linalg.LinAlgError:
        return 0


class PosDefMatrix(theano.Op):
    """
    Check if input is positive definite. Input should be a square matrix.

    """

    #Properties attribute
    __props__ = ()


    #Compulsory if itypes and otypes are not defined

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        assert x.ndim == 2
        o=T.TensorType(dtype='int8', broadcastable = [])()
        return theano.Apply(self, [x], [o])

    # Python implementation:
    def perform(self, node, inputs, outputs):

        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.array(posdef(x), dtype='int8')
        except Exception:
            print('Failed to check if positive definite', x)
            raise
        
    def infer_shape(self, node, shapes):
        return [[]]

    def grad(self, inp, grads):
        x, = inp
        return [x.zeros_like(theano.config.floatX)]

    def __str__(self):
        return "MatrixIsPositiveDefinite"
    
matrix_pos_def = PosDefMatrix()

    
class LKJCorr(Continuous):
    r"""
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
            result = gammaln(2. * T.arange(1, int((p-1) / 2) + 1)).sum()
            if p % 2 == 1:
                result += (0.25 * (p ** 2 - 1) * T.log(np.pi)
                           - 0.25 * (p - 1) ** 2 * T.log(2.)
                           - (p - 1) * gammaln(int((p + 1) / 2)))
            else:
                result += (0.25 * p * (p - 2) * T.log(np.pi)
                           + 0.25 * (3 * p ** 2 - 4 * p) * T.log(2.)
                           + p * gammaln(p / 2) - (p-1) * gammaln(p))
        else:
            result = -(p - 1) * gammaln(n + 0.5 * (p - 1))
            k = T.arange(1, p)
            result += (0.5 * k * T.log(np.pi)
                       + gammaln(n + 0.5 * (p - 1 - k))).sum()
        return result

    def logp(self, x):
        n = self.n
        p = self.p

        X = x[self.tri_index]
        X = T.fill_diagonal(X, 1)

        result = self._normalizing_constant(n, p)
        result += (n - 1.) * T.log(det(X))
        return bound(result,
                     T.all(X <= 1), T.all(X >= -1),
                     matrix_pos_def(X),
                     n > 0)
