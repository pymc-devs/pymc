import warnings

import numpy as np
import theano.tensor as T
from scipy import stats
from theano.tensor.nlinalg import det, matrix_inverse, trace, eigh

from . import transforms
from .distribution import Continuous, Discrete, draw_values, generate_samples
from .special import gammaln, multigammaln
from .dist_math import bound, logpow, factln

__all__ = ['MvNormal', 'Dirichlet', 'Multinomial', 'Wishart', 'LKJCorr']


class MvNormal(Continuous):
    """
    Multivariate normal distribution.

    .. math::

       f(x \mid \pi, T) =
           \frac{|T|^{1/2}}{(2\pi)^{1/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}

    :Parameters:
        mu : vector of means
        tau : precision matrix

    :Support:
        2 array of floats
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
    """
    Dirichlet distribution.

    This is a multivariate continuous distribution.

    .. math::

       f(\mathbf{x}) =
           \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)}
           \prod_{i=1}^{k-1} x_i^{\theta_i - 1}
           \left(1-\sum_{i=1}^{k-1}x_i\right)^\theta_k

    :Parameters:
        a : float tensor
            a > 0
            concentration parameters
            last index is the k index

    :Support:
        x : vector
            sum(x) == 1 and x > 0

    .. note::
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
    """
    Generalization of the binomial
    distribution, but instead of each trial resulting in "success" or
    "failure", each one results in exactly one of some fixed finite number k
    of possible outcomes over n independent trials. 'x[i]' indicates the number
    of times outcome number i was observed over the n trials.

    .. math::
        f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    :Parameters:
        n : int
            Number of trials.
        p : (k,)
            Probability of each one of the different outcomes.
            :math:`\sum_{i=1}^k p_i = 1)`, :math:`p_i \ge 0`.

    :Support:
        x : (ns, k) int
            Random variable indicating the number of time outcome i is
            observed. :math:`\sum_{i=1}^k x_i=n`, :math:`x_i \ge 0`.

    .. note::
        - :math:`E(X_i)=n p_i`
        - :math:`Var(X_i)=n p_i(1-p_i)`
        - :math:`Cov(X_i,X_j) = -n p_i p_j`
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
    """
    The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the precision
    matrix of a multivariate normal distribution. If V=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`C=T{-1}`
    (Not yet implemented)

    .. math::

       f(X \mid n, T) =
           \frac{{\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2}}{2^{nk/2}
           \Gamma_p(n/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of X.

    :Parameters:
      n : int
        Degrees of freedom, > 0.
      V : ndarray
        p x p positive definite matrix


    :Support:
      X : matrix
        Symmetric, positive definite.
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


class LKJCorr(Continuous):
    """
    The LKJ (Lewandowski, Kurowicka and Joe) distribution.

    The LKJ distribution is a prior distribution for correlation matrices.
    If n = 1 this corresponds to the uniform distribution over correlation
    matrices. For n -> oo the LKJ prior approaches the identity matrix.

    For more details see:
    http://www.sciencedirect.com/science/article/pii/S0047259X09000876

    This implementation only returns the values of the upper triangular matrix
    excluding the diagonal. Here is a schematic for p = 5, showing the indexes
    of the elements:
        [[- 0 1 2 3]
         [- - 4 5 6]
         [- - - 7 8]
         [- - - - 9]
         [- - - - -]]

    :Parameters:
      n : float
        Shape parameter, Uniform distribution at n=1, > 0
      p : int
        Dimension of correlation matrix


    :Support:
      x : array of size p * (p - 1) / 2
        Upper triangular matrix values [-1,1].
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
                     n > 0)
