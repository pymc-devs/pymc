from .dist_math import *

from theano.sandbox.linalg import det, solve, matrix_inverse, trace
from theano.tensor import dot, cast
from theano.printing import Print

__all__ = ['MvNormal', 'Dirichlet', 'Multinomial', 'Wishart']

class MvNormal(Continuous):
    """
    Multivariate normal

    :Parameters:
        mu : vector of means
        tau : precision matrix

    .. math::
        f(x \mid \pi, T) = \frac{|T|^{1/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}

    :Support:
        2 array of floats
    """
    def __init__(self, mu, tau, *args, **kwargs):
        super(MvNormal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau = tau

    def logp(self, value):
        mu = self.mu
        tau = self.tau

        delta = value - mu
        k = tau.shape[0]

        return 1/2. * (-k * log(2*pi) + log(det(tau)) - dot(delta.T, dot(tau, delta)))


class Dirichlet(Continuous):
    """
    Dirichlet

    This is a multivariate continuous distribution.

    .. math::
        f(\mathbf{x}) = \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)}\prod_{i=1}^{k-1} x_i^{\theta_i - 1}
        \cdot\left(1-\sum_{i=1}^{k-1}x_i\right)^\theta_k

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
    def __init__(self, a, *args, **kwargs):
        super(Dirichlet, self).__init__(*args, **kwargs)
        self.a = a
        self.k = a.shape[0]
        self.mean = a / sum(a)

        self.mode = switch(all(a > 1),
                           (a - 1) / sum(a - 1),
                           nan)

    def logp(self, value):
        k = self.k
        a = self.a

        # only defined for sum(value) == 1
        return bound(
            sum(logpow(
                value, a - 1) - gammaln(a), axis=0) + gammaln(sum(a)),

            k > 1,
            all(a > 0))


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
        self.mode = cast(round(n * p), 'int32')

    def logp(self, x):
        n = self.n
        p = self.p
        # only defined for sum(p) == 1
        return bound(
            factln(n) + sum(x * log(p) - factln(x)),
            n > 0,
            eq(sum(x), n),
            all(0 <= x), all(x <= n))


class Wishart(Continuous):
    """
    The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the precision
    matrix of a multivariate normal distribution. If V=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`C=T{-1}` (Not yet implemented)

    .. math::
        f(X \mid n, T) = \frac{{\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2}}{2^{nk/2}
        \Gamma_p(n/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of X.

    :Parameters:
      n : int
        Degrees of freedom, > 0.
      p : int
        Dimensionality, > 0
      V : ndarray
        p x p positive definite matrix


    :Support:
      X : matrix
        Symmetric, positive definite.
    """
    def __init__(self, n, p, V, *args, **kwargs):
        super(Wishart, self).__init__(*args, **kwargs)
        self.n = n
        self.p = p
        self.V = V
        self.mean = n * V
        self.mode = switch(1*(n >= p + 1),
                     (n - p - 1) * V,
                      nan)

    def logp(self, X):
        n = self.n
        p = self.p
        V = self.V

        IVI = det(V)

        return bound(
            ((n - p - 1) * log(IVI) - trace(matrix_inverse(V).dot(X)) -
             n * p * log(
             2) - n * log(IVI) - 2 * multigammaln(p, n / 2)) / 2,

            all(n > p - 1))
