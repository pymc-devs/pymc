from .dist_math import *

from theano.sandbox.linalg import det, solve, matrix_inverse, trace
from theano.tensor import dot, cast, gt
from theano.ifelse import ifelse
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
    
    It is also the conjugate Prior for the Precision Matrix parameter of a
    multivariate normal.

    This follows the parameterization given in formula 290 and 291 on page 24
    of [Kevin Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution]
    available at http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    see that paper for further details and references.
    
    To create noninformative priors for covariance or precision matrices, see
    Huang and Wang, "Simple Marginally Noninformative Prior Distributions
    for Covariance Matrices" ( http://ba.stat.cmu.edu/journal/2013/vol08/issue02/huang.pdf )
    and Gelman, "Prior Distributions for variance parameters in hierarchical models"
    ( https://faculty.washington.edu/jmiyamot/bayes/gelmana%20prior%20distributions%20f%20variance%20parameters%20i%20hierarchical%20mods.pdf )
    as well as http://www.themattsimpson.com/2012/08/20/prior-distributions-for-covariance-matrices-the-scaled-inverse-wishart-prior/
    and http://dahtah.wordpress.com/2012/03/07/why-an-inverse-wishart-prior-may-not-be-such-a-good-idea/
    for a discussion of possible problems with priors of covariance and / or precision matrices
    
    :Parameters:
      v : int
        Degrees of freedom, v > p-1 .
      V : ndarray
        p x p positive definite matrix

    :Support:
      X : matrix
        Symmetric, positive definite.
    """
    def __init__(self, v, S, *args, **kwargs):
        super(Wishart, self).__init__(*args, **kwargs)
        self.v = v
        self.p = p = S.shape[0]
        self.S = S
        
        self.mean = v * S
        self.mode = switch(1*(v > p + 1),
                     (v - p - 1) * S,
                      nan)
        'TODO: We should pre-compute the following if the parameters are fixed'
        self.invalid = theano.tensor.fill(S, nan) # Invalid result, if v<p
        self.Z = log(2.)*(v * p / 2.) + multigammaln(p, v / 2.) + log(det(S)) * v / 2.,
        self.inv_S = matrix_inverse(S)    

    def logp(self, X):
        v = self.v
        p = self.p
        Z = self.Z
        inv_S = self.inv_S 
        result = -Z + log(det(X)) * (v - p - 1) / 2. - trace(inv_S.dot(X)) / 2.
        return ifelse(gt(v, p-1), result, self.invalid) 
    
    @staticmethod
    def jeffreys_prior():
        



class InverseWishart(Continuous):
    """
    The Inverse Wishart distribution is the conjugate prior
    for the covariance matrix of a multivariate normal. It is also 
    the distribution of the maximum-likelihood estimator (MLE) of the covariance
    matrix of a multivariate normal distribution. 
    
    This follows the parameterization given in formula 296 and 297 on page 25
    of [Kevin Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution]
    available at http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    see that paper for further details and references.

    To create noninformative priors for covariance or precision matrices, see
    Huang and Wang, "Simple Marginally Noninformative Prior Distributions
    for Covariance Matrices" ( http://ba.stat.cmu.edu/journal/2013/vol08/issue02/huang.pdf )
    and Gelman, "Prior Distributions for variance parameters in hierarchical models"
    ( https://faculty.washington.edu/jmiyamot/bayes/gelmana%20prior%20distributions%20f%20variance%20parameters%20i%20hierarchical%20mods.pdf )
    as well as http://www.themattsimpson.com/2012/08/20/prior-distributions-for-covariance-matrices-the-scaled-inverse-wishart-prior/
    and http://dahtah.wordpress.com/2012/03/07/why-an-inverse-wishart-prior-may-not-be-such-a-good-idea/
    for a discussion of possible problems with priors of covariance and / or precision matrices
    
    :Parameters:
      v : int
        Degrees of freedom, v > p - 1
      inv_S : ndarray
        p x p positive definite matrix (inverted scale matrix)


    :Support:
      X : matrix
        Symmetric, positive definite.
    """
    def __init__(self, v, inv_S, *args, **kwargs):
        super(Wishart, self).__init__(*args, **kwargs)
        self.v = v
        self.p = p = inv_S.shape[0]
        self.inv_S = inv_S
        
        'TODO: We should pre-compute the following if the parameters are fixed'
        S = matrix_inverse(inv_S)   
        self.S = S
        self.invalid = theano.tensor.fill(inv_S, nan) # Invalid result, if v<p
        self.Z = log(2.)*(v * p / 2.) + multigammaln(p, v / 2.) - log(det(S)) * v / 2.,
        self.mean = ifelse(gt(v, p-1), S / ( v - p - 1), self.invalid) 

         
    def logp(self, X):
        v = self.v
        p = self.p
        S = self.S
        Z = self.Z
        result = -Z + log(det(X)) * -(v + p + 1.) / 2. - trace(S.dot(matrix_inverse(X))) / 2.
        return ifelse(gt(v, p-1), result, self.invalid) 
