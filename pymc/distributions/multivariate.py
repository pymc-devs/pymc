from dist_math import *

__all__ = ['Normal']

from theano.sandbox.linalg import det, solve
from theano.tensor import dot

@quickclass
def Normal(mu, Tau):
    """
    Multivariate normal log-likelihood

    .. math::
        f(x \mid \pi, T) = \frac{|T|^{1/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}
    """

    support = 'continuous'

    def logp(value): 
        delta = value - mu
        return 1/2. * ( log(det(Tau)) - dot(delta.T,dot(Tau, delta)))
        
    logp.__doc__ = """
        Multivariate normal log-likelihood with parameters mu={0} and 
        tau={1}.
        
        Parameters
        ----------
        x : 2D array or list of floats
        """
    return locals()


@quickclass
def Dirichlet(k, a): 
    """
    Dirichlet 

    This is a multivariate continuous distribution.

    .. math::
        f(\mathbf{x}) = \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)}\prod_{i=1}^{k-1} x_i^{\theta_i - 1}
        \cdot\left(1-\sum_{i=1}^{k-1}x_i\right)^\theta_k

    :Parameters:
        theta : array
            An (n,k) or (1,k) array > 0.


    .. note::
        Only the first `k-1` elements of `x` are expected. Can be used
        as a parent of Multinomial and Categorical nevertheless.
    """
    support = 'continuous'

    a = ones(k) * a
    def logp(value):

        #only defined for sum(value) == 1 
        return bound(
                sum((a -1)*log(value)) + gammaln(sum(a)) - sum(gammaln(a)),

                k > 2,
                a > 0)
    
    mean = a/sum(a)

    mode = switch(all(a > 1), 
            (a-1)/sum(a-1), 
            nan)

    return locals()


@quickclass
def Multinomial(n, p):
    """
    Generalization of the binomial
    distribution, but instead of each trial resulting in "success" or
    "failure", each one results in exactly one of some fixed finite number k
    of possible outcomes over n independent trials. 'x[i]' indicates the number
    of times outcome number i was observed over the n trials.

    .. math::
        f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    :Parameters:
        x : (ns, k) int
            Random variable indicating the number of time outcome i is
            observed. :math:`\sum_{i=1}^k x_i=n`, :math:`x_i \ge 0`.
        n : int
            Number of trials.
        p : (k,)
            Probability of each one of the different outcomes.
            :math:`\sum_{i=1}^k p_i = 1)`, :math:`p_i \ge 0`.

    .. note::
        - :math:`E(X_i)=n p_i`
        - :math:`Var(X_i)=n p_i(1-p_i)`
        - :math:`Cov(X_i,X_j) = -n p_i p_j`
    """

    support = 'discrete'

    def logp(x): 
        #only defined for sum(p) == 1
        return bound( 
                factln(n) + sum(x*log(p) - factln(x)), 
                n > 0, 
                0 <= x, x <= n)

    mean = n*p

    return locals()



@quickclass
def Wishart(n, p, V):
    """
    The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the precision
    matrix of a multivariate normal distribution. If Tau=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`C=T{-1}` (Not yet implemented)

    .. math::
        f(X \mid n, T) = \frac{{\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2}}{2^{nk/2}
        \Gamma_p(n/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of X.

    :Parameters:
      X : matrix
        Symmetric, positive definite.
      n : int
        Degrees of freedom, > 0.
      Tau : matrix
        Symmetric and positive definite
    """
    support = 'continuous'

    def logp(X): 
        IVI  = det(V)
        return bound( 
                ((n-p-1)*log( IVI ) - trace(solve(V, X)) - n*p *log(2) - n*log( IVI ) -2*multigammaln(p, n/2))/2, 

                n > p -1)

    mean = n*V
    mode = switch(n >= p+1, 
            (n-p-1)*V,
            nan)

    return locals()


