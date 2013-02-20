from dist_math import *

__all__ = ['Normal']

from theano.sandbox.linalg import det
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
def Dirichlet(a, n): 
    """
    Dirichlet distribution
    """

    a = ones(n) * a

    support = 'continuous'

    def logp(value):
        return bound(
                sum((a -1)*log(value)) + gammaln(sum(a)) - sum(gammaln(a)),

                a > 0)
    
    mean = a/sum(a)

    mode = switch(all(a > 1), 
            (a-1)/sum(a-1), 
            nan)

    return locals()

