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
    Dirichlet distribution
    """

    a = ones(k) * a

    support = 'continuous'

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
    Wishart distribution
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


