"""
pymc.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""

from dist_math import * 
from theano.sandbox.linalg import det
from theano.tensor import dot

def Uniform(lower=0, upper=1):
    """
    Continuous uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    Parameters
    ----------
    lower : float
        Lower limit (defaults to 0)
    upper : float
        Upper limit (defaults to 1)
    """
    def dist(value):
        """
        Uniform log-likelihood with parameters lower={0} and upper={1}.
        
        Parameters
        ----------
        x : float
            :math:`lower \leq x \leq upper`
        """
        
        return switch((value >= lb) & (value <= ub),
                  -log(ub-lb),
                  -inf)
    return dist

def Flat():
    """
    Uninformative log-likelihood that returns 0 regardless of 
    the passed value.
    """
    
    def dist(value):
        """
        Uninformative log-likelihood that returns 0 regardless of 
        the passed value.
        """
        return zeros_like(value)
    return dist

def Normal(mu=0.0, tau=1.0):
    """
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Parameters
    ----------
    mu : Mean of the distribution.
    tau : Precision of the distribution, which corresponds to
        :math:`1/\sigma^2` (tau > 0).

    .. note::
        :math:`E(X) = \mu`
        :math:`Var(X) = 1/\tau`

    """
    def dist(value):
        """
        Normal log-likelihood with paraemters mu={0} and tau={1}.
        
        Parameters
        ----------
        x : Input data.
        """.format(mu, tau)
        
        return switch(gt(tau , 0),
			 -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)
    return dist

def Beta(alpha, beta):
    """
    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0

    Example
    -------
    from pymc import Beta
    >>> B = Beta(1,2)
    >>> B(.4)
    0.182321556793954

    .. note::
        :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
        :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    def dist(value):
        """
        Beta log-likelihood with parameters alpha={0} and beta={1}.
        
        Parameters
        ----------
        x : float
            0 < x < 1
        """.format(alpha, beta)
        
        return switch(ge(value , 0) & le(value , 1) &
                  gt(alpha , 0) & gt(beta , 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)
    return dist

def Binomial(n, p):
    """
    Binomial log-likelihood.  The discrete probability distribution 
    of the number of successes in a sequence of n independent yes/no
    experiments, each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{n-x}

    :Parameters:
    x : int 
        Number of successes, > 0.
    n : int 
        Number of Bernoulli trials, > x.
    p : float
        Probability of success in each trial, :math:`p \in [0,1]`.

    .. note::
        :math:`E(X)=np`
        :math:`Var(X)=np(1-p)`

    """
    def dist(value):
        return switch (ge(value , 0) & ge(n , value) & ge(p , 0) & le(p , 1),
                   switch(ne(value , 0) , value*log(p), 0) + (n-value)*log(1-p) + factln(n)-factln(value)-factln(n-value),
                   -inf)
    return dist
    
def BetaBin(alpha, beta, n):
    def dist(value):
        return switch (ge(value , 0) & gt(alpha , 0) & gt(beta , 0) & ge(n , value), 
                   gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(value+1)- gammaln(n-value +1) + gammaln(alpha+value)+ gammaln(n+beta-value)- gammaln(beta+alpha+n),
                   -inf)
    return dist
    
def Bernoulli(p):
    def dist(value):
        return switch(ge(p , 0) & le(p , 1), 
                  switch(value, log(p), log(1-p)),
                  -inf)
    return dist

def T(mu, lam, nu):
    def dist(value):
        return switch(gt(lam  , 0) & gt(nu , 0),
                  gammaln((nu+1.0)/2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu/2.0) - (nu+1.0)/2.0 * log(1.0 + lam *(value - mu)**2/nu),
                  -inf)
    return dist
    
def Cauchy(alpha, beta):
    def dist(value):
        return switch(gt(beta , 0),
                  -log(beta) - log( 1 + ((value-alpha) / beta) ** 2 ),
                  -inf)
    return dist
    
def Gamma(alpha, beta):
    def dist(value):
        return switch(ge(value , 0) & gt(alpha , 0) & gt(beta , 0),
                  -gammaln(alpha) + alpha*log(beta) - beta*value + switch(alpha != 1.0, (alpha - 1.0)*log(value), 0),
                  -inf)
    return dist

def Poisson(lam):
    def dist(value):
        return switch( gt(lam,0),
               #factorial not implemented yet, so
               value * log(lam) - gammaln(value + 1) - lam,
               -inf)
    return dist

def ConstantDist(c):
    def dist(value):
        
        return switch(eq(value, c), 0, -inf)
    return dist

def ZeroInflatedPoisson(theta, z):
    def dist(value):
        return switch(z, 
                      Poisson(theta)(value), 
                      ConstantDist(0)(value))
    return dist

def Bound(dist, lower=-inf, upper=inf):
    def dist(value):
        return switch(ge(value , lower) & le(value , upper),
                  dist(value),
                  -inf)
    return ndist

def TruncT(mu, lam, nu):
    return Bound(T(mu,lam,nu), 0)

def MvNormal(mu, tau):
    def dist(value): 
        delta=value - mu
        return 1/2. * ( log(det(tau)) - dot(delta.T,dot(tau, delta)))
    return dist
