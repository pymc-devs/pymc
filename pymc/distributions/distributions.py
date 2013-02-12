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
        
    Example
    -------
    >>> from pymc.distributions import Uniform
    >>> U = Uniform(0,10)
    >>> U(3.5)
    -2.302585092994046
    """
    def dist(value):
        
        return switch((value >= lb) & (value <= ub),
                  -log(ub-lb),
                  -inf)
    
    dist.__doc__ = """
        Uniform log-likelihood with parameters lower={0} and upper={1}.
        
        Parameters
        ----------
        value : float
            :math:`lower \leq x \leq upper`
        """              
    
    return dist

def Flat():
    """
    Uninformative log-likelihood that returns 0 regardless of 
    the passed value.
    
    Example
    -------
    >>> from pymc.distributions import Flat
    >>> F = Flat()
    >>> F([1000, 0, 15])
    0.0
    """
    
    def dist(value):
        return zeros_like(value)
        
    dist.__doc__ = """
        Uninformative log-likelihood that returns 0 regardless of 
        the passed value.
        """
        
    return dist

def Normal(mu=0.0, tau=1.0):
    """
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    tau : float
        Precision of the distribution, which corresponds to
        :math:`1/\sigma^2` (tau > 0).
        
    Example
    -------
    >>> from pymc.distributions import Normal
    >>> N = Normal(0, 0.01)
    >>> N(-3.2)
    -3.2727236261987187

    .. note::
    - :math:`E(X) = \mu`
    - :math:`Var(X) = 1/\tau`

    """
    def dist(value):
        
        return switch(gt(tau , 0),
			 -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)
    
    dist.__doc__ = """
        Normal log-likelihood with paraemters mu={0} and tau={1}.
        
        Parameters
        ----------
        value : float
            Input data.
        """.format(mu, tau)
    
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
    >>> >>> from pymc.distributions import  Beta
    >>> B = Beta(1,2)
    >>> B(.4)
    0.182321556793954

    .. note::
    - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    def dist(value):
        
        return switch(ge(value , 0) & le(value , 1) &
                  gt(alpha , 0) & gt(beta , 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)
                  
    dist.__doc__ = """
    Beta log-likelihood with parameters alpha={0} and beta={1}.
        
    Parameters
    ----------
    value : float
        0 < x < 1
    """.format(alpha, beta)
                  
    return dist

def Binomial(n, p):
    """
    Binomial log-likelihood.  The discrete probability distribution 
    of the number of successes in a sequence of n independent yes/no
    experiments, each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{n-x}

    Parameters
    ----------
    n : int 
        Number of Bernoulli trials, n > x
    p : float
        Probability of success in each trial, :math:`p \in [0,1]`

    .. note::
    - :math:`E(X)=np`
    - :math:`Var(X)=np(1-p)`

    """
    def dist(value):
        
        return switch(ge(value , 0) & ge(n , value) & ge(p , 0) & le(p , 1),
                   switch(ne(value , 0) , value*log(p), 0) + (n-value)*log(1-p) + factln(n)-factln(value)-factln(n-value),
                   -inf)
    dist.__doc__ = """
        Binomial log-likelihood with parameters n={0} and p={1}.
        
        Parameters
        ----------
        value : int 
            Number of successes, x > 0
        """.format(n,p)
    
    return dist
    
def BetaBin(alpha, beta, n):
    """
    Beta-binomial log-likelihood. Equivalent to binomial random
    variables with probabilities drawn from a
    :math:`\texttt{Beta}(\alpha,\beta)` distribution.

    .. math::
        f(x \mid \alpha, \beta, n) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)} \frac{\Gamma(n+1)}{\Gamma(x+1)\Gamma(n-x+1)} \frac{\Gamma(\alpha + x)\Gamma(n+\beta-x)}{\Gamma(\alpha+\beta+n)}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0
    n : int
        n=x,x+1,\ldots

    Example
    -------
    >>> from pymc.distributions import BetaBin
    >>> B = BetaBin(1,1,10)
    >>> B(3)
    -2.3978952727989

    .. note::
    - :math:`E(X)=n\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=n\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    
    def dist(value):
        
        return switch(ge(value , 0) & gt(alpha , 0) & gt(beta , 0) & ge(n , value), 
                   gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(value+1)- gammaln(n-value +1) + gammaln(alpha+value)+ gammaln(n+beta-value)- gammaln(beta+alpha+n),
                   -inf)
    dist.__doc__ = """
        Beta-binomial log-likelihood with parameters alpha={0}, beta={1},
        and n={2}.
        
        Parameters
        ----------
        value : int
            x=0,1,\ldots,n
        """.format(alpha, beta, n)
    
    return dist
    
def Bernoulli(p):
    """Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes (x=1) and
    failures (x=0).

    .. math::  f(x \mid p) = p^{x} (1-p)^{1-x}

    Parameters
    ----------
    x : int
        Series of successes (1) and failures (0). :math:`x=0,1`
    p : float
        Probability of success. :math:`0 < p < 1`.

    Example
    -------
    >>> from pymc.distributions import Bernoulli
    >>> B = Bernoulli(0.4)
    >>> B([0,1,0,1])
    -2.854232711280291

    .. note::
    - :math:`E(x)= p`
    - :math:`Var(x)= p(1-p)`

    """
    def dist(value):
        
        return switch(ge(p , 0) & le(p , 1), 
                  switch(value, log(p), log(1-p)),
                  -inf)
    
    dist.__doc__ = """
        Bernoulli log-likelihood with parameter p={0}.
        
        Parameters
        ----------
        value : int
            Series of successes (1) and failures (0). :math:`x=0,1`
        """.format(p)
    
    return dist

def T(nu, mu=0, lam=1):
    """
    Non-central Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed. If
    only nu parameter is passed, this specifies a standard (central) 
    Student's T.

    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
        \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    Parameters
    ----------
    nu : int
        Degrees of freedom
    mu : float
        Location parameter (defaults to 0)
    lam : float
        Scale parameter (defaults to 1)


    Example
    -------
    >>> from pymc.distributions import T
    >>> t = T(nu=10, mu=3, lam=5)
    >>> t(1.2)
    -5.436637143685433
    """
    
    def dist(value):
        return switch(gt(lam  , 0) & gt(nu , 0),
                  gammaln((nu+1.0)/2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu/2.0) - (nu+1.0)/2.0 * log(1.0 + lam *(value - mu)**2/nu),
                  -inf)
    dist.__doc__ = """
        Student's T log-likelihood with paramters nu={0}, mu={1} and lam={2}.
        
        Parameters
        ----------
        value : float
            Input data
        """
    
    return dist
    
def Cauchy(alpha, beta):
    """
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    Parameters
    ----------
    alpha : float
        Location parameter
    beta : float
        Scale parameter > 0

    .. note::
    Mode and median are at alpha.

    """
    def dist(value):
        return switch(gt(beta , 0),
                  -log(beta) - log( 1 + ((value-alpha) / beta) ** 2 ),
                  -inf)
                  
    dist.__doc__ = """
        Cauchy log-likelihood with parameters alpha={0} and beta={0}.
        
        Parameters
        ----------
        value : float
            Input data.
        """.format(alpha, beta)
    return dist
    
def Gamma(alpha, beta):
    """
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    Parameters
    ----------
    x : float
        math:`x \ge 0`
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).

    .. note::
    - :math:`E(X) = \frac{\alpha}{\beta}`
    - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """
    def dist(value):
        return switch(ge(value , 0) & gt(alpha , 0) & gt(beta , 0),
                  -gammaln(alpha) + alpha*log(beta) - beta*value + switch(alpha != 1.0, (alpha - 1.0)*log(value), 0),
                  -inf)
    dist.__doc__ = """
        Gamma log-likelihood with paramters alpha={0} and beta={1}.
        
        Parameters
        ----------
        x : float
            math:`x \ge 0`
        """.format(alpha, beta)
    
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
