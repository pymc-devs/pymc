"""
pymc.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""

from dist_math import * 
from theano.sandbox.linalg import det
from theano.tensor import dot

from functools import wraps

def quickclass(fn): 
    class Distribution(object):
        __doc__ = fn.__doc__

        @wraps(fn)
        def __init__(self, *args, **kwargs):  #still need to figure out how to give it the right argument names
                properties = fn(*args, **kwargs) 
                self.__dict__.update(properties)


    Distribution.__name__ = fn.__name__
    return Distribution


@quickclass
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
    def logp(value):
        
        return switch((value >= lower) & (value <= upper),
                  -log(upper-lower),
                  -inf)
    
    logp.__doc__ = """
        Uniform log-likelihood with parameters lower={0} and upper={1}.
        
        Parameters
        ----------
        value : float
            :math:`lower \leq x \leq upper`
        """              
    
    return locals()

@quickclass
def Flat():
    """
    Uninformative log-likelihood that returns 0 regardless of 
    the passed value.
    """
    
    def logp(value):
        return zeros_like(value)
        
    logp.__doc__ = """
        Uninformative log-likelihood that returns 0 regardless of 
        the passed value.
        """
        
    return locals()

@quickclass
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

    .. note::
    - :math:`E(X) = \mu`
    - :math:`Var(X) = 1/\tau`

    """
    def logp(value):
        
        return switch(gt(tau , 0),
			 -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)
    
    logp.__doc__ = """
        Normal log-likelihood with paraemters mu={0} and tau={1}.
        
        Parameters
        ----------
        value : float
            Input data.
        """.format(mu, tau)
    
    return locals()

@quickclass
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

    .. note::
    - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    def logp(value):
        
        return switch(ge(value , 0) & le(value , 1) &
                  gt(alpha , 0) & gt(beta , 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)
                  
    logp.__doc__ = """
    Beta log-likelihood with parameters alpha={0} and beta={1}.
        
    Parameters
    ----------
    value : float
        0 < x < 1
    """.format(alpha, beta)
                  
    return locals()

@quickclass
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
    def logp(value):
        
        return switch(ge(value , 0) & ge(n , value) & ge(p , 0) & le(p , 1),
                   switch(ne(value , 0) , value*log(p), 0) + (n-value)*log(1-p) + factln(n)-factln(value)-factln(n-value),
                   -inf)
    logp.__doc__ = """
        Binomial log-likelihood with parameters n={0} and p={1}.
        
        Parameters
        ----------
        value : int 
            Number of successes, x > 0
        """.format(n,p)
    
    return locals()
    
@quickclass
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

    .. note::
    - :math:`E(X)=n\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=n\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    
    def logp(value):
        
        return switch(ge(value , 0) & gt(alpha , 0) & gt(beta , 0) & ge(n , value), 
                   gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(value+1)- gammaln(n-value +1) + gammaln(alpha+value)+ gammaln(n+beta-value)- gammaln(beta+alpha+n),
                   -inf)
    logp.__doc__ = """
        Beta-binomial log-likelihood with parameters alpha={0}, beta={1},
        and n={2}.
        
        Parameters
        ----------
        value : int
            x=0,1,\ldots,n
        """.format(alpha, beta, n)
    
    return locals()
    
@quickclass
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

    .. note::
    - :math:`E(x)= p`
    - :math:`Var(x)= p(1-p)`

    """
    def logp(value):
        
        return switch(ge(p , 0) & le(p , 1), 
                  switch(value, log(p), log(1-p)),
                  -inf)
    
    logp.__doc__ = """
        Bernoulli log-likelihood with parameter p={0}.
        
        Parameters
        ----------
        value : int
            Series of successes (1) and failures (0). :math:`x=0,1`
        """.format(p)
    
    return locals()

@quickclass
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
    """
    
    def logp(value):
        return switch(gt(lam  , 0) & gt(nu , 0),
                  gammaln((nu+1.0)/2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu/2.0) - (nu+1.0)/2.0 * log(1.0 + lam *(value - mu)**2/nu),
                  -inf)
    logp.__doc__ = """
        Student's T log-likelihood with paramters nu={0}, mu={1} and lam={2}.
        
        Parameters
        ----------
        value : float
            Input data
        """
    
    return locals()
    
@quickclass
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
    def logp(value):
        return switch(gt(beta , 0),
                  -log(beta) - log( 1 + ((value-alpha) / beta) ** 2 ),
                  -inf)
                  
    logp.__doc__ = """
        Cauchy log-likelihood with parameters alpha={0} and beta={0}.
        
        Parameters
        ----------
        value : float
            Input data.
        """.format(alpha, beta)
    return locals()
    
@quickclass
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
    def logp(value):
        return switch(ge(value , 0) & gt(alpha , 0) & gt(beta , 0),
                  -gammaln(alpha) + alpha*log(beta) - beta*value + switch(alpha != 1.0, (alpha - 1.0)*log(value), 0),
                  -inf)
    logp.__doc__ = """
        Gamma log-likelihood with paramters alpha={0} and beta={1}.
        
        Parameters
        ----------
        x : float
            math:`x \ge 0`
        """.format(alpha, beta)
    
    return locals()

@quickclass
def Poisson(mu):
    """
    Poisson log-likelihood.

    The Poisson is a discrete probability
    distribution.  It is often used to model the number of events
    occurring in a fixed period of time when the times at which events
    occur are independent. The Poisson distribution can be derived as
    a limiting case of the binomial distribution.

    .. math::
        f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    Parameters
    ----------
    mu : float
        Expected number of occurrences during the given interval, :math:`\mu \geq 0`.

    .. note::
       - :math:`E(x)=\mu`
       - :math:`Var(x)=\mu`

    """
    def logp(value):
        return switch( gt(mu,0),
               #factorial not implemented yet, so
               value * log(mu) - gammaln(value + 1) - mu,
               -inf)
               
    logp.__doc__ = """
        Poisson log-likelihood with parameters mu={0}.
        
        Parameters
        ----------
        x : int
            :math:`x \in \{0,1,2,...\}`
        """.format(mu)
    return locals()

@quickclass
def ConstantDist(c):
    def logp(value):
        return switch(eq(value, c), 0, -inf)
        
    logp.__doc__ = """
        Constant log-likelihood with parameter c={0}.
        
        Parameters
        ----------
        value : float or int
            Data value(s)
        """.format(c)
    return locals()

@quickclass
def ZeroInflatedPoisson(theta, z):
    def logp(value):
        return switch(z, 
                      Poisson(theta)(value), 
                      ConstantDist(0)(value))
    return locals()

@quickclass
def Bound(dist, lower = -inf, upper = inf):
    def logp(value):
        return switch(ge(value , lower) & le(value , upper),
                  dist.logp(value),
                  -inf)
    return locals()

@quickclass
def MvNormal(mu, Tau):
    """
    Multivariate normal log-likelihood

    .. math::
        f(x \mid \pi, T) = \frac{|T|^{1/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}
    """


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
