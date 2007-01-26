#-------------------------------------------------------------------
# Decorate fortran functions from PyMC.flib to ease argument passing
#-------------------------------------------------------------------
# TODO: Deal with functions that take correlation matrices as arguments.wishart, normal,?
# TODO: test and finalize vectorized multivariate normal like.
# TODO: Add exponweib_expval (how?)
# TODO: Complete docstrings with LaTeX formulas from the tutorial.
# TODO: Do we add size arguments to random generators ? It shouldn't be a 
#   problem, except that vector arguments + size arg. is a bit confusing.   

__docformat__='reStructuredText'
availabledistributions = ['bernoulli', 'beta', 'binomial', 'cauchy', 'chi2', 'dirichlet',
'exponential', 'gamma', 'geometric', 'half_normal', 'hypergeometric',
'inverse_gamma', 'lognormal', 'multinomial', 'multivariate_hypergeometric',
'multivariate_normal', 'negative_binomial', 'normal', 'poisson', 'uniform',
'weibull']


import flib
import numpy as np
from utils import LikelihoodError
from numpy import inf, random, sqrt, log, size, tan, pi
#from decorators import * #Vectorize, fortranlike_method, priorwrap, randomwrap
# Import utility functions
import inspect
random_number = random.random
inverse = np.linalg.pinv


#-------------------------------------------------------------
# Light decorators
#-------------------------------------------------------------

def Vectorize(f):
    """Wrapper to vectorize a scalar function."""
    return np.vectorize(f)

def randomwrap(func):
    """
    Decorator for random value generators

    Allows passing of sequence of parameters, as well as a size argument. 
    
    Convention:
    
      - If size=1 and the parameters are all scalars, return a scalar. 
      - If size=1, the random variates are 1D.
      - If the parameters are scalars and size > 1, the random variates are 1D.
      - If size > 1 and the parameters are sequences, the random variates are
        aligned as (size, max(length)), where length is the parameters size. 
    
    
    :Example:
      >>> rbernoulli(.1)
      0
      >>> rbernoulli([.1,.9])
      array([0,1])
      >>> rbernoulli(.9, size=2)
      array([1,1])
      >>> rbernoulli([.1,.9], 2)
      array([[0, 1],
      [0, 1]])
    """
    # Vectorized functions do not accept keyword arguments, so they
    # must be translated into positional arguments.

    # Find the order of the arguments.
    refargs, varargs, varkw, defaults = inspect.getargspec(func)
    vfunc = np.vectorize(func)
    def wrapper(*args, **kwds):
        # First transform keyword arguments into positional arguments.
        if len(kwds) > 0:
            args = list(args)
            for k in refargs:
                if k in kwds.keys(): args.append(kwds[k])
        
        r = [];s=[];largs=[];length=[1]   
        for arg in args:
            length.append(np.size(arg))
        N = max(length)
        # Make sure all elements are iterable and have consistent lengths, ie
        # 1 or n, but not m and n. 
        for arg in args:
            arr = np.empty(N)
            s = np.size(arg)
            if s == 1:
                arr.fill(arg)
            elif s == N:
                arr = np.array(arg)
            else:
                raise 'Arguments size not allowed.', s
            largs.append(arr)
        
        for arg in zip(*largs):
            r.append(func(*arg))
        
        size = arg[-1] 
        vec_params = len(r)>1
        if size > 1 and vec_params:
            return np.atleast_2d(r).T
        elif vec_params or size > 1:
            return np.concatenate(r)
        else: # Scalar case
            return r[0][0]
            
    wrapper.__doc__ = func.__doc__
    return wrapper

#-------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------

def constrain(value, lower=-inf, upper=inf, allow_equal=False):
    """Apply interval constraint on parameter value."""
    ok = flib.constrain(value, lower, upper, allow_equal)
    if ok == 0:
        raise LikelihoodError

def standardize(x, loc=0, scale=1):
    """Standardize x

    Return (x-loc)/scale
    """
    return flib.standardize(x,loc,scale)

@Vectorize
def gammaln(x):
    """Logarithm of the Gamma function"""
    return flib.gamfun(x)

def expand_triangular(X,k):
    """Expand flattened triangular matrix."""
    X = X.tolist()
    # Unflatten matrix
    Y = array([[0] * i + X[i * k - (i * (i - 1)) / 2 : i * k + (k - i)] for i in range(k)])
    # Loop over rows
    for i in range(k):
        # Loop over columns
        for j in range(k):
            Y[j, i] = Y[i, j]
    return Y


""" Loss functions """

absolute_loss = lambda o,e: absolute(o - e)

squared_loss = lambda o,e: (o - e)**2

chi_square_loss = lambda o,e: (1.*(o - e)**2)/e

def GOFpoints(x,y,expval,loss):
    return sum(np.transpose([loss(x, expval), loss(y, expval)]), 0)

#--------------------------------------------------------
# Statistical distributions
# random generator, expval, log-likelihood
#--------------------------------------------------------

# Bernoulli----------------------------------------------
@randomwrap
def rbernoulli(p,size=1):
    """rbernoulli(p,size=1)
    
    Random Bernoulli variates.
    """
    return random.binomial(1,p,size)

def bernoulli_expval(p):
    """Goodness of fit for bernoulli."""
    return p


def bernoulli_like(x, p):
    r"""bernoulli_like(x, p)
    
    Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes (x=1) and 
    failures (x=0).   

    .. math:: 
        f(x \mid p) = p^{x- 1} (1-p)^{1-x}

    :Parameters:
      - `x`: Series of successes (1) and failures (0). :math:`x=0,1`
      - `p`: Probability of success. :math:`0 < p < 1`
    
    :Example: 
      >>> bernoulli_like([0,1,0,1], .4)
      -2.8542325496673584
    
    :Note:
      - :math:`E(x)= p`
      - :math:`Var(x)= p(1-p)`
    
    """
    constrain(p, 0, 1,allow_equal=True)
    constrain(x, 0, 1,allow_equal=True)
    return flib.bernoulli(x, p)


# Beta----------------------------------------------
@randomwrap
def rbeta(alpha, beta, size=1):
    """rbeta(alpha, beta, size=1)
    
    Random beta variates.
    """
    return random.beta(alpha, beta,size)

def beta_expval(x,alpha, beta):
    expval = 1.0 * alpha / (alpha + beta)
    return expval


def beta_like(x, alpha, beta):
    r"""beta_like(x, alpha, beta)
    
    Beta log-likelihood.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}
        
    :Parameters:
      - `x`: 0 < x < 1
      - `alpha`: > 0
      - `beta`: > 0

    :Example:
      >>> flib.beta(.4,1,2)
      0.18232160806655884

    :Note:
      - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
      - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    
    """
    constrain(alpha, lower=0, allow_equal=True)
    constrain(beta, lower=0, allow_equal=True)
    constrain(x, 0, 1, allow_equal=True)
    return flib.beta_like(x, alpha, beta)

# Binomial----------------------------------------------
@randomwrap
def rbinomial(n,p,size=1):
    """rbinomial(n,p,size=1)
    
    Random binomial variates.
    """
    return random.binomial(n,p,size)
    
def binomial_expval(x,n,p):
    return p*n

def binomial_like(x, n, p):
    """binomial_like(x, n, p)
    
    Binomial log-likelihood.  The discrete probability distribution of the 
    number of successes in a sequence of n independent yes/no experiments, 
    each of which yields success with probability p.
    
    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!}p^x (1-p)^{1-x}
        
    :Parameters:
      x : float 
        Number of successes, > 0.
      n : int
        Number of Bernoulli trials, > x.
      p : float
        Probability of success in each trial, :math:`p \in [0,1]`.     

    :Note:
      :math:`E(X)=np`
      :math:`Var(X)=np(1-p)`   
    """
    constrain(p, 0, 1)
    constrain(n, lower=x)
    constrain(x, 0)
    return flib.binomial(x,n,p)

# Categorical----------------------------------------------
# GOF not working yet, because expval not conform to wrapper spec.
@randomwrap
def rcategorical(probs, minval=0, step=1):
    return flib.rcat(probs, minval, step)

def categorical_expval(probs, minval=0, step=1):
    return sum([p*(minval + i*step) for i, p in enumerate(probs)])

def categorical_like( x, probs, minval=0, step=1):
    """Categorical log-likelihood.
    Accepts an array of probabilities associated with the histogram,
    the minimum value of the histogram (defaults to zero),
    and a step size (defaults to 1).
    """
    # Normalize, if not already
    if sum(probs) != 1.0: probs = probs/sum(probs)
    return flib.categorical(x, probs, minval, step)


# Cauchy----------------------------------------------
@randomwrap
def rcauchy(alpha, beta, size=1):
    """rcauchy(alpha, beta, size=1)
    
    Returns Cauchy random variates.
    """
    return alpha + beta*tan(pi*random_number(size) - pi/2.0)

def cauchy_expval(alpha, beta):
    return alpha

# In wikipedia, the arguments name are k, x0. 
def cauchy_like(x, alpha, beta):
    """cauchy_like(x, alpha, beta)
    
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution. 

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}
    
    :Parameters:
      - `alpha` : Location parameter.
      - `beta`: Scale parameter > 0.
    
    :Note:
      - Mode and median are at alpha.
    """
    constrain(beta, lower=0)
    return flib.cauchy(x,alpha,beta)

# Chi square----------------------------------------------
@randomwrap
def rchi2(k, size=1):
    """rchi2(k, size=1)
    
    Random :math:`\chi^2` variates.
    """
    return random.chisquare(k, size)

def chi2_expval(k):
    return k

def chi2_like(x, k):
    """chi2_like(x, k)
    
    Chi-squared :math:`\chi^2` log-likelihood.

    .. math::
        f(x \mid k) = \frac{x^{\frac{k}{2}-1}e^{-2x}}{\Gamma(\frac{k}{2}) \frac{1}{2}^{k/2}} 

    :Parameters:
      x : float
        :math:`\ge 0`
      k : int 
        Degrees of freedom > 0
    
    :Note:
      - :math:`E(X)=k`
      - :math:`Var(X)=2k`
      
    """
    constrain(x, lower=0)
    constrain(k, lower=0)
    return flib.gamma(x, 0.5*k, 2)

# Dirichlet----------------------------------------------
def rdirichlet(theta, size=1):
    """rdirichlet(theta, size=1)
    
    Dirichlet random variates.
    """
    gammas = rgamma(theta,1,size)
    if size > 1 and np.size(theta) > 1:
        return (gammas.transpose()/gammas.sum(1)).transpose()
    elif np.size(theta)>1:
        return gammas/gammas.sum()
    else:
        return gammas
        
def dirichlet_expval(theta):
    sumt = sum(theta)
    expval = theta/sumt
    return expval

def dirichlet_like(x, theta):
    r"""dirichlet_like(x, theta)
    
    Dirichlet log-likelihood.
    
    This is a multivariate continuous distribution.

    .. math::
        f(\mathbf{x}) = \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)} \prod_{i=1}^k x_i^{\theta_i - 1}

    :Parameters:
      x : (n,k) array 
        Where `n` is the number of samples and `k` the dimension. 
        :math:`0 < x_i < 1`,  :math:`\sum_{i=1}^k x_i = 1`
      theta : (n,k) or (1,k) float
        :math:`\theta > 0`
    """
    
    constrain(theta, lower=0)
    constrain(x, lower=0)
    constrain(sum(x), upper=1) #??
    return flib.dirichlet(x,theta)

# Exponential----------------------------------------------
@randomwrap
def rexponential(beta, size=1):
    """rexponential(beta)
    
    Exponential random variates.
    """
    return random.exponential(beta,size)

def exponential_expval(beta):
    return beta


def exponential_like(x, beta):
    r"""exponential_like(x, beta)
    
    Exponential log-likelihood. 
    
    The exponential distribution is a special case of the gamma distribution 
    with alpha=1. It often describes the duration of an event. 
    
    .. math::
        f(x \mid \beta) = \frac{1}{\beta}e^{-x/\beta}
    
    :Parameters:
      x : float
        :math:`x \ge 0`
      beta : float
        Survival parameter :math:`\beta > 0`
    
    :Note:
      - :math:`E(X) = \beta`
      - :math:`Var(X) = \beta^2`
    """
    constrain(x, lower=0)
    constrain(beta, lower=0)
    return flib.gamma(x, 1, beta)

# Exponentiated Weibull-----------------------------------
@randomwrap
def rexponweib(a, c, loc, scale, size=1):
    """rexponweib(a, c, loc, scale, size=1)
    
    Random exponentiated Weibull variates.
    """
    q = random.uniform(size)
    r = flib.exponweib_ppf(q,a,c)
    return loc + r*scale

def exponweib_like(x, a, c, loc=0, scale=1):
    """exponweib_like(x,a,c,loc=0,scale=1)
    
    Exponentiated Weibull log-likelihood.

    .. math::
        pdf(x) & = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1) \\
        z & = \frac{x-loc}{scale}
    
    :Parameters:
      - `x` : > 0
      - `a` : Shape parameter
      - `c` : > 0
      - `loc` : Location parameter
      - `scale` : Scale parameter > 0.

    """
    return flib.exponweib(x,a,c,loc,scale)

# Gamma----------------------------------------------
@randomwrap
def rgamma(alpha, beta,size=1):
    """rgamma(alpha, beta,size=1)
    
    Random gamma variates.
    """
    return random.gamma(alpha,beta,size)

def gamma_expval(alpha, beta):
    expval = array(alpha) / beta
    return expval

def gamma_like(x, alpha, beta):
    r"""gamma_like(x, alpha, beta)
    
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.
    
    .. math::
        f(x \mid \alpha, \beta) = \frac{x^{\alpha-1}e^{-x/\beta}}{\Gamma(\alpha) \beta^{\alpha}}
    
    :Parameters:
      x : float
        :math:`x \ge 0`
      alpha : float
        Shape parameter :math:`\alpha > 0`.
      beta : float
        Scale parameter :math:`\beta > 0`.
    
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return flib.gamma(x, alpha, beta)


# GEV Generalized Extreme Value ------------------------
def gev_like(x, xi, mu=0, sigma=0):
    r"""gev_like(x, xi, mu=0, sigma=0)
    
    Generalized Extreme Value log-likelihood

    .. math::
        pdf(x \mid \xi,\mu,\sigma) = \frac{1}{\sigma}(1 + \xi z)^{-1/\xi-1}\exp{-(1+\xi z)^{-1/\xi}}
    
    where :math:`z=\frac{x-\mu}{\sigma}`
    
    .. math::
        \sigma & > 0,\\
        x & > \mu-\sigma/\xi \text{ if } \xi > 0,\\
        x & < \mu-\sigma/\xi \text{ if } \xi < 0\\
        x & \in [-\infty,\infty] \text{ if } \xi = 0
        
    """
    return flib.gev(x,xi,loc, scale)

# Geometric----------------------------------------------
@randomwrap
def rgeometric(p, size=1):
    """rgeometric(p, size=1)
    
    Random geometric variates.
    """
    return random.negative_binomial(1, p, size)

def geometric_expval(p):
    return (1. - p) / p

def geometric_like(x, p):
    """geometric_like(x, p)
    
    Geometric log-likelihood. The probability that the first success in a 
    sequence of Bernoulli trials occurs after x trials. 

    .. math::
        f(x \mid p) = p(1-p)^{x-1}
        
    :Parameters:
      x : int
        Number of trials before first success, > 0.
      p : float 
        Probability of success on each trial, :math:`p \in [0,1]`

    :Note:
      - :math:`E(X)=1/p`
      - :math:`Var(X)=\frac{1-p}{p^2}
    
    """
    constrain(p, 0, 1)
    constrain(x, lower=0)
    return flib.negbin2(x, 1, p)

# Half-normal----------------------------------------------
@randomwrap
def rhalf_normal(tau, size=1):
    """rhalf_normal(tau, size=1)
    
    Random half-normal variates.
    """
    return random.normal(0, sqrt(1/tau), size)

def half_normal_expval(tau):
    return sqrt(0.5 * pi / array(tau))

def half_normal_like(x, tau):
    """Half-normal log-likelihood

    half_normal_like(x, tau)

    x > 0, tau > 0
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return flib.hnormal(x, tau)

# Hypergeometric----------------------------------------------
# TODO: replace with random.hypergeometric
# TODO: Select a uniform convention across functions.
def rhypergeometric(draws, red, total, n=None):
    """Returns n hypergeometric random variates of size 'draws'"""

    urn = [1]*red + [0]*(total-red)

    if n:
        return [sum(urn[i] for i in permutation(total)[:draws]) for j in range(n)]
    else:
        return sum(urn[i] for i in permutation(total)[:draws])

def hypergeometric_expval(n,m,N):
    return n * (m / N)

def hypergeometric_like(x, n, m, N):
    """
    Hypergeometric log-likelihood

    hypergeometric_like(x, n, m, N)

    x \in [\max(0, n-N+m), \min(m,n)], m < N, n < N

    Models the probability of drawing x successful draws in n
    draws from N total balls of which m are successes.
    """
    constrain(m, upper=N)
    constrain(n, upper=N)
    constrain(x, max(0, n - N + m), min(m, n))
    return flib.hyperg(x, n, m, N)

# Inverse gamma----------------------------------------------
# Looks this one is identical to rgamma, this is strange.
@randomwrap
def rinverse_gamma(alpha, beta,size=1):
    """rinverse_gamma(alpha, beta,size=1)
    
    Random inverse gamma variates.
    """
    pass

def inverse_gamma_expval(alpha, beta):
    return array(alpha) / beta


def inverse_gamma_like(x, alpha, beta):
    """inverse_gamma_like(x, alpha, beta)
    
    Inverse gamma log-likelihood
   
    x > 0, alpha > 0, beta > 0
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return flib.igamma(x, alpha, beta)

# Lognormal----------------------------------------------
@randomwrap
def rlognormal(mu, tau):
    return random.normal(mu, sqrt(1./tau))

def lognormal_expval(mu, tau):
    return mu


def lognormal_like(x, mu, tau):
    """Log-normal log-likelihood

    lognormal_like(x, mu, tau)

    x > 0, tau > 0
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return flib.lognormal(x,mu,tau)

# Multinomial----------------------------------------------
@randomwrap
def rmultinomial(n,p):
    return random.multinomial

def multinomial_expval(n,p):
    array([pr * n for pr in p])


def multinomial_like(x, n, p):
    """Multinomial log-likelihood with k-1 bins

    multinomial_like(x, n, p)

    x > 0, p > 0, \sum p < 1, \sum x < n
    """
    constrain(p, lower=0)
    constrain(x, lower=0)
    constrain(sum(p), upper=1)
    constrain(sum(x), upper=n)
    return flib.multinomial(x, n, p)

# Multivariate hypergeometric------------------------------
# Hum, this is weird. multivariate_hypergeometric_like takes one parameters m
# and rmultivariate_hypergeometric has two. n= sum(x) ???
def rmultivariate_hypergeometric(draws, colors, n=None):
    """ Returns n multivariate hypergeometric draws of size 'draws'"""

    urn = concatenate([[i]*count for i,count in enumerate(colors)])

    if n:
        draw = [[urn[i] for i in permutation(len(urn))[:draws]] for j in range(n)]

        return [[sum(draw[j]==i) for i in range(len(colors))] for j in range(n)]
    else:
        draw = [urn[i] for i in permutation(len(urn))[:draws]]

        return [sum(draw==i) for i in range(len(colors))]

def multivariate_hypergeometric_expval(m):
    return n * (array(m) / sum(m))


def multivariate_hypergeometric_like(x, m):
    """Multivariate hypergeometric log-likelihood

    multivariate_hypergeometric_like(x, m)

    x < m
    """
    constrain(x, upper=m)
    return flib.mvhyperg(x, m)

# Multivariate normal--------------------------------------
def rmultivariate_normal(mu, tau):
    return random.multivariate_normal(mu, inverse(tau))

def multivariate_normal_expval(mu, tau):
    return mu

def multivariate_normal_like(x, mu, tau):
    r"""Multivariate normal log-likelihood

    multivariate_normal_like(x, mu, tau)

    x: (k,n)
    mu: (k,n) or (k,1)
    tau: (k,k)
    \trace(tau) > 0
    """
    constrain(np.diagonal(tau), lower=0)
    return flib.vec_mvnorm(x, mu, tau)

# Negative binomial----------------------------------------
@randomwrap
def rnegative_binomial(mu, alpha):
    return random.negative_binomial(alpha, alpha / (mu + alpha))

def negative_binomial_expval(mu, alpha):
    return mu


def negative_binomial_like(x, mu, alpha):
    """Negative binomial log-likelihood

    negative_binomial_like(x, mu, alpha)

    x > 0, mu > 0, alpha > 0
    """
    constrain(mu, lower=0)
    constrain(alpha, lower=0)
    constrain(x, lower=0)
    return flib.negbin2(x, mu, alpha)

# Normal---------------------------------------------------
@randomwrap
def rnormal(mu, tau):
    return random.normal(mu, 1./sqrt(tau))

def normal_expval(mu, tau):
    return mu


def normal_like(x, mu, tau):
    """Normal log-likelihood

    normal_like(x, mu, tau)

    tau > 0
    """
    constrain(tau, lower=0)
    return flib.normal(x, mu, tau)


# Poisson--------------------------------------------------
@randomwrap
def rpoisson(mu,size=1):
    return random.poisson(mu,size)

def poisson_expval(mu):
    return mu


def poisson_like(x,mu):
    """Poisson log-likelihood

    poisson_like(x,mu)

    x \geq 0, mu \geq 0
    """
    constrain(x, lower=0,allow_equal=True)
    constrain(mu, lower=0,allow_equal=True)
    return flib.poisson(x,mu)

# Uniform--------------------------------------------------
@randomwrap
def runiform(lower, upper, size=1):
    """Uniform random generator

    runiform(lower, upper, size=1)
    """
    return random.uniform(lower, upper, size)

def uniform_expval(lower, upper):
    return (upper - lower) / 2.

def uniform_like_python(x, lower, upper):
    """Uniform log-likelihood"""
    x = np.atleast_1d(x)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    constrain(x, lower=lower, upper=upper, allow_equal=True)
    return sum(np.log(1. / (np.array(upper) - np.array(lower))))
uniform_like_python._PyMC = True


def uniform_like(x,lower, upper):
    """Uniform log-likelihood

    uniform_like(x,lower, upper)

    x \in [lower, upper]
    """
    return flib.uniform_like(x,lower, upper)

# Weibull--------------------------------------------------
@randomwrap
def rweibull(alpha, beta,size=1):
    tmp = -log(runiform(0, 1, size))
    return beta * (tmp ** (1. / alpha))

def weibull_expval(alpha,beta):
    return beta * gammaln((alpha + 1.) / alpha)

def weibull_like(x, alpha, beta):
    """Weibull log-likelihood

    weibull_like(x, alpha, beta)

    x > 0, alpha > 0, beta > 0
    """
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    constrain(x, lower=0)
    return flib.weibull(x, alpha, beta)

# Wishart---------------------------------------------------
def rwishart(n, Tau, m=None):
    """Returns Wishart random matrices"""
    sigma = inverse(Tau)
    D = [i for i in ravel(t(chol(sigma))) if i]
    np = len(sigma)

    if m:
        return [expand_triangular(flib.wshrt(D, n, np), np) for i in range(m)]
    else:
        return expand_triangular(flib.wshrt(D, n, np), np)

def wishart_expval(n, Tau):
    return n * array(Tau)

def wishart_like(X, n, Tau):
    """Wishart log-likelihood

    wishart_like(X, n, Tau)

    X, T symmetric and positive definite
    n > 0
    """
    constrain(np.diagonal(Tau), lower=0)
    constrain(n, lower=0)
    return flib.wishart(X, n, Tau)

# -----------------------------------------------------------

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()


