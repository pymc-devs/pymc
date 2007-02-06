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
'exponential', 'exponweib', 'gamma', 'geometric', 'half_normal', 'hypergeometric',
'inverse_gamma', 'lognormal', 'multinomial', 'multivariate_hypergeometric',
'multivariate_normal', 'negative_binomial', 'normal', 'poisson', 'uniform',
'weibull', 'wishart']


from PyMC2 import flib
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
      array([0, 1])
      >>> rbernoulli(.9, size=2)
      array([1, 1])
      >>> rbernoulli([.1,.9], 2)
      array([[0, 1],
             [0, 1]])
    """
    # Vectorized functions do not accept keyword arguments, so they
    # must be translated into positional arguments.

    # Find the order of the arguments.
    refargs, varargs, varkw, defaults = inspect.getargspec(func)
    vfunc = np.vectorize(func)
    npos = len(refargs)-len(defaults) # Number of pos. arg.
    nkwds = len(defaults) # Number of kwds args.

    def wrapper2(*args, **kwds):
        # First transform positional arguments into keywds args.
        if len(args)>0:
            for a,n in zip(args, refargs):
                kwds[n]=a
        l = {}
        for k, v in kwds.iteritems():
            l[k] = len(v)
        N = max(l.values())
        if N ==1 :
            return func(**kwds)
        else:
            result = []
            for i in range(N):
                tmp_kwds=jk
                result.append(func(**tmp_kwds))

    def wrapper(*args, **kwds):
        # First transform keyword arguments into positional arguments.
        n = len(args)
        if len(kwds) > 0:
            args = list(args)
            for i,k in enumerate(refargs[n:]):
                if k in kwds.keys():
                    args.append(kwds[k])
                else:
                    args.append(defaults[n-npos+i])

        r = [];s=[];largs=[];length=[1]
        for arg in args:
            length.append(np.size(arg))
        N = max(length)
        # Make sure all elements are iterable and have consistent lengths, ie
        # 1 or n, but not m and n.

        for arg in args:
            s = np.size(arg)
            t = type(arg)
            arr = np.empty(N, type)
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
      >>> beta_like(.4,1,2)
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
    r"""binomial_like(x, n, p)

    Binomial log-likelihood.  The discrete probability distribution of the
    number of successes in a sequence of n independent yes/no experiments,
    each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{1-x}

    :Parameters:
      x : float
        Number of successes, > 0.
      n : int
        Number of Bernoulli trials, > x.
      p : float
        Probability of success in each trial, :math:`p \in [0,1]`.

    :Note:
     - :math:`E(X)=np`
     - :math:`Var(X)=np(1-p)`
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
    r"""cauchy_like(x, alpha, beta)

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
    r"""chi2_like(x, k)

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
def rexponweib(alpha, k, loc, scale, size=1):
    """rexponweib(alpha, k, loc, scale, size=1)

    Random exponentiated Weibull variates.
    """
    q = random.uniform(size=size)
    r = flib.exponweib_ppf(q,alpha,k)
    return loc + r*scale

def exponweib_expval(alpha, k, loc, scale):
    return 'Not implemented yet.'

def exponweib_like(x, alpha, k, loc=0, scale=1):
    r"""exponweib_like(x,alpha,k,loc=0,scale=1)

    Exponentiated Weibull log-likelihood.

    .. math::
        f(x \mid \alpha,k,loc,scale)  & = \frac{\alpha k}{scale} (1-e^{-z^c})^{\alpha-1} e^{-z^c} z^{k-1} \\
        z & = \frac{x-loc}{scale}

    :Parameters:
      - `x` : > 0
      - `alpha` : Shape parameter
      - `k` : > 0
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
@randomwrap
def rgev(xi, mu=0, sigma=1, size=1):
    """rgev(xi, mu=0, sigma=0, size=1)

    Random generalized extreme value (GEV) variates.
    """
    q = random.uniform(size=size)
    z = flib.gev_ppf(q,xi)
    return (z+mu)*sigma

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
# Changed the return value
@randomwrap
def rgeometric(p, size=1):
    """rgeometric(p, size=1)

    Random geometric variates.
    """
    return random.geometric(p, size)

def geometric_expval(p):
    return (1. - p) / p

def geometric_like(x, p):
    r"""geometric_like(x, p)

    Geometric log-likelihood. The probability that the first success in a
    sequence of Bernoulli trials occurs after x trials.

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    :Parameters:
      x : int
        Number of trials before first success, > 0.
      p : float
        Probability of success on an individual trial, :math:`p \in [0,1]`

    :Note:
      - :math:`E(X)=1/p`
      - :math:`Var(X)=\frac{1-p}{p^2}`

    """
    constrain(p, 0, 1)
    constrain(x, lower=0)
    return flib.geometric(x, p)

# Half-normal----------------------------------------------
@randomwrap
def rhalf_normal(tau, size=1):
    """rhalf_normal(tau, size=1)

    Random half-normal variates.
    """
    return np.abs(random.normal(0, sqrt(1/tau), size))

def half_normal_expval(tau):
    return sqrt(0.5 * pi / array(tau))

def half_normal_like(x, tau):
    r"""half_normal_like(x, tau)
    
    Half-normal log-likelihood, a normal distribution with mean 0 and limited
    to the domain :math:`x \in [0, \infty)`.
    
    .. math::
        f(x \mid \tau) = \sqrt{\frac{2\tau}{\pi}}\exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    :Parameters:
      x : float
        :math:`x \ge 0`
      tau : float
        :math:`\tau > 0`
    
    """
    constrain(tau, lower=0)
    constrain(x, lower=0, allow_equal=True)
    return flib.hnormal(x, tau)

# Hypergeometric----------------------------------------------
def rhypergeometric(draws, success, failure, size=1):
    """rhypergeometric(draws, success, failure, size=1)

    Returns hypergeometric random variates.
    """
    return random.hypergeometric(success, failure, draws, size)

def hypergeometric_expval(draws, success, failure):
    return draws * success / (success+failure)

def hypergeometric_like(x, draws, success, failure):
    r"""hypergeometric_like(x, draws, success, failure)

    Hypergeometric log-likelihood. Discrete probability distribution that
    describes the number of successes in a sequence of draws from a finite
    population without replacement.

    .. math::
        f(x \mid draws, successes, failures)

    :Parameters:
      x : int
        Number of successes in a sample drawn from a population. 
        :math:`\max(0, draws-failures) \leq x \leq \min(draws, success)`
      draws : int
        Size of sample.
      success : int
        Number of successes in the population.
      failure : int
        Number of failures in the population.

    :Note:
      :math:`E(X) = \frac{draws failures}{success+failures}`
    """
    constrain(x, max(0, draws - failure), min(success, draws))
    return flib.hyperg(x, draws, success, success+failure)

# Inverse gamma----------------------------------------------
# This one doesn't look kasher. Check it up. 
@randomwrap
def rinverse_gamma(alpha, beta,size=1):
    """rinverse_gamma(alpha, beta,size=1)

    Random inverse gamma variates.
    """
    return random.gamma(1./beta,alpha, size)

def inverse_gamma_expval(alpha, beta):
    return array(alpha) / beta

def inverse_gamma_like(x, alpha, beta):
    r"""inverse_gamma_like(x, alpha, beta)

    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{x^{-\alpha - 1} \exp\{-\frac{1}{\beta x}}}
        {\Gamma(\alpha)\beta^\alpha}

    :Parameters:
      x : float
        x > 0
      alpha : float
        Shape parameter, :math:`\alpha > 0`.
      beta : float
        Scale parameter, :math:`\beta > 0`.

    :Note:
      :math:`E(X)=\frac{1}{\beta(\alpha-1)}` for :math:`\alpha > 1`.
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return flib.igamma(x, alpha, beta)

# Lognormal----------------------------------------------
@randomwrap
def rlognormal(mu, tau,size=1):
    """rlognormal(mu, tau,size=1)

    Return random lognormal variates.
    """
    return random.lognormal(mu, sqrt(1./tau),size)

def lognormal_expval(mu, tau):
    return np.exp(mu + 1./2/tau)

def lognormal_like(x, mu, tau):
    r"""lognormal_like(x, mu, tau)

    Log-normal log-likelihood. Distribution of any random variable whose
    logarithm is normally distributed. A variable might be modeled as
    log-normal if it can be thought of as the multiplicative product of many
    small independent factors.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi x}}
        \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}

    :Parameters:
      x : float
        x > 0
      mu : float
        Location parameter.
      tau : float
        Scale parameter, > 0.

    :Note:
      :math:`E(X)=e^{\mu+\frac{1}{2\tau}}`
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return flib.lognormal(x,mu,tau)
    
# Multinomial----------------------------------------------
@randomwrap
def rmultinomial(n,p,size=1):
    """rmultinomial(n,p,size=1)

    Random multinomial variates.
    """
    return random.multinomial(n,p,size)

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
    r"""multivariate_hypergeometric_like(x, m)
    
    Multivariate hypergeometric log-likelihood

    .. math::
        f(x \mid \pi, T) = \frac{T^{n/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}
    
    x < m
    """
    constrain(x, upper=m)
    return flib.mvhyperg(x, m)

# Multivariate normal--------------------------------------
def rmultivariate_normal(mu, tau, size=1):
    """rmultivariate_normal(mu, tau, size=1)

    Random multivariate normal variates.
    """
    return random.multivariate_normal(mu, inverse(tau), size)

def multivariate_normal_expval(mu, tau):
    return mu

def multivariate_normal_like(x, mu, tau):
    r"""multivariate_normal_like(x, mu, tau)
    
    Multivariate normal log-likelihood

    .. math::
        f(x \mid \pi, T) = \frac{T^{n/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}
    
    x: (k,n)
    mu: (k,n) or (k,1)
    tau: (k,k)
    \trace(tau) > 0
    """
    constrain(np.diagonal(tau), lower=0)
    return flib.vec_mvnorm(x, mu, tau)

# Negative binomial----------------------------------------
@randomwrap
def rnegative_binomial(mu, alpha, size=1):
    """rnegative_binomial(mu, alpha, size=1)

    Random negative binomial variates.
    """
    return random.negative_binomial(alpha, alpha / (mu + alpha),size)

def negative_binomial_expval(mu, alpha):
    return mu


def negative_binomial_like(x, mu, alpha):
    r"""negative_binomial_like(x, mu, alpha)
    
    Negative binomial log-likelihood

    .. math::
        f(x \mid r, p) = \frac{(x+r-1)!}{x! (r-1)!} p^r (1-p)^x
    
    x > 0, mu > 0, alpha > 0
    """
    constrain(mu, lower=0)
    constrain(alpha, lower=0)
    constrain(x, lower=0)
    return flib.negbin2(x, mu, alpha)

# Normal---------------------------------------------------
@randomwrap
def rnormal(mu, tau,size=1):
    """rnormal(mu, tau, size=1)

    Random normal variates.
    """
    return random.normal(mu, 1./sqrt(tau), size)

def normal_expval(mu, tau):
    return mu

def normal_like(x, mu, tau):
    r"""normal_like(x, mu, tau)
    
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}


    :Parameters:
      x : float
        Input data.
      mu : float
        Mean of the distribution.
      tau : float
        Precision of the distribution, > 0.
    
    :Note:
      - :math:`E(X) = \mu`
      - :math:`Var(X) = 1/\tau`
    
    """
    constrain(tau, lower=0)
    return flib.normal(x, mu, tau)


# Poisson--------------------------------------------------
@randomwrap
def rpoisson(mu, size=1):
    """rpoisson(mu, size=1)

    Random poisson variates.
    """
    return random.poisson(mu,size)


def poisson_expval(mu):
    return mu


def poisson_like(x,mu):
    r"""poisson_like(x,mu)

    Poisson log-likelihood. The Poisson a discrete probability distribution.
    It expresses the probability of a number of events occurring in a fixed
    period of time if these events occur with a known average rate, and are
    independent of the time since the last event. The Poisson distribution can
    be derived as a limiting case of the binomial distribution.

    .. math::
        f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    :Parameters:
      x : int
        :math:`x \in {0,1,2,...}`
      mu : float
        Expected number of occurrences that occur during the given interval,
        :math:`\mu \geq 0`.

    :Note:
      - :math:`E(x)=\mu`
      - :math:`Var(x)=\mu`
    """
    constrain(x, lower=0,allow_equal=True)
    constrain(mu, lower=0,allow_equal=True)
    return flib.poisson(x,mu)

# Uniform--------------------------------------------------
@randomwrap
def runiform(lower, upper, size=1):
    """runiform(lower, upper, size=1)

    Random uniform variates.
    """
    return random.uniform(lower, upper, size)

def uniform_expval(lower, upper):
    return (upper - lower) / 2.

def uniform_like(x,lower, upper):
    r"""uniform_like(x, lower, upper)

    Uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    :Parameters:
      x : float
       :math:`lower \geq x \geq upper`
      lower : float
        Lower limit.
      upper : float
        Upper limit.
    """
    return flib.uniform_like(x, lower, upper)

# Weibull--------------------------------------------------
@randomwrap
def rweibull(alpha, beta,size=1):
    tmp = -log(runiform(0, 1, size))
    return beta * (tmp ** (1. / alpha))

def weibull_expval(alpha,beta):
    return beta * gammaln((alpha + 1.) / alpha)

def weibull_like(x, alpha, beta):
    r"""weibull_like(x, alpha, beta)

    Weibull log-likelihood

    .. math::
        f(x \mid \alpha, \beta) = \frac{\alpha x^{\alpha - 1}
        \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    :Parameters:
      x : float
        :math:`x \ge 0`
      alpha : float
        > 0
      beta : float
        > 0

    :Note:
      - :math:`E(x)=\beta \Gamma(1+\frac{1}{\alpha}`
      - :math:`Var(x)=\beta^2 \Gamma(1+\frac{2}{\alpha} - \mu^2`
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
    r"""wishart_like(X, n, Tau)
    
    Wishart log-likelihood

    .. math::
        f(X \mid n, T) = {\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

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


