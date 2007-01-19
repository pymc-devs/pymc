#-------------------------------------------------------------------
# Decorate fortran functions from PyMC.flib to ease argument passing
#-------------------------------------------------------------------
# TODO: Deal with functions that take correlation matrices as arguments.wishart, normal,?
# TODO: test and finalize vectorized multivariate normal like.
# TODO: Add exponweib_expval (how?)



availabledistributions = ['bernoulli', 'beta', 'binomial', 'cauchy', 'chi2', 'dirichlet',
'exponential', 'gamma', 'geometric', 'half_normal', 'hypergeometric',
'inverse_gamma', 'lognormal', 'multinomial', 'multivariate_hypergeometric',
'multivariate_normal', 'negative_binomial', 'normal', 'poisson', 'uniform',
'weibull']


import flib
import numpy as np
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
    =====================================

    Vectorize random value generation functions so an array of parameters may
    be passed.
    """
    # Vectorized functions do not accept keyword arguments, so they
    # must be translated into positional arguments.

    # Find the order of the arguments.
    refargs, varargs, varkw, defaults = inspect.getargspec(func)
    vfunc = np.vectorize(func)
    def wrapper(*args, **kwds):
        """Transform keywords arguments into positional arguments and feed them
        to a vectorized random function."""
        if len(kwds) > 0:
            args = list(args)
            for k in refargs:
                if k in kwds.keys(): args.append(kwds[k])

        return vfunc(*args)
    wrapper.__doc__ = func.__doc__
    return wrapper

#-------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------
class LikelihoodError(ValueError):
    "Log-likelihood is invalid or negative informationnite"

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
def rbernoulli(p):
    return random.binomial(1,p)

def bernoulli_expval(p):
    """Goodness of fit for bernoulli."""
    return p


def bernoulli_like(x, p):
    """Bernoulli log-likelihood

    bernoulli_like(x, p)

    p \in [0,1], x \in [0,1]
    """
    constrain(p, 0, 1,allow_equal=True)
    constrain(x, 0, 1,allow_equal=True)
    return flib.bernoulli(x, p)


# Beta----------------------------------------------
@randomwrap
def rbeta(alpha, beta):
    return random.beta(alpha, beta)

def beta_expval(x,alpha, beta):
    expval = 1.0 * alpha / (alpha + beta)
    return expval


def beta_like(x, alpha, beta):
    """Beta log-likelihood

    beta_like(x, alpha, beta)

    x in [0,1], alpha >= 0, beta >= 0
    """
    constrain(alpha, lower=0, allow_equal=True)
    constrain(beta, lower=0, allow_equal=True)
    constrain(x, 0, 1, allow_equal=True)
    return flib.beta_like(x, alpha, beta)

# Binomial----------------------------------------------
@randomwrap
def rbinomial(n,p):
    return random.binomial(n,p)

def binomial_expval(x,n,p):
    expval = p * n
    return expval


def binomial_like(x, n, p):
    """Binomial log-likelihood

    binomial_like(x, n, p)

    p \in [0,1], n > x, x > 0
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
def rcauchy(alpha, beta, n=None):
    """Returns Cauchy random variates"""
    N = n or max(size(alpha), size(beta))
    return alpha + beta*tan(pi*random_number(n) - pi/2.0)

def cauchy_expval(alpha, beta):
    return alpha

def cauchy_like(x, alpha, beta):
    """Cauchy log-likelhood

    cauchy_like(x, alpha, beta)

    beta > 0
    """
    constrain(beta, lower=0)
    return flib.cauchy(x,alpha,beta)

# Chi square----------------------------------------------
@randomwrap
def rchi2(df):
    return random.chisquare(df)

def chi2_expval(df):
    return df


def chi2_like(x, df):
    """Chi-squared log-likelihood

    chi2_like(x, df)

    x > 0, df > 0
    """
    constrain(x, lower=0)
    constrain(df, lower=0)
    return flib.gamma(y, 0.5*df, 2)

# Dirichlet----------------------------------------------
@randomwrap
def rdirichlet(alphas, n=None):
    """Returns Dirichlet random variates"""
    if n:
        gammas = transpose([rgamma(alpha,1,n) for alpha in alphas])

        return array([g/sum(g) for g in gammas])
    else:
        gammas = array([rgamma(alpha,1) for alpha in alphas])

        return gammas/sum(gammas)

def dirichlet_expval(theta):
    sumt = sum(theta)
    expval = theta/sumt
    return expval

def dirichlet_like(x, theta):
    """Dirichlet log-likelihood

    dirichlet_like(x, theta)

    theta > 0, x > 0, \sum x < 1
    """
    constrain(theta, lower=0)
    constrain(x, lower=0)
    constrain(sum(x), upper=1)
    return flib.dirichlet(x,theta)

# Exponential----------------------------------------------
@randomwrap
def rexponential(beta):
    return random.exponential(beta)

def exponential_expval(beta):
    return beta


def exponential_like(x, beta):
    """Exponential log-likelihood

    exponential_like(x, beta)

    x > 0, beta > 0
    """
    constrain(x, lower=0)
    constrain(beta, lower=0)
    return flib.gamma(x, 1, beta)

# Exponentiated Weibull-----------------------------------
@randomwrap
def rexponweib(a, c, loc, scale, size=1):
    q = random.uniform(size)
    r = flib.exponweib_ppf(q,a,c)
    return loc + r*scale

def exponweib_like(x, a, c, loc=0, scale=1):
    """Exponentiated Weibull log-likelihood

    exponweib_like(x,a,c,loc=0,scale=1)

    x > 0, c > 0, scale >0
    """
    return flib.exponweib(x,a,c,loc,scale)

# Gamma----------------------------------------------
@randomwrap
def rgamma(alpha, beta):
    return random.gamma(1./beta, alpha)

def gamma_expval(alpha, beta):
    expval = array(alpha) / beta
    return expval

def gamma_like(x, alpha, beta):
    """Gamma log-likelihood

    gamma_like(x, alpha, beta)

    x > 0, alpha > 0, beta > 0
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return flib.gamma(x, alpha, beta)


# GEV Generalized Extreme Value ------------------------
def gev_like(x, xi, loc=0, scale=0):
    r"""GEV log-likelihood

    gev_like(x, xi, mu=0, sigma=0)

    .. latex-math::
    pdf(x|xi,\mu,\sigma) = \frac{1}{\sigma}(1 + \xi z)^{-1/\xi-1}\exp{-(1+\xi z)^{-1/\xi}}'
    where z=\frac{x-\mu}{\sigma}

    \sigma > 0,
    x > \mu-\sigma/\xi\, if \xi > 0,
    x < \mu-\sigma/\xi\,\;(\xi < 0)
    x \in [-\infty,\infty]\,\;(\xi = 0)
    """
    return flib.gev(x,xi,loc, scale)

# Geometric----------------------------------------------
@randomwrap
def rgeometric(p):
    return random.negative_binomial(1, p)

def geometric_expval(p):
    return (1. - p) / p


def geometric_like(x, p):
    """Geometric log-likelihood

    geometric_like(x, p)

    x > 0, p \in [0,1]
    """
    constrain(p, 0, 1)
    constrain(x, lower=0)
    return flib.negbin2(x, 1, p)

# Half-normal----------------------------------------------
@randomwrap
def rhalf_normal(tau):
    return random.normal(0, sqrt(1/tau))

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
def rinverse_gamma(alpha, beta):
    pass

def inverse_gamma_expval(alpha, beta):
    return array(alpha) / beta


def inverse_gamma_like(x, alpha, beta):
    """Inverse gamma log-likelihood

    inverse_gamma_like(x, alpha, beta)

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
def rpoisson(mu):
    return random.poisson(mu)

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
#@randomwrap
def rweibull(alpha, beta):
    alpha = np.atleast_1d(alpha)
    tmp = -log(runiform(0, 1, len(alpha)))
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







