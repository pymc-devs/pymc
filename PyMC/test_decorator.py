# Decorate fortran functions from PyMC.flib to ease argument passing
# TODO: Deal with functions that take correlation matrices as arguments.wishart, normal,?
# TODO: Deal with functions that have no fortran counterpart. uniform_like, categorical
# TODO: Think about a structure to deal with GOF tests. 
# TODO: Write a wrapper for random generation functions.
#
# TODO: Make node_to_NDarray decorator better.
# TODO: Replace flat's with ravel's, and if possible avoid resize-ing (try to
# avoid any memory allocation, in fact).


from PyMC import flib
from PyMC import Sampler, LikelihoodError
import numpy as np
import proposition4
from numpy import inf, random
import string
from PyMC.flib import categor as _fcategorical
from PyMC.flib import beta as _fbeta
from PyMC.flib import bernoulli as _fbernoulli
from PyMC.flib import binomial as _fbinomial
from PyMC.flib import cauchy as _fcauchy
from PyMC.flib import dirichlet as _fdirichlet
from PyMC.flib import dirmultinom as _fdirmultinom
from PyMC.flib import gamma as _fgamma
from PyMC.flib import hnormal as _fhalfnormal
from PyMC.flib import hyperg as _fhyperg
from PyMC.flib import igamma as _figamma
from PyMC.flib import lognormal as _flognormal
from PyMC.flib import multinomial as _fmultinomial
from PyMC.flib import mvhyperg as _fmvhyperg
from PyMC.flib import negbin2 as _fnegbin
from PyMC.flib import normal as _fnormal
from PyMC.flib import mvnorm as _fmvnorm
from PyMC.flib import poisson as _fpoisson
from PyMC.flib import weibull as _fweibull
from PyMC.flib import wishart as _fwishart
from PyMC.flib import wshrt

""" Loss functions """

absolute_loss = lambda o,e: absolute(o - e)

squared_loss = lambda o,e: (o - e)**2

chi_square_loss = lambda o,e: (1.*(o - e)**2)/e
    
def node_to_NDarray(arg):
	if isinstance(arg,proposition4.Node):
		return arg.value
	else:
		return arg
		

def fwrap(f, prior=False):
    """Decorator function.
    Assume the arguments are x, par1, par2, ...
    Shape each parameter according to the shape of x.
    Pass x and parameters as one dimensional arrays to the fortran function.
    """
    
    def wrapper(*args, **kwargs):
        """wrapper doc"""
        xshape = np.shape(node_to_NDarray(args[0]))
        newargs = [np.asarray(node_to_NDarray(args[0])).flat]
        for arg in args[1:]:
            newargs.append(np.resize(node_to_NDarray(arg), xshape).flat)
        for key in kwargs.iterkeys():
            kwargs[key] = node_to_NDarray(kwargs[key])
        return f(*newargs, **kwargs)
		
    wrapper.__doc__ = f.__doc__
    wrapper._prior = prior
    wrapper._PyMC = True
    wrapper.__name__ = f.__name__
    return wrapper


def randomwrap(f):
    """Some random generator take only scalar parameters, so this wrapper
    allows to pass an array of parameters.
    """
    return np.vectorize(f)


def priorwrap(f):
    def wrapper(*args, **kwargs):
        return f(args, kwargs)
    wrapper._prior=False
    wrapper.__doc__ = string.capwords(f.__name__) + ' prior'
    return wrapper
    
def likeandgofwrap(like, func_gof):
    """Decorator function.
    Defines a method for Sampler that combines the likelihood and the gof test.
    """
    # A lot of stuff is done to maintain compatibility with the current implementation.
    # Some of it could go away, as kwarg prior.
    # Also, it looks like all we need to know to compute the gof function is expval. 
    # The rest only consists in drawing random values and computing the loss. 
    # So we could define exp_distribution, and build the gof function at import. 
    def wrapper(*args, **kwargs):
        self = args[0]
        if self._gof is True:
            try:
                prior = kwargs['prior']
            except NameError:
                prior = False
            
            if prior is False:
                try:
                    name = kwargs['name']
                except NameError:
                    name = like._name
                
                try:    
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                gof_points = func_gof(args[1:])
                self._gof_loss.append(gof_points)
        return like(*args, **kwargs)
    wrapper.__doc__ = like.__doc__+'\n'+like.__name__+\
        '(self, '+string.join(like.func_code.co_varnames, ', ')+\
        ', name='+like.__name__[:-5]+')'
    return wrapper
        

def constrain(value, lower=-inf, upper=inf, allow_equal=False):
    """Apply interval constraint on parameter value."""
    if allow_equal:
        if np.any(lower > value) or np.any(value > upper):
            raise LikelihoodError
    else:
        if np.any(lower >= value) or np.any(value >= upper):
            raise LikelihoodError


# Bernoulli----------------------------------------------
@fwrap    
def bernoulli_like(x, p):
    """Bernoulli log-likelihood
    bernoulli_like(x, p)
    """
    constrain(p, 0, 1)
    constrain(x, 0, 1)
    return _fbernouilli(x, p)
        
def rbernoulli(p):
    return random.binomial(1,p)
        
        
def _bernoulli_gof(x,p):
    """Goodness of fit for bernouilli."""
    expval = p
                
    # Simulated values
    y = rbernoulli(p)
    
    # Generate GOF points
    gof_points = GOFpoints(x,y,expval,loss)
    
    return gof_points
    
def GOFpoints(x,y,expval,loss):
    return sum(transpose([loss(x, expval), loss(y, expval)]), 0)

# Beta----------------------------------------------
@fwrap
def beta_like(x, alpha, beta):
    """Beta log-likelihood
    beta_like(x, alpha, beta)
    """
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    constrain(x, 0, 1)
    return _fbeta(x, alpha, beta)

@randomwrap
def rbeta(alpha, beta):
    return random.beta(alpha, beta)

def _beta_gof(x,alpha, beta):
    """Goodness of fit for beta."""
    expval = 1.0 * alpha / (alpha + beta)
                
    # Simulated values
    y = array([rbeta(a, b) for a, b in zip(alpha, beta)])
    
    # Generate GOF points
    gof_points = sum(transpose([loss(x, expval), loss(y, expval)]), 0)
    
    return gof_points

    
# Binomial----------------------------------------------
@fwrap
def binomial_like(x, n, p):
    """Binomial log-likelihood
    binomial_like(x, n, p)
    """
    constrain(p, 0, 1)
    constrain(n, lower=x)
    constrain(x, 0)
    return _fbinomial(x,n,p)

def rbinomial(n,p):
    return random.binomial(n,p)


# Categorical----------------------------------------------
def categorical_like( x, probs, minval=0, step=1):
    """Categorical log-likelihood. 
    Accepts an array of probabilities associated with the histogram, 
    the minimum value of the histogram (defaults to zero), 
    and a step size (defaults to 1).
    """
    # Normalize, if not already
    if sum(probs) != 1.0: probs = probs/sum(probs)
    return _fcategorical(x, probs, minval, step)


# Cauchy----------------------------------------------
@fwrap
def cauchy_like(x, alpha, beta):
    """Cauchy log-likelhood
    cauchy_like(x, alpha, beta)
    """
    constrain(beta, lower=0)
    return _fcauchy(x,alpha,beta)

#@rwrap #?
def rcauchy(alpha, beta, n=None):
    """Returns Cauchy random variates"""
    N = n or max(size(alpha), size(beta))
    return alpha + beta*tan(pi*random_number(n) - pi/2.0)


# Chi square----------------------------------------------
@fwrap
def chi2_like(x, df):
    """Chi-squared log-likelihood
    chi2_like(x, df)
    """
    constrain(x, lower=0)
    constrain(df, lower=0)
    return _fgamma(y, 0.5*df, 2)

@randomwrap
def rchi2(df):
    return random.chisquare(df)
    
# Dirichlet----------------------------------------------
@fwrap
def dirichlet_like(x, theta):
    """Dirichlet log-likelihood
    dirichlet_like(x, theta)
    """
    constrain(theta, lower=0)
    constrain(x, lower=0)
    constrain(sum(x), upper=1)
    return _fdirichlet(x,theta)

def rdirichlet(alphas, n=None):
    """Returns Dirichlet random variates"""
    
    if n:
        gammas = transpose([rgamma(alpha,1,n) for alpha in alphas])
        
        return array([g/sum(g) for g in gammas])
    else:
        gammas = array([rgamma(alpha,1) for alpha in alphas])
        
        return gammas/sum(gammas)

# Exponential----------------------------------------------
@fwrap
def exponential_like(x, beta):
    """Exponential log-likelihood
    exponential_like(x, beta)
    """
    constrain(x, lower=0)
    constrain(beta, lower=0)
    return _fgamma(x, 1, beta)

@randomwrap
def rexponential(beta):
    return random.exponential(beta)


# Gamma----------------------------------------------
@fwrap   
def gamma_like(x, alpha, beta):
    """Gamma log-likelihood
    gamma_like(x, alpha, beta)
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)        
    return _fgamma(x, alpha, beta)
        
@randomwrap
def rgamma(alpha, beta):
    return random.gamma(1./beta, alpha)


# Geometric----------------------------------------------
@fwrap
def geometric_like(x, p):
    """Geometric log-likelihood
    geometric_like(x, p)
    """
    constrain(p, 0, 1)
    constrain(x, lower=0)
    return _fnegbin(x, 1, p)

@randomwrap
def rgeometric(p):
    return random.negative_binomial(1, p)


# Half-normal----------------------------------------------
@fwrap
def half_normal_like(x, tau):
    """Half-normal log-likelihood
    half_normal_like(x, tau)
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return _fhalfnormal(x, tau)
    
def rhalf_normal(tau):
    return random.normal(0, sqrt(1/tau))    
    
    
# Hypergeometric----------------------------------------------
@fwrap
def hypergeometric_like(x, n, m, N):
    """
    Hypergeometric log-likelihood
    hypergeometric_like(x, n, m, N)
    
    Distribution models the probability of drawing x successful draws in n
    draws from N total balls of which m are successes.
    """
    constrain(m, upper=N)
    constrain(n, upper=N)
    constrain(x, max(0, n - N + m), min(m, n))
    return _fhyperg(x, n, m, N)

def rhyperg(draws, red, total, n=None):
    """Returns n hypergeometric random variates of size 'draws'"""
    
    urn = [1]*red + [0]*(total-red)
    
    if n:
        return [sum(urn[i] for i in permutation(total)[:draws]) for j in range(n)]
    else:
        return sum(urn[i] for i in permutation(total)[:draws])


# Inverse gamma----------------------------------------------
@fwrap
def inverse_gamma_like(x, alpha, beta):
    """Inverse gamma log-likelihood
    inverse_gamma_like(x, alpha, beta)
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return _figamma(x, alpha, beta)
    
# Looks this one is identical to rgamma, this is strange.    
def rinverse_gamma(alpha, beta):
    pass 

# Lognormal----------------------------------------------
@fwrap
def lognormal_like(x, mu, tau):
    """Log-normal log-likelihood
    lognormal_like(x, mu, tau)
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return _flognormal(x,mu,tau)

@randomwrap
def rlognormal(mu, tau):
    return random.normal(mu, sqrt(1./tau))

# Multinomial----------------------------------------------
@fwrap
def multinomial_like(x, n, p):
    """Multinomial log-likelihood with k-1 bins"""
    constrain(p, lower=0)
    constrain(x, lower=0)
    constrain(sum(p), upper=1)
    constrain(sum(x), upper=n)
    return _fmultinomial(x, n, p)


# Multivariate hypergeometric------------------------------
@fwrap
def multivariate_hypergeometric_like(x, m):
    """Multivariate hypergeometric log-likelihood"""
    constrain(x, upper=m)
    return _fmvhyperg(x, m)

# Multivariate normal--------------------------------------
# Won't work if tau is a correlation matrix.
@fwrap
def multivariate_normal_like(x, mu, tau):
    """Multivariate normal log-likelihood"""
    constrain(diagonal(tau), lower=0)
    return _fmvnorm(x, mu, tau)

# Negative binomial----------------------------------------
@fwrap
def negative_binomial_like(x, mu, alpha):
    """Negative binomial log-likelihood"""
    constrain(mu, lower=0)
    constrain(alpha, lower=0)
    constrain(x, lower=0)
    return _fnegbin(x, mu, alpha)

@randomwrap
def rnegative_binomial(mu, alpha):
    return random.negative_binomial(alpha, alpha / (mu + alpha))

# Normal---------------------------------------------------
@fwrap
def normal_like(x, mu, tau):
	"""Normal log-likelihood"""
	constrain(tau, lower=0)
	return _fnormal(x, mu, tau)

@randomwrap
def rnormal(mu, tau):
    return random.normal(mu, 1./sqrt(tau))

# Poisson--------------------------------------------------
@fwrap
def poisson_like(x,mu):
    """Poisson log-likelihood
    poisson_like(x,mu)"""
    constrain(x, lower=0,allow_equal=True)
    constrain(mu, lower=0,allow_equal=True)
    return _fpoisson(x,mu)
    
@randomwrap
def rpoisson(mu):
    return random.poisson(mu)
    
# Uniform--------------------------------------------------
def uniform_like(x, lower, upper):
    """Uniform log-likelihood"""
    x = np.atleast_1d(x)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    constrain(x, lower=lower, upper=upper, allow_equal=True)
    return sum(np.log(1. / (np.array(upper) - np.array(lower))))

uniform_like._PyMC = True
@randomwrap
def runiform(lower, upper, size=1):
    return random.uniform(lower, upper, size)

# Weibull--------------------------------------------------
@fwrap
def weibull_like(x, alpha, beta):
    """Weibull log-likelihood"""
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    constrain(x, lower=0)
    return _fweibull(x, alpha, beta)
    
def rweibull(alpha, beta):
    return beta * (-log(runiform(0, 1, len(alpha))) ** (1. / alpha))

# Wishart---------------------------------------------------
# This won't work if Tau is a matrix.
def wishart_like(X, n, Tau):
    """Wishart log-likelihood"""
    constrain(diagonal(Tau), lower=0)
    constrain(n, lower=0)
    return _fwishart(X, n, Tau)

def rwishart(n, Tau, m=None):
    """Returns Wishart random matrices"""
    sigma = linalg.pinv(Tau)
    D = [i for i in ravel(t(chol(sigma))) if i]
    np = len(sigma)
    
    if m:
        return [expand_triangular(wshrt(D, n, np), np) for i in range(m)]
    else:
        return expand_triangular(wshrt(D, n, np), np)

def expand_triangular(X,k):
    # Expands flattened triangular matrix
    
    # Convert to list
    X = X.tolist()
    
    # Unflatten matrix
    Y = array([[0] * i + X[i * k - (i * (i - 1)) / 2 : i * k + (k - i)] for i in range(k)])
    
    # Loop over rows
    for i in range(k):
        # Loop over columns
        for j in range(k):
            Y[j, i] = Y[i, j]
    
    return Y


all_names = locals().copy()
likelihoods = {}
goodnesses = {}
for name, obj in all_names.iteritems():
    if name[-5:] == '_like' and hasattr(obj, '_PyMC'):
        likelihoods[name[:-5]] = locals()[name]
    if name[-4:] == '_gof':
        goodnesses[name[:-4]] = locals()[name]
    
# Create priors 
for name,func in likelihoods.iteritems():
    newname = name+'_prior'
    locals()[newname] = priorwrap(func)

    
# Assign likelihoods combined with goodness of fit functions and assign to Sampler.
for name, like in likelihoods.iteritems():
    try:
        gof = goodnesses['_'+name]
        setattr(Sampler, name+'_like2', likeandgofwrap(like, gof))
    except KeyError:
        pass
    
       

    
