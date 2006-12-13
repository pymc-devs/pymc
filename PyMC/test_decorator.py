# Decorate fortran functions from PyMC.flib to ease argument passing
# TODO: Deal with functions that take correlation matrices as arguments.wishart, normal,?
# TODO: Deal with functions that have no fortran counterpart. uniform_like, categorical
# TODO: Make node_to_NDarray decorator better.
# TODO: Replace flat's with ravel's, and if possible avoid resize-ing (try to
# avoid any memory allocation, in fact).
# For GOF tests, the wrapper fwrap could check the value of a global variable
# _GOF, and call the gof function instead of the likelihood. 
# TODO: make sure gamfun is vectorized in flib

import flib
from PyMC import Sampler, LikelihoodError
import numpy as np
import proposition4
from numpy import inf, random, sqrt
import string
# Import the raw fortran likelihoods
from flib import categor as _fcategorical
from flib import beta as _fbeta
from flib import bernoulli as _fbernoulli
from flib import binomial as _fbinomial
from flib import cauchy as _fcauchy
from flib import dirichlet as _fdirichlet
from flib import dirmultinom as _fdirmultinom
from flib import exponweib as _exponweib
from flib import gamma as _fgamma
from flib import hnormal as _fhalfnormal
from flib import hyperg as _fhyperg
from flib import igamma as _figamma
from flib import lognormal as _flognormal
from flib import multinomial as _fmultinomial
from flib import mvhyperg as _fmvhyperg
from flib import negbin2 as _fnegbin
from flib import normal as _fnormal
from flib import mvnorm as _fmvnorm
from flib import poisson as _fpoisson
from flib import uniform_like as _funiform
from flib import weibull as _fweibull
from flib import wishart as _fwishart
from flib import wshrt

# Import utility functions
from flib import constrain as _fconstrain
from flib import standardize as _fstandardize
inverse = np.linalg.pinv
import __main__


""" Loss functions """

absolute_loss = lambda o,e: absolute(o - e)

squared_loss = lambda o,e: (o - e)**2

chi_square_loss = lambda o,e: (1.*(o - e)**2)/e

    
def node_to_NDarray(arg):
	if isinstance(arg,proposition4.Node):
		return arg.value
	else:
		return arg
		
def GOFpoints(x,y,expval,loss):
    return sum(np.transpose([loss(x, expval), loss(y, expval)]), 0)


def fwrap(f):
    """
    Decorator function for likelihoods
    ==================================
    
    Wrap function f(*args, **kwds) where f is a likelihood defined in flib.
    
    Assume args = (x, param1, param2, ...)
    Before passing the arguments to the function, the wrapper makes sure that 
    the parameters have the same shape as x.


    Add compatibility with GoF (Goodness of Fit) tests 
    --------------------------------------------------
    * Add a 'prior' keyword (True/False)
    * If the keyword 'gof' is given and is True, return the GoF (Goodness of Fit)
    points instead of the likelihood. 
    * A 'loss' keyword can be given, to specify the loss function used in the 
    computation of the GoF points. 
    """
    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.
    snapshot = __main__.__dict__
    
    # Find the functions needed to compute the gof points.
    expval_func = snapshot['_'+name+'_expval']
    random_func = snapshot['r'+name]
    
    def wrapper(*args, **kwds):
        """This wraps a likelihood."""
        
        # Shape manipulations
        xshape = np.shape(node_to_NDarray(args[0]))
        newargs = [np.asarray(node_to_NDarray(args[0]))]
        for arg in args[1:]:
            newargs.append(np.resize(node_to_NDarray(arg), xshape))
        for key in kwds.iterkeys():
            kwds[key] = node_to_NDarray(kwds[key])  
        
        if kwds.pop('gof', False) and not kwds.pop('prior', False):
            """Return gof points."""            
            loss = kwds.pop('gof', squared_loss)
            #name = kwds.pop('name', name)
            expval = expval_func(*newargs[1:], **kwds)
            y = random_func(*newargs[1:], **kwds)
            gof_points = GOFpoints(newargs[0],y,expval,loss)
            return gof_points
        else:
            """Return likelihood."""
            try:
                return f(*newargs, **kwds)
            except LikelihoodError:
                return -np.Inf
        

    # Assign function attributes to wrapper.
    wrapper.__doc__ = f.__doc__
    wrapper._PyMC = True
    wrapper.__name__ = f.__name__
    wrapper.name = name
    return wrapper


def randomwrap(f):
    """
    Wrapper for random value generators
    ===================================
    
    Vectorize random value generation functions so an array of parameters may 
    be passed.
    """
    return np.vectorize(f)


def priorwrap(f):
    """
    Wrapper to create prior functions
    =================================
    
    Given a likelihood function, return a prior function. 
    
    The only thing that changes is that the _prior attribute is set to True.
    """
    def wrapper(*args, **kwds):
        kwds['prior'] = True
        return f(args, kwds)
    wrapper.__doc__ = string.capwords(f.__name__) + ' prior'
    return wrapper
    
def likeandgofwrap(f):
    """
    Decorator function building likelihood method for Sampler
    =========================================================
    
    Wrap function f(*args, **kwds) where f is a likelihood defined in flib.
    
    Assume args = (self, x, param1, param2, ...)
    Before passing the arguments to the function, the wrapper makes sure that 
    the parameters have the same shape as x.


    Add compatibility with Sampler class
    --------------------------------------------------
    * Add a name keyword.
    * Add a 'prior' keyword (True/False).
    * If self._gof is True and prior is False, return the GoF (Goodness of Fit),
    put the gof points in self._gof_loss
    * A 'loss' keyword can be given, to specify the loss function used in the 
    computation of the GoF points. 
    """
    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.
    snapshot = __main__.__dict__
    
    # Find the functions needed to compute the gof points.
    expval_func = snapshot['_'+name+'_expval']
    random_func = snapshot['r'+name]
    
    def wrapper(*args, **kwds):
        self = args.pop(0)
        
        # Shape manipulations
        xshape = np.shape(args[0])
        newargs = [np.asarray(args[0])]
        for arg in args[1:]:
            newargs.append(np.resize(arg, xshape))
        
        if self._gof and not kwds.pop('prior', False):
            """Compute gof points."""               
            name = kwds.pop('name', name)
            try:    
                self._like_names.append(name)
            except AttributeError:
                pass
                         
            expval = expval_func(*newargs[1:], **kwds)
            y = random_func(*newargs[1:], **kwds)
            gof_points = GOFpoints(newargs[0],y,expval,self.loss)
            self._gof_loss.append(gof_points)
        
        else:
            """Return likelihood."""
            try:
                return f(*newargs, **kwds)
            except LikelihoodError:
                return -np.Inf
        

    # Assign function attributes to wrapper.
##    wrapper.__doc__ = f.__doc__+'\n'+like.__name__+\
##        '(self, '+string.join(f.func_code.co_varnames, ', ')+\
##        ', name='+name +')'
    wrapper.__name__ = f.__name__
    wrapper.name = name
    return wrapper

        
def constrain(value, lower=-inf, upper=inf, allow_equal=False):
    """Apply interval constraint on parameter value."""
    ok = _fconstrain(value, lower, upper, allow_equal)
    if ok == 0:
        raise LikelihoodError
        
# Bernoulli----------------------------------------------
def rbernoulli(p):
    return random.binomial(1,p)
        
def _bernoulli_expval(p):
    """Goodness of fit for bernoulli."""
    return p
                
@fwrap    
def bernoulli_like(x, p):
    """Bernoulli log-likelihood
    
    bernoulli_like(x, p)

    p \in [0,1], x \in [0,1]
    """
    constrain(p, 0, 1)
    constrain(x, 0, 1)
    return _fbernoulli(x, p)
        

# Beta----------------------------------------------
@randomwrap
def rbeta(alpha, beta):
    return random.beta(alpha, beta)

def _beta_expval(x,alpha, beta):
    """Goodness of fit for beta."""
    expval = 1.0 * alpha / (alpha + beta)
    return expval

@fwrap        
def beta_like(x, alpha, beta):
    """Beta log-likelihood
    
    beta_like(x, alpha, beta)
    
    x in [0,1], alpha >= 0, beta >= 0 
    """
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    constrain(x, 0, 1)
    return _fbeta(x, alpha, beta)

# Binomial----------------------------------------------
def rbinomial(n,p):
    return random.binomial(n,p)

def _binomial_expval(x,n,p):
    expval = p * n
    return expval

@fwrap
def binomial_like(x, n, p):
    """Binomial log-likelihood
    
    binomial_like(x, n, p)
    
    p \in [0,1], n > x, x > 0 
    """
    constrain(p, 0, 1)
    constrain(n, lower=x)
    constrain(x, 0)
    return _fbinomial(x,n,p)

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
@randomwrap
def rcauchy(alpha, beta, n=None):
    """Returns Cauchy random variates"""
    N = n or max(size(alpha), size(beta))
    return alpha + beta*tan(pi*random_number(n) - pi/2.0)

def _cauchy_expval(alpha, beta):
    return alpha

@fwrap
def cauchy_like(x, alpha, beta):
    """Cauchy log-likelhood
    
    cauchy_like(x, alpha, beta)

    beta > 0
    """
    constrain(beta, lower=0)
    return _fcauchy(x,alpha,beta)

# Chi square----------------------------------------------
@randomwrap
def rchi2(df):
    return random.chisquare(df)
    
def _chi2_expval(df):
    return df

@fwrap
def chi2_like(x, df):
    """Chi-squared log-likelihood

    chi2_like(x, df)
    
    x > 0, df > 0
    """
    constrain(x, lower=0)
    constrain(df, lower=0)
    return _fgamma(y, 0.5*df, 2)

# Dirichlet----------------------------------------------
def rdirichlet(alphas, n=None):
    """Returns Dirichlet random variates"""
    
    if n:
        gammas = transpose([rgamma(alpha,1,n) for alpha in alphas])
        
        return array([g/sum(g) for g in gammas])
    else:
        gammas = array([rgamma(alpha,1) for alpha in alphas])
        
        return gammas/sum(gammas)

def _dirichlet_expval(theta):
    sumt = sum(theta)
    expval = theta/sumt
    return expval

@fwrap
def dirichlet_like(x, theta):
    """Dirichlet log-likelihood
    
    dirichlet_like(x, theta)
    
    theta > 0, x > 0, \sum x < 1
    """
    constrain(theta, lower=0)
    constrain(x, lower=0)
    constrain(sum(x), upper=1)
    return _fdirichlet(x,theta)

# Exponential----------------------------------------------
@randomwrap
def rexponential(beta):
    return random.exponential(beta)

def _exponential_expval(beta):
    return beta

@fwrap
def exponential_like(beta):
    """Exponential log-likelihood
    
    exponential_like(x, beta)

    x > 0, beta > 0
    """
    constrain(x, lower=0)
    constrain(beta, lower=0)
    return _fgamma(x, 1, beta)

# Gamma----------------------------------------------
@randomwrap
def rgamma(alpha, beta):
    return random.gamma(1./beta, alpha)

def _gamma_expval(alpha, beta):
    expval = array(alpha) / beta
    return expval

@fwrap   
def gamma_like(x, alpha, beta):
    """Gamma log-likelihood
    
    gamma_like(x, alpha, beta)

    x > 0, alpha > 0, beta > 0
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)        
    return _fgamma(x, alpha, beta)
        

# Geometric----------------------------------------------
@randomwrap
def rgeometric(p):
    return random.negative_binomial(1, p)

def _geometric_expval(p):
    return (1. - p) / p

@fwrap
def geometric_like(x, p):
    """Geometric log-likelihood

    geometric_like(x, p)

    x > 0, p \in [0,1]
    """
    constrain(p, 0, 1)
    constrain(x, lower=0)
    return _fnegbin(x, 1, p)

# Half-normal----------------------------------------------
def rhalf_normal(tau):
    return random.normal(0, sqrt(1/tau))    
    
def _half_normal_expval(tau):
    return sqrt(0.5 * pi / array(tau))

@fwrap
def half_normal_like(x, tau):
    """Half-normal log-likelihood
    
    half_normal_like(x, tau)

    x > 0, tau > 0
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return _fhalfnormal(x, tau)
    
# Hypergeometric----------------------------------------------
def rhypergeometric(draws, red, total, n=None):
    """Returns n hypergeometric random variates of size 'draws'"""
    
    urn = [1]*red + [0]*(total-red)
    
    if n:
        return [sum(urn[i] for i in permutation(total)[:draws]) for j in range(n)]
    else:
        return sum(urn[i] for i in permutation(total)[:draws])

def _hypergeometric_expval(n,m,N):
    return n * (m / N)

@fwrap
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
    return _fhyperg(x, n, m, N)

# Inverse gamma----------------------------------------------
# Looks this one is identical to rgamma, this is strange.    
def rinverse_gamma(alpha, beta):
    pass 

def _inverse_gamma_expval(alpha, beta):
    return array(alpha) / beta

@fwrap
def inverse_gamma_like(x, alpha, beta):
    """Inverse gamma log-likelihood
    
    inverse_gamma_like(x, alpha, beta)

    x > 0, alpha > 0, beta > 0
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return _figamma(x, alpha, beta)
    
# Lognormal----------------------------------------------
@randomwrap
def rlognormal(mu, tau):
    return random.normal(mu, sqrt(1./tau))

def _lognormal_expval(mu, tau):
    return mu

@fwrap
def lognormal_like(x, mu, tau):
    """Log-normal log-likelihood
    
    lognormal_like(x, mu, tau)

    x > 0, tau > 0 
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return _flognormal(x,mu,tau)

# Multinomial----------------------------------------------
@randomwrap
def rmultinomial(n,p):
    return random.multinomial

def _multinomial_expval(n,p):
    array([pr * n for pr in p])

@fwrap
def multinomial_like(x, n, p):
    """Multinomial log-likelihood with k-1 bins
    
    multinomial_like(x, n, p)
    
    x > 0, p > 0, \sum p < 1, \sum x < n
    """
    constrain(p, lower=0)
    constrain(x, lower=0)
    constrain(sum(p), upper=1)
    constrain(sum(x), upper=n)
    return _fmultinomial(x, n, p)
    
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

def _multivariate_hypergeometric_expval(m):
    return n * (array(m) / sum(m))

@fwrap
def multivariate_hypergeometric_like(x, m):
    """Multivariate hypergeometric log-likelihood
    
    multivariate_hypergeometric_like(x, m)
    
    x < m
    """
    constrain(x, upper=m)
    return _fmvhyperg(x, m)

# Multivariate normal--------------------------------------
# Wrapper won't work if tau is a correlation matrix.
def rmultivariate_normal(mu, tau):
    return random.multivariate_normal(mu, inverse(tau))

def _multivariate_normal_expval(mu, tau):
    return mu

@fwrap
def multivariate_normal_like(x, mu, tau):
    """Multivariate normal log-likelihood
    
    multivariate_normal_like(x, mu, tau)
    
    \trace(tau) > 0
    """
    constrain(diagonal(tau), lower=0)
    return _fmvnorm(x, mu, tau)

# Negative binomial----------------------------------------
@randomwrap
def rnegative_binomial(mu, alpha):
    return random.negative_binomial(alpha, alpha / (mu + alpha))

def _negative_binomial_expval(mu, alpha):
    return mu

@fwrap
def negative_binomial_like(x, mu, alpha):
    """Negative binomial log-likelihood
    
    negative_binomial_like(x, mu, alpha)
    
    x > 0, mu > 0, alpha > 0
    """
    constrain(mu, lower=0)
    constrain(alpha, lower=0)
    constrain(x, lower=0)
    return _fnegbin(x, mu, alpha)

# Normal---------------------------------------------------
@randomwrap
def rnormal(mu, tau):
    return random.normal(mu, 1./sqrt(tau))

def _normal_expval(mu, tau):
    return mu

@fwrap
def normal_like(x, mu, tau):
    """Normal log-likelihood

    normal_like(x, mu, tau)
    
    tau > 0
    """    
    constrain(tau, lower=0)
    return _fnormal(x, mu, tau)
    
    
# Poisson--------------------------------------------------
@randomwrap
def rpoisson(mu):
    return random.poisson(mu)
    
def _poisson_expval(mu):
    return mu

@fwrap
def poisson_like(x,mu):
    """Poisson log-likelihood
    
    poisson_like(x,mu)
    
    x \geq 0, mu \geq 0
    """
    constrain(x, lower=0,allow_equal=True)
    constrain(mu, lower=0,allow_equal=True)
    return _fpoisson(x,mu)
    
# Uniform--------------------------------------------------
@randomwrap
def runiform(lower, upper, size=1):
    return random.uniform(lower, upper, size)

def _uniform_expval(lower, upper):
    return (upper - lower) / 2.

def uniform_like_python(x, lower, upper):
    """Uniform log-likelihood"""
    x = np.atleast_1d(x)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    constrain(x, lower=lower, upper=upper, allow_equal=True)
    return sum(np.log(1. / (np.array(upper) - np.array(lower))))
uniform_like_python._PyMC = True

@fwrap
def uniform_like(x,lower, upper):
    """Uniform log-likelihood
    
    uniform_like(x,lower, upper)
    
    x \in [lower, upper]
    """
    return _funiform(x,lower, upper)

# Weibull--------------------------------------------------
def rweibull(alpha, beta):
    return beta * (-log(runiform(0, 1, len(alpha))) ** (1. / alpha))

def _weibull_expval(alpha,beta):
    return beta * gamfun((a + 1.) / a) 

@fwrap
def weibull_like(x, alpha, beta):
    """Weibull log-likelihood
    
    weibull_like(x, alpha, beta)
    
    x > 0, alpha > 0, beta > 0
    """
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    constrain(x, lower=0)
    return _fweibull(x, alpha, beta)
    
# Wishart---------------------------------------------------
# All these won't work if Tau is a matrix.
def rwishart(n, Tau, m=None):
    """Returns Wishart random matrices"""
    sigma = inverse(Tau)
    D = [i for i in ravel(t(chol(sigma))) if i]
    np = len(sigma)
    
    if m:
        return [expand_triangular(wshrt(D, n, np), np) for i in range(m)]
    else:
        return expand_triangular(wshrt(D, n, np), np)

def _wishart_expval(n, Tau):
    return n * array(Tau)

def wishart_like(X, n, Tau):
    """Wishart log-likelihood
    
    wishart_like(X, n, Tau)
    
    X, T symmetric and positive definite
    n > 0
    """
    constrain(diagonal(Tau), lower=0)
    constrain(n, lower=0)
    return _fwishart(X, n, Tau)

# -----------------------------------------------------------

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
    
# Create priors 
for name,func in likelihoods.iteritems():
    newname = name+'_prior'
    locals()[newname] = priorwrap(func)

    
# Assign likelihoods combined with goodness of fit functions and assign to 
# Sampler. It doesn't work so easily. Methods must be created at class instanti
# ation. 
for name, like in likelihoods.iteritems():
    try:
        setattr(Sampler, name+'_like2', likeandgofwrap(like))
    except KeyError:
        pass
    
       

    
