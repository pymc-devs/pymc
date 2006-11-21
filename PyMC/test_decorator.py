# Decorate fortran functions from flib to ease argument passing
import PyMC.flib as flib
from PyMC import Sampler, LikelihoodError
import numpy as np
from numpy import inf
import string
from flib import categor as _fcategorical
from flib import beta as _fbeta
from flib import bernoulli as _fbernoulli
from flib import binomial as _fbinomial
from flib import cauchy as _fcauchy
from flib import dirichlet as _fdirichlet
from flib import dirmultinom as _fdirmultinom
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
from flib import weibull as _fweibull
from flib import wishart as _fwishart

def fwrap(f, prior=False):
    """Decorator function.
    Assume the arguments are x, par1, par2, ...
    Shape each parameter according to the shape of x.
    Pass x and parameters as one dimensional arrays to the fortran function.
    """
    
    def wrapper(*args, **kwargs):
        """wrapper doc"""
        xshape = np.shape(args[0])
        newargs = [np.asarray(args[0]).flat]
        for arg in args[1:]:
            newargs.append(np.resize(arg, xshape).flat)
        return f(*newargs, **kwargs)
    wrapper.__doc__ = f.__doc__
    wrapper._prior = prior
    wrapper._PyMC = True
    wrapper.__name__ = f.__name__
    return wrapper


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

def categorical_like( x, probs, minval=0, step=1):
    """Categorical log-likelihood. 
    Accepts an array of probabilities associated with the histogram, 
    the minimum value of the histogram (defaults to zero), 
    and a step size (defaults to 1).
    """
    # Normalize, if not already
    if sum(probs) != 1.0: probs = probs/sum(probs)
    return _fcategorical(x, probs, minval, step)

@fwrap
def beta_like(x, alpha, beta):
    """Beta log-likelihood
    beta_like(x, alpha, beta)
    """
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    constrain(x, 0, 1)
    return _fbeta(x, alpha, beta)

@fwrap    
def bernoulli_like(x, p):
    """Bernoulli log-likelihood
    bernoulli_like(x, p)
    """
    constrain(p, 0, 1)
    constrain(x, 0, 1)
    return _fbernouilli(x, p)
        
        
def _bernoulli_gof(x,p,loss):
    """Goodness of fit for bernouilli."""
    expval = p
                
    # Simulated values
    y = array([rbinomial(1, _p) for _p in p])
    
    # Generate GOF points
    gof_points = sum(transpose([loss(x, expval), loss(y, expval)]), 0)
    
    return gof_points
    
    
@fwrap
def binomial_like(x, n, p):
    """Binomial log-likelihood
    binomial_like(x, n, p)
    """
    constrain(p, 0, 1)
    constrain(n, lower=x)
    constrain(x, 0)
    return _fbinomial(x,n,p)

@fwrap
def cauchy_like(x, alpha, beta):
    """Cauchy log-likelhood
    cauchy_like(x, alpha, beta)
    """
    constrain(beta, lower=0)
    return _fcauchy(x,alpha,beta)

@fwrap
def chi2_like(x, df):
    """Chi-squared log-likelihood
    chi2_like(x, df)
    """
    constrain(x, lower=0)
    constrain(df, lower=0)
    return _fgamma(y, 0.5*df, 2)

@fwrap
def dirichlet_like(x, theta):
    """Dirichlet log-likelihood
    dirichlet_like(x, theta)
    """
    constrain(theta, lower=0)
    constrain(x, lower=0)
    constrain(sum(x), upper=1)
    return _fdirichlet(x,theta)

@fwrap
def exponential_like(x, beta):
    """Exponential log-likelihood
    exponential_like(x, beta)
    """
    constrain(x, lower=0)
    constrain(beta, lower=0)
    return _fgamma(x, 1, beta)

@fwrap   
def gamma_like(x, alpha, beta):
    """Gamma log-likelihood
    gamma_like(x, alpha, beta)
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)        
    return _fgamma(x, alpha, beta)
    
@fwrap
def geometric_like(x, p):
    """Geometric log-likelihood
    geometric_like(x, p)
    """
    constrain(p, 0, 1)
    constrain(x, lower=0)
    return _fnegbin(x, 1, p)

@fwrap
def half_normal_like(x, tau):
    """Half-normal log-likelihood
    half_normal_like(x, tau)
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return _fhalfnormal(x, tau)
    
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

@fwrap
def inverse_gamma_like(x, alpha, beta):
    """Inverse gamma log-likelihood
    inverse_gamma_like(x, alpha, beta)
    """
    constrain(x, lower=0)
    constrain(alpha, lower=0)
    constrain(beta, lower=0)
    return _figamma(x, alpha, beta)
    
@fwrap
def lognormal_like(x, mu, tau):
    """Log-normal log-likelihood
    lognormal_like(x, mu, tau)
    """
    constrain(tau, lower=0)
    constrain(x, lower=0)
    return _flognormal(x,mu,tau)



@fwrap        
def poisson_like(x,mu):
    """Poisson log-likelihood
    poisson_like(x,mu)"""
    constrain(x, lower=0)
    constrain(mu, lower=0)
    return _fpoisson(x,mu)
    



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
        print name
        setattr(Sampler, name+'_like2', likeandgofwrap(like, gof))
    except KeyError:
        pass
    
        
# pour gof , on peut probablement wrapper de maniere dynamique les fonctions, et leur donner un nom dynamique itou. 
    
