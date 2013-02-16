'''

@author: johnsalvatier
'''
from dist_math import * 
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
def Uniform(lb, ub):
    def logp(value):
        return switch((value >= lb) & (value <= ub),
                  -log(ub-lb),
                  -inf)
    return locals()

@quickclass
def Flat():
    def logp(value):
        return zeros_like(value)
    return locals()

@quickclass
def Normal(mu = 0.0, tau = 1.0):
    def logp(value):
        return switch(gt(tau , 0),
			 -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)
    return locals()

@quickclass
def Beta(alpha, beta):
    def logp(value):
        return switch(ge(value , 0) & le(value , 1) &
                  gt(alpha , 0) & gt(beta , 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)
    return locals()


@quickclass
def Binomial( n, p):
    def logp(value):
        return switch (ge(value , 0) & ge(n , value) & ge(p , 0) & le(p , 1),
                   switch(ne(value , 0) , value*log(p), 0) + (n-value)*log(1-p) + factln(n)-factln(value)-factln(n-value),
                   -inf)
    return locals()


@quickclass
def BetaBin(alpha, beta, n):
    def logp(value):
        return switch (ge(value , 0) & gt(alpha , 0) & gt(beta , 0) & ge(n , value), 
                   gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(value+1)- gammaln(n-value +1) + gammaln(alpha+value)+ gammaln(n+beta-value)- gammaln(beta+alpha+n),
                   -inf)
    return locals()


@quickclass
def Bernoulli(p):
    def logp(value):
        return switch(ge(p , 0) & le(p , 1), 
                  switch(value, log(p), log(1-p)),
                  -inf)
    return locals()


@quickclass
def T(mu, lam, nu):
    def logp(value):
        return switch(gt(lam  , 0) & gt(nu , 0),
                  gammaln((nu+1.0)/2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu/2.0) - (nu+1.0)/2.0 * log(1.0 + lam *(value - mu)**2/nu),
                  -inf)
    return locals()
    
@quickclass
def Cauchy(alpha, beta):
    def logp(value):
        return switch(gt(beta , 0),
                  -log(beta) - log( 1 + ((value-alpha) / beta) ** 2 ),
                  -inf)
    return locals()
    
@quickclass
def Gamma(alpha, beta):
    def logp(value):
        return switch(ge(value , 0) & gt(alpha , 0) & gt(beta , 0),
                  -gammaln(alpha) + alpha*log(beta) - beta*value + switch(alpha != 1.0, (alpha - 1.0)*log(value), 0),
                  -inf)
    return locals()

@quickclass
def Poisson(lam):
    def logp(value):
        return switch( gt(lam,0),
                       #factorial not implemented yet, so
                       value * log(lam) - gammaln(value + 1) - lam,
                       -inf)
    return locals()

@quickclass
def Constantlogp(c):
    def logp(value):
        
        return switch(eq(value, c), 0, -inf)
    return locals()

@quickclass
def ZeroInflatedPoisson(theta, z):
    def logp(value):
        return switch(z, 
                      Poisson(theta)(value), 
                      Constantlogp(0)(value))
    return locals()

@quickclass
def Bound(dist, lower = -inf, upper = inf):
    def nlogp(value):
        return switch(ge(value , lower) & le(value , upper),
                  dist.logp(value),
                  -inf)
    return ndist

@quickclass
def TruncT(mu, lam, nu):
    return Bound(T(mu,lam,nu), 0)


from theano.sandbox.linalg import det
from theano.tensor import dot

@quickclass
def MvNormal(mu, tau):
    def logp(value): 
        delta = value - mu
        return 1/2. * ( log(det(tau)) - dot(delta.T,dot(tau, delta)))
    return locals()
