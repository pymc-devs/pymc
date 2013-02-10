'''

@author: johnsalvatier
'''
from dist_math import * 

def Uniform(lb, ub):
    def dist(value):
        return switch((value >= lb) & (value <= ub),
                  -log(ub-lb),
                  -inf)
    return dist

def Flat():
    def dist(value):
        return zeros_like(value)
    return dist

def Normal(mu = 0.0, tau = 1.0):
    def dist(value):
        return switch(gt(tau , 0),
			 -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)
    return dist

def Beta(alpha, beta):
    def dist(value):
        return switch(ge(value , 0) & le(value , 1) &
                  gt(alpha , 0) & gt(beta , 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)
    return dist

def Binomial( n, p):
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

def Bound(dist, lower = -inf, upper = inf):
    def ndist(value):
        return switch(ge(value , lower) & le(value , upper),
                  dist(value),
                  -inf)
    return ndist

def TruncT(mu, lam, nu):
    return Bound(T(mu,lam,nu), 0)


from theano.sandbox.linalg import det
from theano.tensor import dot

def MvNormal(mu, tau):
    def dist(value): 
        delta = value - mu
        return 1/2. * ( log(det(tau)) - dot(delta.T,dot(tau, delta)))
    return dist
