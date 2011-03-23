'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import switch, log
from numpy import pi, inf
from special import gammaln, factln

def Normal(value, mu = 0.0, tau = 1.0):
    return switch(tau > 0, -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)

def Beta(value, alpha, beta):
    return switch((value >= 0) & (value <= 1) &
                  (alpha > 0) & (beta > 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)
def Binomial(value, n, p):
    return switch ((value >= 0) & (n >= value) & (p >= 0) & (p <= 1),
                   switch(value != 0, value*log(p), 0) + (n-value)*log(1-p) + factln(n)-factln(value)-factln(n-value),
                   -inf)
    
def BetaBin(value, alpha, beta, n):
    return switch ((value >= 0) & (alpha > 0) & (beta > 0) & (n >= value), 
                   gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(value+1)- gammaln(n-value +1) + gammaln(alpha+value)+ gammaln(n+beta-value)- gammaln(beta+alpha+n),
                   -inf)

def Bernoulli(value, p):
    return switch((p >= 0) & (p <= 1), 
                  switch(value, log(p), log(1-p)),
                  -inf)

def T(value, mu, lam, nu):
    return switch((lam > 0) & (nu > 0),
                  gammaln((nu+1.0)/2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu/2.0) - (nu+1.0)/2.0 * log(1.0 + lam *(value - mu)**2/nu),
                  -inf)
    
def Cauchy(value, alpha, beta):
    return switch(beta > 0,
                  -log(beta) - log( 1 + ((value-alpha) / beta) ** 2 ),
                  -inf)
    
def Gamma(value, alpha, beta):
    return switch((value >= 0) & (alpha > 0) & (beta > 0),
                  -gammaln(alpha) + alpha*log(beta) - beta*value + switch(alpha != 1.0, (alpha - 1.0)*log(value), 0),
                  -inf)