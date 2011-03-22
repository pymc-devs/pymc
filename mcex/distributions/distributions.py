'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import log, switch
from numpy import pi, inf
from special import gammaln

def Normal(value, mu = 0.0, tau = 1.0):
    return switch(tau > 0, -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)

def Beta(value, alpha, beta):
    return switch((value >= 0) & (value <= 1) &
                  (alpha > 0) & (beta > 0),
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(value) + (beta-1)*log(1-value),
                  -inf)