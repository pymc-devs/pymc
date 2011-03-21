'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import switch, log
from numpy import pi, inf

def Normal(value, mu = 0.0, tau = 1.0):
    return switch(tau > 0, -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)

def Bernoulli(value, p):
    return switch((p >= 0) & (p <= 1), 
                  switch(value, log(p), log(1-p)),
                  -inf)
