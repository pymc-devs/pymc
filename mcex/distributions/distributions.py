'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import * 
from numpy import pi, inf

def Normal(value, mu = 0.0, tau = 1.0):
    return switch(tau > 0, -0.5 * tau * (value-mu)**2 + 0.5*log(0.5*tau/pi), -inf)

