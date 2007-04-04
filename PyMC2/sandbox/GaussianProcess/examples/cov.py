# GaussianProcess/examples/cov.py

from GaussianProcess import *
from numpy import *

def exp_cov(x,y,pow,amp,scale):
    """
    amp and scale must be positive
    pow must be positive and less than 2
    """
    
    C = zeros((len(x), len(y)))
    for i in xrange(len(x)):
        for j in xrange(len(y)):
            C[i,j] = amp * exp(-(abs(x-y) / scale) ** pow)
    return C

C = Covariance(eval_fun = exp_cov, nu = .49, amp = 1., scale = .3)
C.plot(x=arange(-1.,1.,.1), y=arange(-1.,1.,.1))