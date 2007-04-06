# GaussianProcess/examples/mean.py

from GaussianProcess import *
from numpy import *

def exp_cov(x,y,p,a,s):
    """
    a and s must be positive
    p must be positive and less than 2
    """

    C = zeros((len(x), len(y)))
    for i in xrange(len(x)):
        for j in xrange(len(y)):
            C[i,j] = a * exp(-(abs(x-y) / s) ** p)
    return C

C = Covariance(eval_fun = exp_cov, p = .49, a = 1., s = .3)

def linfun(x, a, b, c):
    return a * x ** 2 + b * x + c

M = Mean(eval_fun = linfun, C = C, a = 1., b = .5, c = 2.)
x=arange(-1.,1.,.1)
plot(x, M(x), 'k-')
