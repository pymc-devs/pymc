# GaussianProcess/examples/realizations.py

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

def linfun(x, m, b):
    return m * x + b

M = Mean(eval_fun = linfun, C = C, m = 1., b = 0.)

f=[]
for i in range(3):
    f.append(Realization(M,C))

plot_envelope(M,C)
for i in range(3):
    plot(arange(-1.,1.,.1), f[i])
