# TODO: Wrap the Fortran and Python covariance functions in a single
# TODO: package, and decorate them to automatically pass in the 'symm'
# TODO: 'amp' and 'scale' arguments.

from pylab import *
from numpy.random import normal
from cov_funs import *


d = 2

x=arange(-1.,1.,.01)
y=x
N=len(x)

N = 10
x=normal(size=(N,d)).squeeze()
y=normal(size=(N,d)).squeeze()
x.sort(0)
y.sort(0)

C=zeros((N,N), dtype=float)

# nu = .9
# alpha = 1.
# phi=1.
# Matern(C,x,y,nu,alpha,phi)

# nu = .9
# scale = .5
# amp = 2.
# NormalizedMatern(C,x,y,nu,scale,amp)

# scale = .5
# amp = 2.
# axi_gauss(C,x,y,scale,amp)

# scale = .5
# amp = 2.
# pow=1.5
# axi_exp(C,x,y,scale,amp,pow)


contourf(C)
print diag(C)
colorbar()
show()