# TODO: You need to condition the mean together with the covariance, there's no way around it.
# TODO: You can make the 'M' argument to condition optional though, so people can just condition
# TODO: a covariance if they want to.

import GPRealization
reload(GPRealization)
from GPRealization import *

from cov_funs import *
from numpy import *
from pylab import *

x=arange(-1.,1.,.02)
# obs_mesh = array([-1.,.5])
obs_mesh = array([-.8, 0., .8])
obs_taus =  100.*ones(len(obs_mesh), dtype=float)

def linfun(x, m, b):
    return m * x + b

C = GPCovariance(   eval_fun = NormalizedMatern,
                    base_mesh = x,
                    nu = 1.,
                    amp = 1.,
                    scale = .3)

M = GPMean( eval_fun = linfun,
            C = C,
            m = 1.,
            b = 0.)
            
condition(  C=C,
            obs_mesh = obs_mesh,
            M=M,
            obs_taus = obs_taus,
            # lintrans = array([3.]),
            obs_vals = zeros(len(obs_mesh)))

f=[]
for i in range(3):
    f.append(GPRealization(M,C))

sig = sqrt(diag(C))
clf()
plot(x,M,'k-')
plot(x,M-sig,'k-.')
plot(x,M+sig,'k-.')
for i in range(3):
    plot(x,f[i])
    
show()

# print C(array([.5,.6,.7,.9]), array([.2,.3,.9]))
# print C(array([.5,.6,.7,.9])))
# print M(array([.5,.6,.7,.9]))


# clf()            
# M.plot()
# C.plot()
# plot(C.base_mesh, diag(C))
# show()
