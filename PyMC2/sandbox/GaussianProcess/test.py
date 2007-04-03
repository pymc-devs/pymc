# TODO: You need to condition the mean together with the covariance, there's no way around it.
# TODO: You can make the 'M' argument to condition optional though, so people can just condition
# TODO: a covariance if they want to.

import Realization
reload(Realization)
from Realization import *

import GPutils
reload(GPutils)
from GPutils import observe, plot_envelope

from cov_funs import *
from numpy import *
from pylab import *

x=arange(-1.,1.,.02)
# obs_mesh = array([-1.,.5])
obs_mesh = array([-.8, 0., .8])
obs_taus =  1000000.*ones(len(obs_mesh), dtype=float)

def linfun(x, m, b):
    return m * x + b

C = Covariance(   eval_fun = NormalizedMatern,
                    base_mesh = x,
                    nu = .49,
                    amp = 1.,
                    scale = .3)

M = Mean( eval_fun = linfun,
            C = C,
            m = 1.,
            b = 0.)
            
# observe(    C=C,
#             obs_mesh = obs_mesh,
#             M=M,
#             obs_taus = obs_taus,
#             # lintrans = array([3.]),
#             obs_vals = zeros(len(obs_mesh)))

f=[]
for i in range(3):
    f.append(Realization(M,C))

# print C ** 2

plot_envelope(M,C)
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
