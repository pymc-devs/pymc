from pymc import *
from pymc.gp import *
from pymc.gp.cov_funs import matern
from numpy import *
from pylab import *

from numpy.random import normal



x = arange(-1.,1.,.1)

# Prior parameters of C
diff_degree = Uniform('diff_degree', .1, 3)
amp = Lognormal('amp', mu=.4, tau=1.)
scale = Lognormal('scale', mu=.5, tau=1.)

# The covariance dtrm C is valued as a Covariance object.                    
@deterministic
def C(eval_fun = matern.euclidean, diff_degree=diff_degree, amp=amp, scale=scale):
    return NearlyFullRankCovariance(eval_fun, diff_degree=diff_degree, amp=amp, scale=scale)


# Prior parameters of M
a = Normal('a', mu=1., tau=1.)
b = Normal('b', mu=.5, tau=1.)
c = Normal('c', mu=2., tau=1.)

# The mean M is valued as a Mean object.
def linfun(x, a, b, c):
    # return a * x ** 2 + b * x + c    
    return 0.*x + c
@deterministic
def M(eval_fun = linfun, a=a, b=b, c=c):
    return Mean(eval_fun, a=a, b=b, c=c)


# The GP itself
fmesh = linspace(-pi/3.3,pi/3.3,4)
f = GP(name="f", M=M, C=C, mesh=fmesh, init_mesh_vals = 0.*fmesh)


# Observation precision
# V = Gamma('V', alpha=3., beta=3./.002, value=.002)
V = .0001

# The data d is just array-valued. It's normally distributed about GP.f(obs_x).
@data
@stochastic
def d(value=normal(size=len(fmesh)), mu=f, V=V):
    """
    Data
    """
    mu_eval = mu(fmesh)
    return flib.normal(value, mu_eval, 1./V)