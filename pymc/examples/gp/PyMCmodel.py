import pymc as pm
import pymc.gp as gp
from pymc.gp.cov_funs import matern
import numpy as np
import matplotlib.pyplot as pl

from numpy.random import normal



x = np.arange(-1.,1.,.1)

# Prior parameters of C
diff_degree = pm.Uniform('diff_degree', .1, 3)
amp = pm.Lognormal('amp', mu=.4, tau=1.)
scale = pm.Lognormal('scale', mu=.5, tau=1.)

# The covariance dtrm C is valued as a Covariance object.
@pm.deterministic
def C(eval_fun = gp.matern.euclidean, diff_degree=diff_degree, amp=amp, scale=scale):
    return gp.FullRankCovariance(eval_fun, diff_degree=diff_degree, amp=amp, scale=scale)


# Prior parameters of M
a = pm.Normal('a', mu=1., tau=1.)
b = pm.Normal('b', mu=.5, tau=1.)
c = pm.Normal('c', mu=2., tau=1.)

# The mean M is valued as a Mean object.
def linfun(x, a, b, c):
    # return a * x ** 2 + b * x + c
    return 0.*x + c
@pm.deterministic
def M(eval_fun = linfun, a=a, b=b, c=c):
    return gp.Mean(eval_fun, a=a, b=b, c=c)


# The GP itself
fmesh = np.linspace(-np.pi/3.3,np.pi/3.3,4)
f = gp.GP(name="f", M=M, C=C, mesh=fmesh, init_mesh_vals = 0.*fmesh)


# Observation precision
# V = Gamma('V', alpha=3., beta=3./.002, value=.002)
V = .0001

# The data d is just array-valued. It's normally distributed about GP.f(obs_x).
@pm.observed
@pm.stochastic
def d(value=np.random.normal(size=len(fmesh)), mu=f, V=V):
    """
    Data
    """
    mu_eval = mu(fmesh)
    return pm.flib.normal(value, mu_eval, 1./V)
