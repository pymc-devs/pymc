from pymc import *
from pymc.gp import *
from pymc.gp.cov_funs import matern
from getdata import *
from pylab import *

# ============
# = The mean =
# ============

# Vague prior for the overall mean
m = Uninformative('m', value=0)

def constant(x, val):
    return zeros(x.shape[:-1],dtype=float) + val

@deterministic
def M(m=m):
    return Mean(constant, val=m)

# ==================
# = The covariance =
# ==================

# Informative prior for the degree of differentiability
diff_degree = Uniform('diff_degree',.5,2,value=1)

# Vague priors for the amplitude and scale
amp = Exponential('amp',7.e-5, value=np.std(v))
scale = Exponential('scale',4e-3, value=x.max()/2)
@deterministic
def C(amp=amp, scale=scale, diff_degree=diff_degree):
    return FullRankCovariance(matern.euclidean, diff_degree = diff_degree, amp = amp, scale = scale)

# ===================
# = The GP submodel =
# ===================
walker_v = GPSubmodel('walker_v', M, C, mesh)

# ============
# = The data =
# ============

# Vague prior for the observation variance
V = Exponential('V',5e-9, value=np.std(v))

# This is the observation of the elevation variable 'v', plus normally-distributed error.
d = Normal('d',walker_v.f_eval,1./V,value=v,observed=True)


