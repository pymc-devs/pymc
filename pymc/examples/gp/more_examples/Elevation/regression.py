from makemap import make_map
from pymc.gp import *
from pymc.gp.cov_funs import matern
from getdata import *
from pylab import *


C = Covariance(matern.geo_deg, diff_degree = .8, amp = std(elevation), scale = 1.)

def constant(x, val):
    """docstring for parabolic_fun"""
    return zeros(x.shape[:-1],dtype=float) + val
    
M = Mean(constant, val = 0.)

observe(M,C,obs_mesh = obs_mesh, obs_vals = elevation)

# Plot the mean surface
make_map(M)
title('Mean surface')
# savefig('../../../../Docs/figs/elevmean.pdf')

# Plot the SD
make_map(C)
title('Variance surface')
# savefig('../../../../Docs/figs/elevvar.pdf')

# Plot some realizations
for i in range(2):
    make_map(Realization(M,C))
    title('A random surface drawn from the GP')
    # savefig('../../../../Docs/figs/elevdraw%i.pdf' % i)
