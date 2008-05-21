from pymc.gp import *
from pymc.gp.cov_funs import *
from pymc import *

@data
@stoch
def D_amp(value=.12, alpha=5., mean=.1):
    """D_amp ~ distribution(scale)"""
    return gamma_like(value, alpha, mean/alpha)

D_scale = 100.

@dtrm
def D_C(amp=D_amp, scale=D_scale):
    """D_C = function(amp, scale)"""
    C = BasisCovariance(basis = fourier_basis, scale = scale, coef_cov = diag(amp**2/arange(1,100,dtype=float)**((1.5))))
    return C

@dtrm
def D_M(mean=1.):
    """The mean function is constant at 1"""
    M = Mean(lambda x: 0.*x+log(mean))
    return M
    
M = D_M.value
C = D_C.value

mesh = arange(0., 5000+800, 800)


observe(M=M, C=C, obs_vals = 0.*mesh, obs_mesh = mesh)
# plot(mesh, D.value(mesh)+1.)
plot_envelope(M,C,mesh)

# for i in range(10):
#     f = Realization(M,C)
#     plot(D.mesh, f(mesh))