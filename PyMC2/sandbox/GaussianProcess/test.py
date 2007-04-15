from GaussianProcess import *
from GaussianProcess.cov_funs import matern
from numpy import *
from numpy.linalg import eigh, cholesky
from scipy.linalg import lu
from GP_fortran_utils import *


# x will be the base mesh
x = arange(-1.,1.,.5)


# Create a Covariance object with x as the base mesh
C = Covariance( eval_fun = matern, 
                base_mesh = x,
                diff_degree = 1.4, amp = .4, scale = .5)

C_mat = C.view(matrix)

def fast_chol(C):
    chol = C.copy()
    info=dpotrf_wrap(chol)
    if info>0:
        chol = C.copy()
        dgetrf_wrap(chol)
    return chol

def fast_LU(C):
    lu = C.copy()
    info, ipiv = dgetrf_wrap(lu)
    return lu
    
def trisolve(chol_factor, b):
    b_copy = b.copy()
    info = dpotrs_wrap(chol_factor, b)
    if info<0:
        raise ValueError

# for i in xrange(10):
#     # val, vec = eigh(C_mat)
#     # L = cholesky(C_mat)
#     # L,D,U = lu(C_mat)
#     # L = fast_chol(C_mat)
#     # L = fast_LU(C_mat)
    
    
# dx = .01, 100 iter:
# eigh: 8.6s
# cholesky: 1.45s
# lu: 1.38s