from GaussianProcess import *
from GaussianProcess.cov_funs import matern
from numpy import *
from numpy.linalg import eigh, cholesky
from scipy.linalg import lu


# x will be the base mesh
x = ones(3,dtype=float)


# Create a Covariance object with x as the base mesh
C = Covariance( eval_fun = matern, 
                diff_degree = 1.4, amp = .4, scale = .5)

B=asmatrix(ones((3,3),dtype=float))
B[2,2] = 1.5
# B[1,1]=1.5

# B=C([1,2,3])

good_rows, U=robust_chol(B)
b= gentle_trisolve(U,x,good_rows = good_rows)
print U.T*U
print U[:,good_rows]*b
