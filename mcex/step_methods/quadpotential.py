from utils import normal
from numpy import dot
from numpy.linalg import solve
from scipy.linalg import cholesky, cho_solve

def quad_potential(C, is_cov):
    if C.ndim == 1:
        if is_cov:
            return ElemWiseQuadPotential(C)
        else: 
            return ElemWiseQuadPotential(1./C)
    else:
        if is_cov:
            return QuadPotential(C)
        else :
            return QuadPotential_Inv(C) 

class ElemWiseQuadPotential(object):
    def __init__(self, v):
        self.s = v**.5
    def velocity(self, x):
        return self.s* x
    def random(self):
        return self.velocity(normal(size = self.s.shape))
    def energy(self, x):
        return dot(self.s, x)**2

class QuadPotential_Inv(object):
    def __init__(self, A):
        self.L = cholesky(A, lower = True)
        
    def velocity(self, x ):
        return cho_solve((self.L, True), x)
        
    def random(self):
        n = normal(size = self.L.shape[0])
        return dot(self.L, n)
    
    def energy(self, x):
        L1x = solve(self.L, x)
        return .5 * dot(L1x.T, L1x)


class QuadPotential(object):
    def __init__(self, A):
        self.A = A
        self.L = cholesky(A, lower = True)
    
    def velocity(self, x):
        return dot(self.A, x)
    
    def random(self):
        n = normal(size = self.L.shape[0])
        return solve(self.L.T, n)
    
    def energy(self, x):
        return .5 * dot(x, dot(self.A, x))
        