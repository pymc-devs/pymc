from ..core import *

def array_step(astep, f, vars):
    mapping = DASpaceMap(vars)
    
    def step(state, chain):
        def fn( a):
            return f(mapping.rproject(a, chain))
        
        state, achain = astep(fn, state, mapping.project(chain))
        return state, mapping.rproject(achain, chain)
    return step


from numpy.random import uniform, normal
from numpy import dot, log , isfinite

def metrop_select(mr, q, q0):
    if isfinite(mr) and log(uniform()) < mr:
        return q
    else: 
        return q0
    
def cholesky_normal(size , cholC):
    return dot(normal(size = size), cholC)



def eigen(a, n = -1): 
    
    #if len(a.shape) == 0: # if we got a 0-dimensional array we have to turn it back into a 2 dimensional one 
    #    a = a[np.newaxis,np.newaxis]
    a = np.atleast_2d(a)
       
    if n == -1:
        n = a.shape[0]
        
    eigenvalues, eigenvectors = np.linalg.eigh(a)

    indicies = np.argsort(eigenvalues)[::-1]
    return eigenvalues[indicies[0:n]], eigenvectors[:,indicies[0:n]]