"""
Indexable and callable. Values on base mesh ('static') must always exist,
indexing returns one of those. Also keeps two nodes and an array:

"""

from numpy import *
from numpy.random import normal
from PyMC2 import msqrt
from GPCovariance import GPCovariance
from GPMean import GPMean
from GPutils import *

class GPRealization(ndarray):
    
    def eval_fun():
        pass
    def cov_fun():
        pass
                
    cov_params = None
    mean_params = None
    base_mesh = None
    base_reshape = None

    ndim = None
    n_obs_sofar = None
    obs_mesh_sofar = None
    M_sofar = None
    C_sofar = None
    C = None
    M = None
    
    __array_priority__ = 0.

    def __new__(subtype, 
                M,
                C,
                init_base_array = None):
        
        # You may need to reshape these so f2py doesn't puke.           
        base_mesh = C.base_mesh
        cov_params = C.params
        mean_params = M.mean_params
        
        if len(base_mesh.shape)>1:
            ndim = base_mesh.shape[-1]
        else:
            ndim = 1

        base_reshape = base_mesh.reshape(-1,ndim)
        length = base_reshape.shape[0]
                
        if init_base_array is not None:
            f = init_base_array.view(subtype)
        else:
            q=reshape(asarray(C.S.T * normal(size = length)), base_mesh.shape)
            f = q.view(subtype)
        
        f.base_mesh = base_mesh
        f.cov_params = cov_params
        f.mean_params = mean_params
        f.base_reshape = base_reshape        
        f.cov_fun = C.eval_fun
        f.C = C
        f.mean_fun = M.eval_fun
        f.M = M
        
        return f
        
            
    def __call__(self, x):

        if not isinstance(x,ndarray):
            x=array([x])
        orig_shape = x.shape
        x=x.reshape(-1,self.ndim)
        lenx = x.shape[0]

        
    def __repr__(self):
        # Thanks to the author of ma.array for this code.
        s = repr(self.__array__()).replace('array', 'array aspect: ')

        l = s.splitlines()
        for i in range(1, len(l)):
            if l[i]:
                l[i] = ' '*9 + l[i]
        array_part = '\n'.join(l)

        mean_fun_part = 'Associated mean function: ' + str(self.mean_fun)
        cov_fun_part = 'associated covariance function: ' + str(self.cov_fun)

        return '\n'.join(['Gaussian process realization',mean_fun_part,cov_fun_part,array_part])
        

    def __array_wrap__(self, array_in):
        return array_in.view(ndarray)
        

    def plot(self, mesh=None):
        from pylab import plot
        if self.ndim==1:
            plot(self.base_mesh, self.view(ndarray))
        elif self.ndim==2:
            contourf(self.base_mesh[:,0], self.base_mesh[:,1],self.view(ndarray))