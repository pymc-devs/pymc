__docformat__='reStructuredText'

from numpy import *
from GPutils import regularize_array

class Mean(ndarray):
    """
    M = GPMean(eval_fun, C, **mean_params)
    
    A Gaussian process mean function.
    
    :Arguments:
        - eval_fun: A function that takes a scalar or ndarray as its first argument and returns an ndarray.
        - C: A Gaussian process covariance function
        - mean_params: Parameters to be passed to eval_fun
    
    :SeeAlso: GPCovariance, GPRealization, GaussianProcess, condition
    """
    
    def eval_fun():
        pass
    def cov_fun():
        pass
                
    cov_params = None
    base_mesh = None
    base_reshape = None
    ndim = None
    conditioned = False
    obs_mesh = None
    base_mesh = None
    obs_vals = None
    obs_taus = None
    Q_mean_under = None
    obs_len = None
    observed = False
    
    __array_priority__ = 0.
    
    def __new__(subtype, 
                eval_fun,
                C,   
                **mean_params):
        
        # You may need to reshape these so f2py doesn't puke.           
        base_mesh = C.base_mesh
        cov_params = C.params
        cov_fun = C.eval_fun
        ndim = C.ndim
        base_reshape = C.base_reshape
            
        # Call the covariance evaluation function
        if base_mesh is not None:
            data=eval_fun(base_mesh, **mean_params)
        else:
            data=array([])
                
        M = data.view(subtype)        
        
        M.eval_fun = eval_fun
        M.cov_fun = cov_fun
        M.cov_params = cov_params
        M.mean_params = mean_params
        M.base_mesh = base_mesh
        M.ndim = ndim
        M.base_reshape = base_reshape
        
        # Return the data
        return M

    def __call__(self, x):
        
        orig_shape = shape(x)
        x=regularize_array(x)
        ndimx = x.shape[-1]
        x=x.reshape(-1, ndimx)
        lenx = x.shape[0]
        
        if self.ndim is not None:
            if not self.ndim == ndimx:
                raise ValueError, "The number of spatial dimensions of x does not match the number of spatial dimensions of the Mean instance's base mesh."        
        

        # Evaluate the unconditioned mean
        M = self.eval_fun(x,**self.mean_params)
        if not self.conditioned:
            return M.reshape(orig_shape)
        
        # Condition
        RF = self.cov_fun(x, self.obs_mesh, **self.cov_params)
        M += (RF * self.Q_mean_under).view(ndarray).reshape(M.shape)
            
        return M.reshape(orig_shape)
        
    def __repr__(self):
        return object.__repr__(self)        

    def __str__(self):
        # Thanks to the author of ma.array for this code.
        s = repr(self.__array__()).replace('array', 'array aspect: ')

        l = s.splitlines()
        for i in range(1, len(l)):
            if l[i]:
                l[i] = ' '*9 + l[i]
        array_part = '\n'.join(l)

        functional_part = 'functional aspect: ' + str(self.eval_fun)
        cov_fun_part = 'associated covariance function: ' + str(self.cov_fun)

        return '\n'.join(['Gaussian process covariance',functional_part,cov_fun_part,array_part])
        
    def __getitem__(self, *args):
        return self.view(ndarray).__getitem__(*args)

    def __getslice__(self, *args):
        return self.view(ndarray).__getslice__(*args)        

    def __array_wrap__(self, array_in):
        return array_in.view(ndarray)

    def plot(self, mesh=None):
        from pylab import plot
        if self.ndim==1:
            plot(self.base_mesh, self.view(ndarray))
        elif self.ndim==2:
            contourf(self.base_mesh[:,0], self.base_mesh[:,1],self.view(ndarray))