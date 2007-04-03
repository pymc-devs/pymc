"""
Just like GP_cov, but with a mean. Will need to have a covariance
matrix as a parent to implement conditioning.
"""

from numpy import *

class GPMean(ndarray):
    
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
    
    __array_priority__ = 0.
    
    def __new__(subtype, 
                eval_fun,
                C,   
                **mean_params):
        
        # You may need to reshape these so f2py doesn't puke.           
        base_mesh = C.base_mesh
        cov_params = C.params
        cov_fun = C.eval_fun
        
        # if not sum(obs_vals.shape) == sum(C.obs_mesh.shape):
        #     raise ValueError, 'C.obs_mesh and obs_vals must be the same shape. ' + \
        #                         "C.obs_mesh's shape is " + repr(C.obs_mesh.shape) + \
        #                         " but obs_vals' shape is " + repr(obs_vals.shape)
        
        if len(base_mesh.shape)>1:
            ndim = base_mesh.shape[-1]
        else:
            ndim = 1
            
        base_reshape = base_mesh.reshape(-1,ndim)        
        length = base_reshape.shape[0]        
        # Call the covariance evaluation function
        data=eval_fun(base_mesh, **mean_params)
                
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

        not_array = False

        if not isinstance(x,ndarray):
            x=array([x])
        orig_shape = x.shape
        x=x.reshape(-1,self.ndim)
        lenx = x.shape[0]

        # Evaluate the unconditioned mean
        M = self.eval_fun(x,**self.mean_params)
        if not self.conditioned:
            return M.reshape(orig_shape)
        
        # Condition
        RF = zeros((lenx,self.obs_len),dtype=float)
        self.cov_fun(RF, x, self.obs_mesh, **self.cov_params)
        M += (RF * self.Q_mean_under).view(ndarray).reshape(M.shape)
            
        return M.reshape(orig_shape)

    def __repr__(self):
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

    def __array_wrap__(self, array_in):
        return array_in.view(ndarray)

    def plot(self, mesh=None):
        from pylab import plot
        if self.ndim==1:
            plot(self.base_mesh, self.view(ndarray))
        elif self.ndim==2:
            contourf(self.base_mesh[:,0], self.base_mesh[:,1],self.view(ndarray))