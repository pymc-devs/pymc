# TODO: When you use the one-at-a-time style, each point has to compute its M and C
# TODO: based on EACH OTHER POINT. So you need to store dev_sofar with that in mind.
# TODO: Two options: Either store each evaluation vector as a chunk and curse over
# TODO: the chunks, or do EVERYTHING one-at-a-time. Recommend the former.

# TODO: Check in __call__ if this argument is already in mesh so far. If so, skip
# TODO: the recompute. Also, it's too slow! Speed it up.

__docformat__='reStructuredText'

"""
Indexable and callable. Values on base mesh ('static') must always exist,
indexing returns one of those. Also keeps two nodes and an array:

"""

from numpy import *
from numpy.random import normal
from numpy.linalg import cholesky, eigh, solve
from Covariance import Covariance
from Mean import Mean
from GPutils import regularize_array, enlarge_covariance

class Realization(ndarray):
    
    """
    f = Realization(M, C[, init_base_array])
    
    A realization from a Gaussian process. It's indexable and callable.
    
    :Arguments:
        - M: A Gaussian process mean function.
        - C: A Gaussian process covariance function.
        - init_base_array:  An optional ndarray giving the value of f over its base mesh (got from C). If no value is given, f's value over its base mesh is sampled given M and C.
                        
    :SeeAlso:
    GPMean, GPCovariance, GaussianProcess, condition
    """
    
    def mean_fun():
        pass
    def cov_fun():
        pass
                
    cov_params = None
    mean_params = None
    base_mesh = None
    base_reshape = None

    ndim = None
    C = None
    M = None

    N_obs_sofar = 0
    obs_mesh_sofar = None
    dev_sofar = None
    M_sofar = None
    C_sofar = None
    
    __array_priority__ = 0.

    def __new__(subtype, 
                M,
                C,
                init_base_array = None):
        
        # You may need to reshape these so f2py doesn't puke.           
        base_mesh = C.base_mesh
        base_reshape = C.base_reshape
        cov_params = C.params
        mean_params = M.mean_params
        
        ndim = C.ndim
            
        if base_mesh is not None:
            
            length = base_reshape.shape[0]
            obs_mesh_sofar = base_reshape
            M_sofar = M.ravel()
            C_sofar = C
            N_obs_sofar = length
            
            if init_base_array is not None:
                # If the value over the base array is specified, check it out.
                
                if not init_base_array.shape == M.shape:
                    raise ValueErrror, 'Argument init_base_array must be same shape as M.'

                f = init_base_array.view(subtype)

            else:
                # Otherwise, draw a value over the base array.

                q=reshape(asarray(C.S.T * normal(size = length)), base_mesh.shape)
                f = (M+q).view(subtype)
        else:
            f = array([]).view(subtype)
            base_reshape = array([])
            obs_mesh_sofar = None
            M_sofar = None
            C_sofar = None
            N_obs_sofar = 0
        
        f.obs_mesh_sofar = obs_mesh_sofar
        f.M_sofar = M_sofar
        f.C_sofar = C_sofar
        f.N_obs_sofar = N_obs_sofar
        f.base_mesh = base_mesh
        f.cov_params = cov_params
        f.mean_params = mean_params
        f.base_reshape = base_reshape        
        f.cov_fun = C.eval_fun
        f.C = C
        f.mean_fun = M.eval_fun
        f.M = M
        f.ndim = C.ndim
        f.dev_sofar = f.view(ndarray).ravel() - f.M_sofar
        
        return f
        
            
    def __call__(self, x):

        orig_shape = shape(x)
        x=regularize_array(x)
        xndim = x.shape[-1]
        
        # Either compare x's number of dimensions to self's number of dimensions,
        # or else set self's number of dimensions to that of x.
        if self.ndim is not None:
            if not xndim == self.ndim:
                raise ValueError, "The last dimension of x (the number of spatial dimensions) must be the same as self's ndim."
        else:
            self.ndim = xndim
            
        x = x.reshape(-1,self.ndim)
        lenx = x.shape[0]
        
        M_now = self.M(x).ravel()
        C_now = self.C(x)
        
        M_pure = M_now.copy()
        C_pure = C_now.copy()
        
        # print self.N_obs_sofar
        # First observation:
        if self.N_obs_sofar == 0:
            
            self.M_sofar = M_now
            self.C_sofar = C_now
            self.obs_mesh_sofar = x

        # Subsequent observations:    
        else:    

            # Iterative conditioning may be better for this, but it is probably not:

            RF = self.C(x,self.obs_mesh_sofar)
                        
            C_now -= RF * solve(self.C_sofar, RF.T)
            M_now += asarray(RF * solve(self.C_sofar, self.dev_sofar)).ravel()

            self.obs_mesh_sofar = concatenate((self.obs_mesh_sofar, x), axis=0)
            self.M_sofar = concatenate((self.M_sofar, M_pure), axis=0)
            self.C_sofar = enlarge_covariance(self.C_sofar, RF, C_pure)
            
        
        try:
            sig = cholesky(C_now)

        except:
            val, vec = eigh(C_now)
            sig = asmatrix(zeros(vec.shape))

            for i in range(len(val)):
                if val[i]<0.:
                    val[i]=0.
                sig[:,i] = vec[:,i]*sqrt(val[i])

        f = M_now + asarray(sig * normal(size=lenx)).ravel()


        
        if self.N_obs_sofar > 0:
            self.dev_sofar = concatenate((self.dev_sofar, f - M_pure), axis=0)
        else:
            self.dev_sofar = f - M_pure
            
        self.N_obs_sofar += lenx
        
        return f.reshape(orig_shape)                

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

        mean_fun_part = 'Associated mean function: ' + str(self.mean_fun)
        cov_fun_part = 'Associated covariance function: ' + str(self.cov_fun)
        obs_part = 'Number of evaluations so far: %i' % self.N_obs_sofar

        return '\n'.join(['Gaussian process realization',mean_fun_part,cov_fun_part,obs_part,array_part])
                
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