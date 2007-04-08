__docformat__='reStructuredText'

# TODO: implement lintrans and dependent observation precision at some point.

from numpy import *
from numpy.linalg import eigh, solve
from GPutils import regularize_array


class Covariance(matrix):
    
    """
    C=Covariance(eval_fun, base_mesh, **params)
    
    Valued as a special GP covariance object. On instantiation,
    this object produces a covariance matrix over base_mesh, according to actual_GP_fun,
    with parameters **params, conditioned on lintrans acting on its evaluations at 
    obs_mesh, with `observation precision' obs_taus.
    
    The object is indexable and callable. When indexed, it returns one of its stored
    elements (static usage). When called with a point couple as arguments (dynamic usage), 
    it returns the covariance of that point couple, again conditioned on the lintrans and obs_mesh
    and all.
    
    :Arguments:
    - eval_fun: A function that takes a matrix as its first argument
                and two ndarrays or vectors as its second and third 
                arguments, and writes a covariance matrix in-place over
                its first argument.
    - base_mesh: The base mesh.
    - params: Parameters to be passed to eval_fun.
    
    :Attributes:
    - S: Matrix square root.
    - Eval: Eigenvalues
    - Evec: Eigenvectors
    - logD: log-determinant
    
    :SeeAlso:
    GPMean, GPRealization, GaussianProcess, condition
    """
    
    def eval_fun():
        pass        
    params = None
    base_mesh = None
    base_reshape = None
    ndim = None
    conditioned = False
    obs_mesh = None
    base_mesh = None
    obs_taus = None 
    Q = None   
    obs_len = None
    __matrix__ = None
    observed = False
    
    __array_priority__ = 0.
    
    def __new__(subtype, 
                eval_fun,   
                base_mesh = None,
                **params):

        if not base_mesh is None:
            base_mesh = regularize_array(base_mesh)
        
            ndim = base_mesh.shape[-1]
            base_reshape = base_mesh.reshape(-1,ndim)            
            length = base_reshape.shape[0]        
        
            # Call the covariance evaluation function
            data=eval_fun(base_reshape, base_reshape, **params)
        else:
            data = array([])
            base_reshape = None
            ndim = None

        C = data.view(subtype)    
            
        C.base_mesh = base_mesh
        C.base_reshape = base_reshape
        C.ndim = ndim    
        C.eval_fun = eval_fun
        C.params = params
        C.__matrix__ = data
        C.update_sig_and_e()
        
        # Return the data
        return C
    
    def update_sig_and_e(self):
        if self.base_mesh is not None:
            val, vec = eigh(self)
            self.Eval = val
            self.Evec = vec
            sig = asmatrix(zeros(vec.shape))
            for i in range(len(val)):
                if val[i]<0.:
                    val[i]=0.
                sig[:,i] = vec[:,i]*sqrt(val[i])
            self.S = sig.T
        
            self.logEval = log(val)
        else:
            self.Eval = array([])
            self.Evec = array([])
            self.S = array([])
            self.logEval = array([])
                
    def __call__(self, x, y=None):
        
        x=regularize_array(x)
        ndimx = x.shape[-1]
        x=x.reshape(-1,ndimx)
        lenx = x.shape[0]

        if self.ndim is not None:
            if not self.ndim == ndimx:
                raise ValueError, "The number of spatial dimensions of x does not match the number of spatial dimensions of the Covariance instance's base mesh."
            
        if y is None or y is x:
            
            C=self.eval_fun(x,x,**self.params)

            if not self.conditioned:
                return C
                
            RF = self.eval_fun(x, self.obs_mesh, **self.params)
            C -= RF * solve(self.Q , RF.T)
            
            return C

        else:
            y=regularize_array(y)
            ndimy = y.shape[-1]
            y=y.reshape(-1,ndimx)
            leny = y.shape[0]
            
            if not ndimx==ndimy:
                raise ValueError, 'The last dimension of x and y (the number of spatial dimensions) must be the same.'
            
            combo = vstack((x,y))
        
            # Evaluate the covariance
            C=self.eval_fun(combo,combo,**self.params)
            if not self.conditioned:
                return C[:lenx,lenx:]

            # Condition
            RF = self.eval_fun(combo, self.obs_mesh, **self.params)
            C -= RF * solve(self.Q, RF.T)
            
            return C[:lenx,lenx:]

    def __repr__(self):
        return object.__repr__(self)        
        
    def __str__(self):
        # Thanks to the author of ma.array for this code.
        s = repr(self.__array__()).replace('array', 'matrix aspect: ')
        
        l = s.splitlines()
        for i in range(1, len(l)):
            if l[i]:
                l[i] = ' '*10 + l[i]
        matrix_part = '\n'.join(l)
        
        functional_part = 'functional aspect: ' + str(self.eval_fun)
        
        return '\n'.join(['Gaussian process covariance',functional_part,matrix_part])
        
    def __getitem__(self, *args):
        return self.view(matrix).__getitem__(*args)
    
    def __getslice__(self, *args):
        return self.view(matrix).__getslice__(*args)     
            
    # def __ipow__(self)
    # def __pow__(self)
    # def __idiv__(self)
    # def __imul__(self)
    # def __iadd__(self)   
    # def __isub__(self)
    
    def __array_wrap__(self, matrix_in):
        return matrix_in.view(matrix)
        
    # Also need to forbid in-place multiplication and implement some methods, otherwise 
    # exponentiation makes it puke, etc.

    def plot(self, mesh=None):
        from pylab import contourf, colorbar
        if self.ndim==1:
            contourf(self.base_mesh, self.base_mesh, self.view(ndarray))
        else:
            contourf(self.view(ndarray))
        colorbar()            