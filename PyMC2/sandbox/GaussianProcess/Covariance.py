__docformat__='reStructuredText'

# TODO: implement lintrans and dependent observation precision at some point.
# TODO: Make the covariance functions return a matrix, not update it in-place. 
# TODO: That's stupid. There's no benefit so far for in-place updating.

from numpy import *
from PyMC2.utils import msqrt


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
    
    __array_priority__ = 0.
    
    def __new__(subtype, 
                eval_fun,   
                base_mesh = array([]),
                **params):
        
        # You may need to reshape these so f2py doesn't puke.           
        base_mesh = base_mesh.squeeze()
        
        if len(base_mesh.shape)>1:
            ndim = base_mesh.shape[-1]
        else:
            ndim = 1
            
        base_reshape = base_mesh.reshape(-1,ndim)
        length = base_reshape.shape[0]        
        data = asmatrix(zeros((length,length), dtype=float))

        C = data.view(subtype)
        
        # Call the covariance evaluation function
        eval_fun(data, base_reshape, base_reshape, **params)
        
        C.eval_fun = eval_fun
        C.params = params
        C.base_mesh = base_mesh
        C.base_reshape = base_reshape
        C.ndim = ndim
        C.__matrix__ = data
        
        # Return the data
        return C
                
    def __call__(self, x, y=None):
        
        if not isinstance(x,ndarray):
            x = array([x])
        x=x.reshape(-1,self.ndim)
        lenx = x.shape[0]
            
        if y is None:
            
            C = asmatrix(zeros((lenx,lenx)),dtype=float)
            self.eval_fun(C,x,x,**self.params)
            if not self.conditioned:
                return C
                
            RF = asmatrix(zeros((lenx,self.obs_len),dtype=float))
            self.eval_fun(RF, x, self.obs_mesh, **self.params)    
            C -= RF * self.Q * RF.T
            
            return C

        else:
            if not isinstance(y,ndarray):
                y = array([y])
            y=y.reshape(-1,self.ndim)
            leny = y.shape[0]
            
            combo = vstack((x,y))
            
            C = asmatrix(zeros((lenx+leny,lenx+leny),dtype=float))
        
            # Evaluate the covariance
            self.eval_fun(C,combo,combo,**self.params)
            if not self.conditioned:
                return C[:lenx,lenx:]

            # Condition        
            RF = asmatrix(zeros((lenx + leny, self.obs_len), dtype=float))
            self.eval_fun(RF, combo, self.obs_mesh, **self.params)
            C -= RF * self.Q * RF.T
            
            return C[:lenx,lenx:]
        
        
    def __repr__(self):
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
    
    def _sqrt(self):
        return msqrt(self)
    S = property(_sqrt)

    def plot(self, mesh=None):
        from pylab import contourf, colorbar
        if self.ndim==1:
            contourf(self.base_mesh, self.base_mesh, self.view(ndarray))
        else:
            contourf(self.view(ndarray))
        colorbar()            