from numpy import ndarray, asarray, array, matrix

def regularize_array(A):
    """
    Takes an ndarray as an input.
    - If the array is one-dimensional, it's assumed to be an array of input values.
    - If the array is more than one-dimensional, its last index is assumed to curse
      over spatial dimension.
    
    Either way, the return value is at least two dimensional. A.shape[-1] gives the
    number of spatial dimensions.
    """
    if not isinstance(A,ndarray):
        A = array(A)
    
    if len(A.shape) <= 1:
        return A.reshape(-1,1)
        
    else:
        return asarray(A, dtype=float)
        
def fwrap(cov_fun):

    def wrapper(x,y,*args,**kwargs):
        symm = (x is y)
        kwargs['symm']=symm
        return cov_fun(regularize_array(x),regularize_array(y),*args,**kwargs).view(matrix)

    return wrapper