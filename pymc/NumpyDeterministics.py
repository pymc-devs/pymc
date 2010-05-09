"""
pymc.NumpyDeterministics
"""

__docformat__='reStructuredText'
import PyMCObjects as pm
import numpy as np
from numpy import sum, ones, zeros, ravel, shape, size, newaxis

__all__ = ['sum']#+['iter_','complex_','int_','long_','float_','oct_','hex_']

#accumulations 
_accumulation_deterministics = ['sum', 'prod']


#transformations (broadcasted)
_generic = ['abs', 'hypot', 'exp', 'log', 'sqrt']
_trig = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2']
_hyp_trig = ['sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh']
_transformation_deterministics = _generic + _trig + _hyp_trig


def ndSum(a, axis = None):
    r = np.sum(a, axis = axis)
    return r


_sum_hist = {}
def sum_jacobian (parameter, a, axis):
    try:
        return _sum_hist[shape(a)]
    except KeyError:
        j = ones(shape(a))
        _sum_hist[shape(a)] = j
        return j
    
def sum(a, axis = None):

    parents = {'a': a, 'axis' : axis}
    
    return pm.Deterministic(eval = np.sum, 
                            doc = "sum", 
                            name = "sum(" + str(a) + ", axis =" + str(axis) + ")",
                            parents = parents,
                            trace=False,
                            plot=False,
                            jacobian = sum_jacobian,
                            jacobian_format = {'a' : 'accumulation_operation'})
    
def abs_jacobian(parameter, a):
    return np.sign(a)

def sin_jacobian(parameter, a):
    return np.cos(a)

def cos_jacobian(parameter, a):
    return -np.sin(a)

def exp_jacobian(parameter, a):
    return np.exp(a)

def log_jacobian(parameter, a):
    return 1.0/a

