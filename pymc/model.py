'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from point import *
from types import *

import theano
from theano import function, scan
from theano.tensor import TensorType, add, sum, grad, arange, flatten, concatenate, constant

import numpy as np 

__all__ = ['Model', 'logp', 'dlogp', 'continuous_vars'] 

def Variable(name, shape, dtype='float64'):
    """
    Creates a TensorVariable of the given shape and type
    
    Parameters
    ----------
    
    shape : int or vector of ints        
    dtype : str
    
    Examples
    --------
    
    
    """
    shape = np.atleast_1d(shape)
    var = TensorType(str(dtype), shape == 1)(name)
    var.dshape = tuple(shape)
    var.dsize = int(np.prod(shape))
    return var

class Model(object):
    """
    Base class for encapsulation of the variables and 
    likelihood factors of a model.
    """
    
    def __init__(self, test_point):
       self.vars = []
       self.factors = [] 
       self.test_point = clean_point(test_point)
       if test_point is not None:
           theano.config.compute_test_value = 'raise'
       else:
           theano.config.compute_test_value = 'off'

    """
    these functions add random variables
    it is totally appropriate to add new functions to Model
    """
    def Data(model, data, distribution):
        args = map(constant, as_iterargs(data))
        model.factors.append(distribution(*args))

    def Var(model, name, distribution, shape = 1, dtype = 'float64'):
        var = Variable(name, shape, dtype)
        model.vars.append(var)
        if model.test_point is not None: 
            var.tag.test_value = model.test_point[name]
        model.factors.append(distribution(var))
        return var

    def fn(self, calc, mode = None):
        if hasattr(calc, '__iter__'):
            out = [f(self) for f in calc]
        else:
            out = calc(self)

        return PointFunc(function(self.vars,
                    out, 
                    allow_input_downcast = True, mode = mode))

    def logp(self):
        return self.fn(logp)

    def dlogp(self, *args, **kwargs):
        return self.fn(dlogp(*args, **kwargs))


     

def as_iterargs(data):
    if isinstance(data, tuple): 
        return data
    if hasattr(data, 'columns'): #data frames
        return [np.asarray(data[c]) for c in data.columns] 
    else:
        return [data]

       
def continuous_vars(model):
    return [ var for var in model.vars if var.dtype in continuous_types] 

class PointFunc(object): 
    def __init__(self, f):
        self.f = f
    def __call__(self,state):
        return self.f(**state)

"""
These functions build log-posterior graphs (and derivatives)
""" 
def logp(model):
    """
    Calculates the log-probability of a specified model
        
    Parameters
    ----------
        
    model : Model  
        
    Examples
    --------
        
    >>> an example
        
    """
    return add(*map(sum, model.factors))

def flatgrad(f, v):
    return flatten(grad(f, v))

def gradient(f, dvars):
    return concatenate([flatgrad(f, v) for v in dvars])
def jacobian(f, dvars):
    def jac(v):
        def grad_i(i, f1, v): 
            return flatgrad(f1[i], v)

        return scan(grad_i, sequences=arange(f.shape[0]), non_sequences=[f,v])[0]

    return concatenate(map(jac, dvars))

def hessian(f, dvars):
    return jacobian(gradient(f, dvars), dvars)

def dlogp(dvars=None, n=1):
    """
    Returns a function for calculating the derivative of the output 
    of another function.
    
    Parameters
    ----------
    d_calc : function
    
    Returns
    -------
    der_calc : function
    """
    
    dfn = [gradient, hessian]
    def dlogp_calc(model):
         
        if dvars is None:
            vars = continuous_vars(model)
        else: 
            vars = dvars 
        return dfn[n-1](logp(model), vars)

    return dlogp_calc


#theano stuff 
theano.config.warn.sum_div_dimshuffle_bug = False
