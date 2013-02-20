'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from point import *
from types import *

import theano
import theano.tensor as t

from theano import function, scan


import numpy as np 

__all__ = ['Model', 'logp', 'dlogp', 'continuous_vars'] 

def Variable(name, shape, dtype, testval):
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
    var = t.TensorType(str(dtype), shape == 1)(name)
    var.dshape = tuple(shape)
    var.dsize = int(np.prod(shape))
    var.tag.test_value = np.ones(shape, dtype)*testval

    return var

val_defaults = {'discrete'   : ['mode'], 
                'continuous' : ['median', 'mean', 'mode']}

def try_defaults(dist):
    vals =  val_defaults[dist.support]

    for val in vals: 
        if hasattr(dist, val): 
            return getattr(dist,val)
    raise AttributeError(str(dist) + " does not have a value for any of: " + str(vals))



class Model(object):
    """
    Base class for encapsulation of the variables and 
    likelihood factors of a model.
    """
    
    def __init__(self):
       self.vars = []
       self.factors = [] 

    """
    these functions add random variables
    it is totally appropriate to add new functions to Model
    """
    def Data(model, data, distribution):
        args = map(t.constant, as_iterargs(data))
        model.factors.append(distribution.logp(*args))

    def Var(model, name, distribution, shape = 1, dtype = None, testval = try_defaults):
        if not dtype:
            dtype = default_type[distribution.support]
        
        var = Variable(name, shape, dtype, get_test_val(distribution, testval))

        model.vars.append(var)
        model.factors.append(distribution.logp(var))
        return var


    def fn(self, calc, mode = None):
        if hasattr(calc, '__call__'):
            out = calc(self)
        else:
            out = [f(self) for f in calc]

        return PointFunc(
                function(self.vars,out, 
                    allow_input_downcast = True, mode = mode))

    def logp(self):
        return self.fn(logp)

    def dlogp(self, *args, **kwargs):
        return self.fn(dlogp(*args, **kwargs))

    @property
    def test_point(self):
        return dict( (str(var), var.tag.test_value) for var in self.vars)




def as_iterargs(data):
    if isinstance(data, tuple): 
        return data
    if hasattr(data, 'columns'): #data frames
        return [np.asarray(data[c]) for c in data.columns] 
    else:
        return [data]

def get_test_val(dist, val):
    try :
        val = getattr(dist, val)
    except TypeError:
        pass

    if hasattr(val, '__call__'):
        val = val(dist)

    if isinstance(val, t.TensorVariable):
        return val.tag.test_value
    else:
        return val

    



       
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
    return t.add(*map(t.sum, model.factors))

def flatgrad(f, v):
    return t.flatten(t.grad(f, v))

def gradient(f, dvars):
    return t.concatenate([flatgrad(f, v) for v in dvars])
def jacobian(f, dvars):
    def jac(v):
        def grad_i(i, f1, v): 
            return flatgrad(f1[i], v)

        return scan(grad_i, sequences=t.arange(f.shape[0]), non_sequences=[f,v])[0]

    return t.concatenate(map(jac, dvars))

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
theano.config.compute_test_value = 'raise'
