'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from point import *
from types import *

from theano import theano, tensor as t, function
from theano.gof.graph import inputs

import numpy as np 

__all__ = ['Model', 'compilef', 'gradient', 'hessian'] 

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
    """
    def Data(model, data, dist):
        args = map(t.constant, as_iterargs(data))
        model.factors.append(dist.logp(*args))

    def Var(model, name, dist, shape = 1, dtype = None, testval = try_defaults):
        if not dtype:
            dtype = default_type[dist.support]
        
        var = Variable(name, shape, dtype, get_test_val(dist, testval))

        model.vars.append(var)
        model.factors.append(dist.logp(var))
        return var

    def TransformedVar(model, name, dist, transform, logjacobian, shape = 1, dtype = None, testval = try_defaults): 
        if not dtype:
            dtype = default_type[dist.support]
        
        var = Variable('transformed_' + name, shape, dtype, get_test_val(dist, testval))

        model.vars.append(var)

        tvar = transform(var)
        model.factors.append(dist.logp(tvar) + logjacobian(var))

        return tvar, var

    @property
    def logp(model):
        """
        log-probability of the model
            
        Parameters
        ----------
            
        model : Model  

        Returns
        -------

        logp : Theano scalar
            
        """
        return t.add(*map(t.sum, model.factors))

    @property
    def logpc(model): 
        return compilef(model.logp)

    def dlogpc(model, vars = None): 
        return compilef(gradient(model.logp, vars))

    def d2logpc(model, vars = None):
        return compilef(hessian(model.logp, vars))

    @property
    def test_point(self):
        return dict( (str(var), var.tag.test_value) for var in self.vars)

    @property
    def cont_vars(model):
        return typefilter(model.vars, continuous_types) 

def compilef(outs, mode = None):
    return PointFunc(
                function(inputvars(outs), outs, 
                         allow_input_downcast = True, 
                         on_unused_input = 'ignore',
                         mode = mode)
           )


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

def makeiter(a): 
    if isinstance(a, (tuple, list)):
        return a
    else :
        return [a]

def inputvars(a): 
    return [v for v in inputs(makeiter(a)) if isinstance(v, t.TensorVariable)]

"""
Theano derivative functions 
""" 

def cont_inputs(f):
    return typefilter(inputvars(f), continuous_types)

def gradient1(f, v):
    """flat gradient of f wrt v"""
    return t.flatten(t.grad(f, v))

def gradient(f, vars = None):
    if not vars: 
        vars = cont_inputs(f)

    return t.concatenate([gradient1(f, v) for v in vars], axis = 0)

def jacobian1(f, v):
    """jacobian of f wrt v"""
    f = t.flatten(f)
    idx = t.arange(f.shape[0])
    
    def grad_i(i): 
        return gradient1(f[i], v)

    return theano.map(grad_i, idx)[0]

def jacobian(f, vars = None):
    if not vars: 
        vars = cont_inputs(f)

    return t.concatenate([jacobian1(f, v) for v in vars], axis = 1)

def hessian(f, vars = None):
    return -jacobian(gradient(f, vars), vars)


#theano stuff 
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
