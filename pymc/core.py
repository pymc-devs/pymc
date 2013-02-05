'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import theano
from theano import function, scan
from theano.tensor import TensorType, add, sum, grad,  flatten, arange, concatenate, constant

import numpy as np 
import time 
from history import NpHistory

# TODO Can we change this to just 'Variable'? 
def FreeVariable(name, shape, dtype='float64'):
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
    
    def __init__(self):
       self.vars = []
       self.factors = [] 

"""
these functions add random variables
"""

def make_constants(data) : 
    """
    Helper function for creating constants from data
        
    Parameters
    ----------
        
    data : tuple, list, array  
        
    Examples
    --------
        
    """
    if isinstance(data, tuple): 
        return tuple(map(constant, data))
    else:
        return constant(data)
        
        
def AddData(model, data, distribution):
    """
    Adds data node to specified model, according to specified distribution
        
    Parameters
    ----------
        
    model : Model
    data : vector, list or array
    distribution : function 
        
    Examples
    --------

    >>> model = Model()
    >>> data = np.random.normal(size = (2, 20))
    >>> AddData(model, data, Normal(mu = x, tau = .75**-2))
        
    """
    
    model.factors.append(distribution(make_constants(data)))

def AddVar(model, name, distribution, shape=1, dtype='float64'):
    """
    Adds variable to specified model
        
    Parameters
    ----------
        
    model : Model
        Model object to which variable is to be added
    name : str
        Variable name
    distribution : function
        Distribution associated with variable
    shape : int or tuple
        Shape of variable (defaults to 1)
    dtype : str  
        Type of variable (defaults to float64)
        
    Examples
    --------
    
    >>> model = Model()
    >>> x = AddVar(model, 'x', Normal(mu = .5, tau = 2.**-2), (2,1))
        
    """
    var = FreeVariable(name, shape, dtype)
    model.vars.append(var)
    model.factors.append(distribution(var))
    return var
    
# TODO Document AddVarIndirectElemewise
def AddVarIndirectElemewise(model, name, proximate_calc, distribution, shape=1):
    var = FreeVariable(name, shape)
    model.vars.append(var)
    prox_var = proximate_calc(var)
    
    model.factors.append(distribution(prox_var) + log_jacobian_determinant(prox_var, var))
    
    return var
    
    
def continuous_vars(model):
    """
    Returns a list of the continuous variables in a specified model
        
    Parameters
    ----------
        
    model : Model  
        
    """
    
    return [var for var in model.vars if var.dtype in continuous_types] 


"""
these functions compile log-posterior functions (and derivatives)
"""
def model_func(model, calcs, mode=None):
    f = function(model.vars, 
             calcs,
             allow_input_downcast=True, mode= mode)
    def fn(state):
        return f(**state)
    return fn

def model_logp(model, mode = None):
    return model_func(model, logp_calc(model), mode)

def model_dlogp(model, dvars = None, mode = None ):    
    return model_func(model, dlogp_calc(model, dvars), mode)
    
def model_logp_dlogp(model, dvars = None, mode = None ):    
    return model_func(model, [logp_calc(model), dlogp_calc(model, dvars)], mode)


"""
The functions build graphs from other graphs
"""
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



def hessian_diag(f, dvars):
    def hess(v):
        df = flatten(grad(f, v))
        def grad_i (i, df1, v): 
            return flatgrad(df1[i], v)[i]
        
        return scan(grad_i, sequences = arange(f.shape[0]), non_sequences = [df,v])[0]

    return concatenate(map(hess, dvars))

def log_jacobian_determinant(var1, var2):
    # need to find a way to calculate the log of the jacobian determinant easily, 
    raise NotImplementedError()
    # in the case of elemwise operations we can just sum the gradients
    # so we might just test if var1 is elemwise wrt to var2 and then calculate the gradients, summing their logs
    # otherwise throw an error

"""
These functions build log-posterior graphs (and derivatives)
""" 
def logp_calc(model):
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

def dercalc(d_calc):
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
    def der_calc(model, dvars = None):
        if dvars is None:
            dvars = continuous_vars(model)
        
        return d_calc(logp_calc(model), dvars)
    return der_calc

dlogp_calc = dercalc(gradient)
hess_calc = dercalc(jacobian)
hess_diag_calc = dercalc(hessian_diag)

    
class DASpaceMap(object):
    """ 
    DASpaceMap encapsulates a mapping of dict space <-> array 
    space
    """
    def __init__(self, free_vars):
        self.dimensions = 0
        
        self.slices = {}
        self.vars = {}
        for var in free_vars:       
            self.vars[str(var)] = var 
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.dsize)
            self.dimensions += var.dsize
            
    def project(self, d, a = None):
        """
        Projects dict space to array space
        
        Parameters
        ----------
        
        d : dict
        a : array
            Defaults to None
        
        """
        if a is None:
            a = np.empty(self.dimensions)
        else:
            a = np.copy(a)
        
        for varname, value in d.iteritems():
            try:    
                a[self.slices[varname]] = np.ravel(value)
            except KeyError:
                pass
        return a
    
    def rproject(self, a, d = {}):
        """
        Projects array space to dict space
        
        Parameters
        ----------
        
        a : array
        d : dict
            Defaults to empty dict
        
        """
        d = d.copy()
            
        for var, slc in self.slices.iteritems():
            d[var] = np.reshape(a[slc], self.vars[var].dshape)
            
        return d

# TODO Can we change `sample_history` to `trace`?
def sample(draws, step, point, sample_history=None, state=None):
    """
    Draw a number of samples using the given step method. 
    Multiple step methods supported via compound step method 
    returns the amount of time taken.
        
    Parameters
    ----------
        
    draws : int  
        The number of samples to draw
    step : function
        A step function
    point : float or vector
        The current sample index
    sample_history : NpHistory
        A trace of past values (defaults to None)
    state : 
        The current state of the sampler (defaults to None)
        
    Examples
    --------
        
    >>> an example
        
    """
    
    # Instantiate a trace if there is not one passed
    if not sample_history :
        sample_history = NpHistory(draws)
    
    # Keep track of sampling time    
    tstart = time.time()
    
    for _ in xrange(int(draws)):
        state, point = step(state, point)
        sample_history.record(point)
        
    return sample_history, state, (time.time() - tstart)

# Sets of dtypes

bool_types = set(['int8'])
   
int_types = set(['int8',
            'int16' ,   
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64'])
float_types = set(['float32',
              'float64'])
complex_types = set(['complex64',
                'complex128'])
continuous_types = float_types | complex_types
discrete_types = bool_types | int_types

#theano stuff 
theano.config.warn.sum_div_dimshuffle_bug = False
