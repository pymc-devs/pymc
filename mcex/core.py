'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import theano
from theano.tensor import sum, grad, TensorType, TensorVariable
from theano import function
import numpy as np 
from __builtin__ import sum as buitin_sum
import time 

def FreeVariable( name, shape, dtype = 'float64'):
    """creates a TensorVariable of the given shape and type"""
    shape = np.atleast_1d(shape)
    var = TensorType(str(dtype), shape == 1)(name)
    var.dshape = tuple(shape)
    var.dsize = int(np.prod(shape))
    return var

class Model(object):
    """encapsulates the variables and the likelihood factors"""
    def __init__(self):
       self.vars = []
       self.factors = [] 

"""
these functions add random variables
"""

def AddData(model, data, distribution):
    model.factors.append(distribution(data))

def AddVar(model, name, distribution, shape = 1, dtype = 'float64'):
    var = FreeVariable(name, shape, dtype)
    model.vars.append(var)
    model.factors.append(distribution(var))
    return var
    
def AddVarIndirectElemewise(model, name,proximate_calc, distribution, shape = 1):
    var = FreeVariable(name, shape)
    model.vars.append(var)
    prox_var = proximate_calc(var)
    
    model.factors.append(distribution(prox_var) + log_jacobian_determinant(prox_var, var))
    return var
    
def log_jacobian_determinant(var1, var2):
    # need to find a way to calculate the log of the jacobian determinant easily, 
    raise NotImplementedError()
    # in the case of elemwise operations we can just sum the gradients
    # so we might just test if var1 is elemwise wrt to var2 and then calculate the gradients, summing their logs
    # otherwise throw an error
    return
    
def continuous_vars(model):
    return [ var for var in model.vars if var.dtype in continuous_types] 


"""
these functions compile log-posterior functions (and derivatives)
"""
def model_logp(model, mode = None):
    f = function(model.vars, logp_graph(model), allow_input_downcast = True, mode = mode)
    def fn(state):
        return f(**state)
    return fn

def model_dlogp(model, dvars = None, mode = None ):    
    if dvars is None :
        dvars = continuous_vars(model)
    
    mapping = IASpaceMap(dvars)
    
    logp = logp_graph(model)    
    f = function(model.vars, [grad(logp, var) for var in dvars],
                 allow_input_downcast = True, mode = mode)
    def fn(state):
        return mapping.project(f(**state))
    return fn
    
def model_logp_dlogp(model, dvars = None, mode = None ):    
    if dvars is None :
        dvars = continuous_vars(model)
    
    mapping = IASpaceMap(dvars)
    
    logp = logp_graph(model)
    calculations = [logp] + [grad(logp, var) for var in dvars]
        
    f = function(model.vars, calculations, allow_input_downcast = True, mode = mode)
    def fn(state):
        r = f(**state)
        return r[0], mapping.project(r[1:])
    return fn
        
    
def logp_graph(model):
    return buitin_sum((sum(factor) for factor in model.factors))
    

    
class DASpaceMap(object):
    """ encapsulates a mapping of 
        dict space <-> array space"""
    def __init__(self,free_vars):
        self.dimensions = 0
        
        self.slices = {}
        self.vars = {}
        for var in free_vars:       
            self.vars[str(var)] = var 
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.dsize)
            self.dimensions += var.dsize
            
    def project(self, d, a = None):
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
        d = d.copy()
            
        for var, slice in self.slices.iteritems():
            d[var] = np.reshape(a[slice], self.vars[var].dshape)
            
        return d
    
    
class IASpaceMap(object):
    """ encapsulates a mapping of 
        iterable space -> array space"""
    def __init__(self,free_vars):
        self.dimensions = 0
        
        self.slices = []
        for var in free_vars:       
            self.slices.append(slice(self.dimensions, self.dimensions + var.dsize))
            self.dimensions += var.dsize
            
    def project(self, d , a = None):
        if a is None:
            a = np.empty(self.dimensions)
        else:
            a = np.copy(a)
        
        for slc, v in zip(self.slices, d):    
            a[slc] = np.ravel(v)
        return a
            
def sample(draws, step_method, chain_state, sample_history ):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""
    start = time.time()
    for i in xrange(int(draws)):
        chain_state = step_method.step(chain_state)
        sample_history.record(chain_state, step_method)
        
    return (time.time() - start)


def sample1(draws, step_method, chain_state, sample_history ):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""
    start = time.time()
    for i in xrange(int(draws)):
        step_method.step(chain_state, step_state)
        
        sample_history.record(chain_state)
        
    return (time.time() - start)


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
