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

def FreeVariable( name, shape, dtype):
    """creates a TensorVariable of the given shape and type"""
    shape = np.atleast_1d(shape)
    var = TensorType(str(dtype), shape == 1)(name)
    var.dshape = tuple(shape)
    var.dsize = int(np.prod(shape))
    return var

class ModelView(object):
    """
    encapsulates the probability model
    """
    def __init__(self, free_vars, logps, derivative_vars = []):

        self.free_vars = free_vars
        self.logps = logps
        
        if derivative_vars == 'all' :
            derivative_vars = [ var for var in free_vars if var.dtype in continuous_types]
            
        self.derivative_vars = derivative_vars 
        
        
        logp_calculation = buitin_sum((sum(logp) for logp in logps))
        
        self.mapping = VariableMapping(derivative_vars)
        self.derivative_order = [str(var) for var in derivative_vars]
        
        
        calculations = [logp_calculation] + [grad(logp_calculation, var) for var in derivative_vars]
            
        self.function = function(free_vars, calculations, allow_input_downcast = True)
        
        
    def _evaluate(self,d_repr, chain_state):
        """
        returns logp, derivative1, derivative2...
        does not currently do beyond derivative1
        """
        results = iter(self.function(**chain_state))
        if self.derivative_vars:
            return results.next(), d_repr(1, self.derivative_order, results)
        else :
            return results.next()
        
    def evaluate(self, chain_state):
        def dict_representation(n, ordering, it):
            #replaceable with dict comprehensions
            values = {}
            for key, value in zip(ordering, it):
                values[key] = value
            return values

        return self._evaluate(dict_representation, chain_state)
    
    def evaluate_as_vector(self, chain_state):
        """
        perhaps needs to be moved out of Evaluate
        """   
        def vector_representation(n, ordering, it):
            return self.mapping.subspace(zip(ordering,it))
        return self._evaluate(vector_representation, chain_state)
        
    def project(self, chain_state, vector):
        return self.mapping.project(chain_state, vector)
    def subspace(self, chain_state):
        return self.mapping.dict_subspace(chain_state)
    
class VariableMapping(object):
    """encapsulates a subset of variables represented as a vector
    each vector defines a subspace of the whole parameter space 
    """
    def __init__(self,free_vars):
        self.dimensions = 0
        
        self.slices = {}
        self.vars = {}
        for var in free_vars:       
            self.vars[str(var)] = var 
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.dsize)
            self.dimensions += var.dsize
            
    def dict_subspace(self, values):
        """gets the subspace given a dict
        """
        return self.subspace( values.iteritems())

    def subspace(self,varset_values):
        """
        returns a vector that defines a subspace given an iterable of (variablename, value)
        """
        vector = np.empty(self.dimensions)
        
        for varname, value in varset_values:
            try:    
                vector[self.slices[varname]] = np.ravel(value)
            except KeyError:
                pass
                
        return vector
    
     
    def project(self,values, vector):
        """
        projects the values onto the subspace defined by the vector
        """
        values = values.copy()
        for var, slice in self.slices.iteritems():
            values[var] = np.reshape(vector[slice], self.vars[var].dshape)
            
        return values 

def sample(draws, step_method, chain_state, sample_history ):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""
    start = time.time()
    for i in xrange(int(draws)):
        chain_state = step_method.step(chain_state)
        sample_history.record(chain_state, step_method)
        
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
