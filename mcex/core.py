'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import *
from theano import function, grad 
from numpy import pi, inf
import numpy as np 
from __builtin__ import sum as bsum
import itertools 

class FreeVariable(TensorVariable ):
    def __init__(self, name, shape, dtype):
        
        ttype = TensorType(str(dtype), np.array(shape) == 1)
        TensorVariable.__init__(self, name, ttype)
        self.shape = shape
        self.size = np.prod(shape)

class SampleHistory(object):
    def __init__(self, model, max_draws):
        self.max_draws = max_draws
        samples = {}
        for var in model.free_vars: 
            samples[var] = np.empty((max_draws,) + var.shape)
            
        self._samples = samples
        self.nsamples = 0
    
    def record(self, samples):
        if self.nsamples < self.max_draws:
            for var, sample in self._samples.iteritems():
                self._samples[var][self.nsamples,...] = samples[var]
            self.nsamples += 1
        else :
            raise ValueError('out of space!')
        
    def __getitem__(self, key):
        return self._samples[key][0:self.nsamples,...]

class Model(object):
    def __init__(self, free_vars, logps):

        self.free_vars = free_vars
        self.logps = logps

class ChainState(object):
    """
    Encapsulates the state of the chain
    """
    def __init__(self):
        
        self.values = {}
        self.values_considered = {}
        
    def accept(self):
        self.values = self.values_considered
        
    def reject(self):
        self._consideration_state = self._state.copy()
    
class Evaluation(object):
    """
    encapsulates the action of evaluating the state using the model.
    """
    def __init__(self, model, derivative_vars = [], nderivative = 0):
        self.nderivative = nderivative 
        
        logp_calculation = bsum((sum(logp) for logp in model.logps))
        
        calculations = [logp_calculation]
        
        if nderivative == 1:
            calculations += [grad(logp_calculation, free_var) for free_var in derivative_vars]
            self.derivative_order = [ str(var) for var in derivative_vars]
            
        
        
        self.evaluate = function(model.free_vars, calculations)
        
    def evaluate(self, chain_state):
        """
        returns logp, derivative1, derivative2...
        does not currently do beyond derivative1
        """
        results = iter(self._evaluate(**chain_state.values_considered))
        return itertools.chain((next(results),) (dict_group(order,results) for order in self.derivative_orders))
        

def dict_group(order, it):
    #replaceable with dict comprehensions
    values = {}
    for key in order:
        values[key] = next(it)  
    return values
    
        
class VariableMapping(object):
    """encapsulates a mapping between a subset set of variables and a vector
    """
    def __init__(self,free_vars):
        self.dimensions = 0
        self.slices = {}
        
        for var in free_vars:        
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.size)
            self.dimensions += var.size
    
    def apply_to_dict(self, values):
        return self._apply(values.iteritems())
            
    def _apply(self, var_vals):
        vector = np.empty(self.dimensions)
        
        for var, value in var_vals:
            try:
                vector[self.slices[var]] = np.ravel(value)
            except KeyError:
                pass
                
        return vector
    
     
    def update_with_inverse(self,values, vector):
        for var, slice in self.slices.iteritems():
            values[var] = np.reshape(vector[slice], var.shape)
            
        return values 

def sample(model, draws, model, step_method,chain_state = None, sample_history = None):
    if chain_state is None:
        chain_state = ChainState()
    
    if sample_history is None:
        sample_history = SampleHistory(model, draws)
    
    for i in xrange(draws):
        step_method.step(chain_state)
        sample_history.record(chain_state)
        
    return sample_history
        