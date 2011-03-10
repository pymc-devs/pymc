'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import *
from theano import function, grad 
from numpy import pi, inf
import numpy as np 
from __builtin__ import sum as bsum

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
    def __init__(self, free_vars, logps, gradient_vars):

        self.free_vars = free_vars

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
        
    def evaluate(self, chain_state, dimensions = None, slices = None):
        """
        returns logp, derivative1, derivative2...
        does not currently do beyond derivative1
        """
        return self._package_results(self._evaluate(**chain_state.values_considered), 
                                     dimensions, 
                                     slices)
        
    def _package_results(self,results, dimensions, slices):    
        
        if dimensions is None and slices is None:
            #need N dimensional symmetric dictionary here for derivatives > 1 
            if self.nderivative == 1: 
                derivatives = {}
                for var_name, derivative in zip(self.derivative_order,results[1:]):
                    derivatives[var_name] = derivative
                return results[0], derivatives
            return results[0]
        else :
            pass
        
class VariableMapping(object):
    """encapsulates a mapping between a a set of variables and a vector
    """
    def __init__(self,free_vars):
        self.dimensions = 0
        self.slices = {}
        
        for var in free_vars:        
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.size)
            self.dimensions += var.size
            
    def apply(self,values):
        vector = np.empty(self.dimensions)
        
        for var, value in values.iteritems():
            vector[self.slices[var]] = np.ravel(value)
            
        return vector 
    def apply_inverse(self, vector):
        values = {}
        for var, slice in self.slices.iteritems():
            values[var] = np.reshape(vector[slice], var.shape)
            
        return values 

def flatten_vars

def sample(model, draws,step_method, sample_history = None):

    step_method.set_model(model)
    
    if sample_history is None:
        sample_history = SampleHistory(model, draws)
    
    for i in xrange(draws):
        step_method.step()
        sample_history.record(step_method.chain_state)
        
    return sample_history
        