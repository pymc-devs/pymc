'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from theano.tensor import *
from theano import function
import numpy as np 
from __builtin__ import sum as bsum

class FreeVariable(TensorVariable ):
    def __init__(self, name, shape, dtype):
        
        ttype = TensorType(str(dtype), np.array(shape) == 1)
        TensorVariable.__init__(self, name, ttype)
        self.dshape = shape
        self.dsize = np.prod(shape)

class SampleHistory(object):
    def __init__(self, model, max_draws):
        self.max_draws = max_draws
        samples = {}
        for var in model.free_vars: 
            samples[var] = np.empty((max_draws,) + var.dshape)
            
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
    def __init__(self, free_vars, logps, derivative_vars = []):

        self.free_vars = free_vars
        self.logps = logps
        self.derivative_vars = derivative_vars 
        
        self.eval = Evaluation(self, derivative_vars)
        
    submodels = {}
    def submodel(self, logps = [], derivative_vars = []):
        if not logps :
            logps = self.logps
        
        key = tuple(tuple(logps), tuple(derivative_vars))
        try :
            return self.submodels[key]
        except:
            self.submodels[key] = Model(self, self.free_vars, logps, derivative_vars)
            return self.submodels[key]


class ChainState(object):
    
    """
    Encapsulates the state of the chain
    """
    def __init__(self, values):
        
        self.values = values
        self.reject()
    
    def accept(self):
        self.values = self.values_considered
        
    def reject(self):
        self.values_considered = self.values.copy()
    
class Evaluation(object):
    """
    encapsulates the action of evaluating the state using the model.
    """
    def __init__(self, model, derivative_vars = []):
        self.derivative = len(derivative_vars) > 0 
        
        logp_calculation = bsum((sum(logp) for logp in model.logps))
        
        self.derivative_order = [str(var) for var in derivative_vars]
        
        calculations = [logp_calculation] + [grad(logp_calculation, var) for var in derivative_vars]
            
        self.function = function(model.free_vars, calculations)
        
    def _evaluate(self,d_repr, chain_state):
        """
        returns logp, derivative1, derivative2...
        does not currently do beyond derivative1
        """
        results = iter(self.function(**chain_state.values_considered))
        if self.derivative:
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
         
    def evaluate_as_vector(self, mapping, chain_state):   
        def vector_representation(n, ordering, it):
            mapping.apply(zip(ordering,it))
        return self._evaluate(vector_representation, chain_state)
        
class VariableMapping(object):
    """encapsulates a mapping between a subset set of variables and a vector
    may in the future 
    """
    def __init__(self,free_vars):
        self.dimensions = 0
        
        self.slices = {}
        for var in free_vars:        
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.dsize)
            self.dimensions += var.size
            
    
    def apply_to_dict(self, values):
        return self.apply( values.iteritems())

    def apply(self,varset_values):
        vector = np.empty(self.dimensions)
        
        for var, value in varset_values:
            try:    
                vector[self.slices[var]] = np.ravel(value)
            except KeyError:
                pass
                
        return vector
    
     
    def update_with_inverse(self,values, vector):
        for var, slice in self.slices.iteritems():
            values[var] = np.reshape(vector[slice], var.shape)
            
        return values 

def sample(draws, sampler,chain_state = None, sample_history = None):
    if sample_history is None:
        sample_history = SampleHistory(sampler.model, draws)
    
    for i in xrange(draws):
        step_method.step(chain_state)
        sample_history.record(chain_state)
        
    return sample_history
        