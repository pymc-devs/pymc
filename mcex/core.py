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
from utils import combinations_with_replacement

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
        self.values_considered = self.values.copy()
    

def unique_derivative_variable_sets(n, variables):
    return [frozenset(varset) for varset in itertools.combinations_with_replacement(variables,n)]

class Evaluation(object):
    """
    encapsulates the action of evaluating the state using the model.
    """
    def __init__(self, model, derivative_vars = [], nderivative = 0):
        self.nderivative = nderivative 
        
        logp_calculation = bsum((sum(logp) for logp in model.logps))
        
        calculations = [logp_calculation]
        variable_names = [str(var) for var in derivative_vars]
        
        self.derivative_orders = [unique_derivative_variable_set(n,variable_names) for n in range(nderivative +1)]
        
        calculations = []
        
        for orders in self.derivative_orders:
            for varset in orders:
                calc = logp_calculation
                for var in varset:
                    calc = grad(calc, var)
                calculations.append(calc)
            
        self.function = function(model.free_vars, calculations)
        
    def _evaluate(self,d_repr, chain_state):
        """
        returns logp, derivative1, derivative2...
        does not currently do beyond derivative1
        """
        results = iter(self.function(**chain_state.values_considered))
        return tuple(d_repr(n, self.derivative_orders[i],islice(results, n + 1)) for n in range(self.nderivative + 1))
    
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
            mapping.iapply(ordering,it)
        return self._evaluate(vector_representation, chain_state)
        
class VariableMapping(object):
    """encapsulates a mapping between a subset set of variables and a vector
    may in the future 
    """
    def __init__(self,free_vars, max_n = 3):
        self.dimensions = 0
        
        slices = {}
        variable_names = [str(var) for var in derivative_vars]
        var_index = dict((free_vars[i], i) for i in range(len(freevars)))
        
        for var in free_vars:        
            self.slices[str(var)] = slice(self.dimensions, self.dimensions + var.size)
            self.dimensions += var.size
            
        self.slice_tuples = {}
        for n in range(nderivative +1): 
            for varset in unique_derivative_variable_set(n,variable_names):
                self.slice_tuples[varset] = tuple(slices[var] for var in sorted(varset, key = lambda var: var_index[var]))
            
    
    def apply_to_dict(self, values):
        return self.apply( values.iteritems())

    def apply(self,varsets_values, n = 1):
        #breaks for n > 1
        vector = np.empty((self.dimensions,)*n)
        
        for varset, value in varset_values
            try:    
                if instanceof(varset, basestr):
                    varset = frozenset((varset,))
                vector[self.slice_tuples[varset]] = np.ravel(value)#breaks for n > 1
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
        