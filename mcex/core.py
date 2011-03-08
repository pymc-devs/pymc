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
        logp_calculation = bsum((sum(logp) for logp in logps))
        grad_calculations = [grad(logp_calculation, free_var) for free_var in gradient_vars]
    
        self.evaluate = function(free_vars, [logp_calculation] + grad_calculations)
        self.free_vars = free_vars


def sample(model, draws,step_method, sample_history = None):

    step_method.set_model(model)
    
    if sample_history is None:
        sample_history = SampleHistory(model, draws)
    
    for i in xrange(draws):
        sample_history.record(step_method.step())
        
    return sample_history
        