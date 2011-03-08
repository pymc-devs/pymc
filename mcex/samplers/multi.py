'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np 

class MultiStep(object):
        
    def init(self, model):
        self.model = model
        self.slices, self.dimensions = flatten_vars(model.free_vars)

def values_to_vector(values, slices, dimensions):
    vector = np.empty(dimensions)
    
    for var, value in values.iteritems():
        vector[slices[var]] = np.ravel(value)
        
    return vector 

def vector_to_values(vector, slices, dimensions):
    values = {}
    for var, slice in slices.iteritems():
        values[var] = np.reshape(vector[slice], var.shape)
        
    return vector 

def flatten_vars(free_vars):
    """Compute the dimension of the sampling space and identify the slices
    belonging to each stochastic.
    """
    dimensions = 0
    slices = {}
    
    for var in free_vars:        
        slices[var] = slice(dimensions, dimensions + var.size)
        dimensions += var.size
        
    return slices, dimensions