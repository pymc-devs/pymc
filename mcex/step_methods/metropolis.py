'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np

class MetropolisStep(object):
    """Hamiltonian step method"""
    def __init__(self,model,covariance, scaling = .25, ):
        self.model = model
        self.zero = np.zeros(self.model.mapping.dimensions)
        self.covariance = covariance
        self.scaling = scaling
        
    def step(self, chain_state):

        delta = np.random.multivariate_normal(mean = self.zero ,cov = self.covariance * self.scaling) 
        
        current_state = self.model.subspace(chain_state)
        proposed_state = self.model.project(chain_state, current_state + delta)
        
        current_logp = self.model.evaluate_as_vector(current_state)
        proposed_logp = self.model.evaluate_as_vector(proposed_state)

        log_metrop_ratio = (proposed_logp) - (current_logp) 
        
        if (np.isfinite(log_metrop_ratio) and 
            np.log(np.random.uniform()) < log_metrop_ratio):
            
            return proposed_state
        else: 
            return chain_state
