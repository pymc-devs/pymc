'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np
from ..core import *

class MetropolisStep(object):
    """Hamiltonian step method"""
    def __init__(self,model, vars, covariance, scaling = .25, ):
        self.mapping = DASpaceMap(vars)
        self.logp = model_logp(model, vars)
        
        self.zero = np.zeros(self.model.mapping.dimensions)
        self.covariance = covariance * scaling
        
    def step(self, chain_state):

        delta = np.random.multivariate_normal(mean = self.zero ,cov = self.covariance) 
        
        current_state = self.mapping.project(chain_state)
        proposed_state = current_state + delta
        
        current_logp = self.logp(chain_state)
        proposed_logp = self.logp(self.mapping.rproject(proposed_state, chain_state))

        log_metrop_ratio = (proposed_logp) - (current_logp) 
        
        if (np.isfinite(log_metrop_ratio) and 
            np.log(np.random.uniform()) < log_metrop_ratio):
            
            return proposed_state
        else: 
            return chain_state
