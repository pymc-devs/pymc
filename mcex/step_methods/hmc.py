'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np
from ..core import *

class HMCStep(object):
    """Hamiltonian step method"""
    def __init__(self,model, vars, covariance, step_size_scaling = .25, trajectory_length = 2. , debug = False):
        self.mapping = DASpaceMap(vars)
        self.logp_d = model_logp_dlogp(model, vars)
        
        self.zero = np.zeros(self.mapping.dimensions)
        
        self.covariance = covariance
        self.inv_covariance = np.linalg.inv(covariance)
        
        step_size = step_size_scaling * self.mapping.dimensions**(1/4.)
      
        if np.size(step_size) > 1:
            self.step_size_max, self.step_size_min = step_size
        else :
            self.step_size_max = self.step_size_min = step_size 
        self.trajectory_length = trajectory_length   
        
        self.debug = debug
        self.d = 5
        
    def step(self, chain_state):
        #randomize step size
        step_size = np.random.uniform(self.step_size_min, self.step_size_max)
        step_count = int(np.floor(self.trajectory_length / step_size))
        
        
        q = self.mapping.project(chain_state)
        start_logp, gradient = self.logp_d(chain_state)
        
        current_logp = start_logp
        
        # momentum scale proportional to inverse of parameter scale (basically sqrt(covariance))
        p = np.random.multivariate_normal(mean = self.zero ,cov = self.inv_covariance) 
        start_p = p
        
        #use the leapfrog method
        p = p - (step_size/2) * -gradient # half momentum update
        
        for i in range(step_count): 
            #alternate full variable and momentum updates
            q = q + step_size * np.dot(self.covariance, p)
            
            proposed_state = self.mapping.rproject(q, chain_state)
            
            current_logp, gradient = self.logp_d(proposed_state)
            
            if i != step_count - 1:
                p = p - step_size * -gradient
             
        p = p - (step_size/2) * -gradient  # do a half step momentum update to finish off
        
        p = -p 
            
        log_metrop_ratio = (-start_logp) - (-current_logp) + self.kenergy(start_p) - self.kenergy(p)
        
        
        
        self.acceptr = np.minimum(np.exp(log_metrop_ratio), 1.)
        
        if self.debug: 
            print self.acceptr, log_metrop_ratio, start_logp, current_logp, start_p, p, self.kenergy(start_p), self.kenergy(p)
            if not np.isfinite(self.acceptr) :
                if self.d < 0:
                    print self.logp_d(proposed_state)
                    raise ValueError
                else:
                    self.d -= 1
        
        if (np.isfinite(log_metrop_ratio) and 
            np.log(np.random.uniform()) < log_metrop_ratio):
            
            return proposed_state
        else: 
            return chain_state
                
    
    def kenergy (self, x):
        return .5 * np.dot(x,np.dot(self.covariance, x))
