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
    
    
    
    
    
    
from numpy.random import uniform, normal
from numpy import floor, dot, log , isfinite
from numpy.linalg import inv, cholesky
    
    
    
def hmc_step(logp_d,n, C, step_size_scaling = .25, trajectory_length = 2. , debug = False):
    step_size = step_size_scaling * n**(1/4.)
    
    cholInvC = cholesky(inv(C))
    
    def step(self, q0):
        #randomize step size
        e = uniform(.85, 1.15) * step_size
        nstep = int(floor(trajectory_length / step_size))
        
        q = q0
        logp0, gradient = logp_d(q)
        logp = logp0
        
        # momentum scale proportional to inverse of parameter scale (basically sqrt(covariance))
        p = p0 = dot(normal(size = q.shape), cholInvC)
        
        #use the leapfrog method
        p = p - (e/2) * -gradient # half momentum update
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + e * dot(C, p)
             
            logp, dlogp = logp_d(q)
            
            if i != nstep - 1:
                p = p - e * -dlogp
             
        p = p - (e/2) * -dlogp  # do a half step momentum update to finish off
        
        p = -p 
            
        log_metrop_ratio = (-logp0) - (-logp) + K(C, p0) - K(C, p)
        
        
        if (isfinite(log_metrop_ratio) and 
            log(uniform()) < log_metrop_ratio):
            
            return q
        else: 
            return q0
                
    
def K (cov, x):
    return .5 * dot(x,dot(cov, x))
