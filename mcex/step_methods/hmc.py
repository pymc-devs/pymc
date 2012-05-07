'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy.random import uniform, normal
from numpy import floor, dot, log , isfinite
from numpy.linalg import inv, cholesky

from utils import *
from ..core import * 


# todo : 
#make step method use separate gradient and logp functions

def hmc_step(model, vars, C, step_size_scaling = .25, trajectory_length = 2. ):
    n = C.shape[0]
    
    logp_d_dict = model_logp_dlogp(model, vars)
    
    step_size = step_size_scaling * n**(1/4.)
    
    cholInvC = cholesky(inv(C))
    
    def step(logp_d, state, q0):
        
        #randomize step size
        e = uniform(.85, 1.15) * step_size
        nstep = int(floor(trajectory_length / step_size))
        
        q = q0
        logp0, dlogp = logp_d(q)
        logp = logp0
        
        # momentum scale proportional to inverse of parameter scale (basically sqrt(covariance))
        p = p0 = cholesky_normal( q.shape, cholInvC)
        
        #use the leapfrog method
        p = p - (e/2) * -dlogp # half momentum update
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + e * dot(C, p)
             
            logp, dlogp = logp_d(q)
            
            if i != nstep - 1:
                p = p - e * -dlogp
             
        p = p - (e/2) * -dlogp  # do a half step momentum update to finish off
        
        p = -p 
            
        mr = logp - logp0 + K(C, p0) - K(C, p)
        
        return state, metrop_select(mr, q, q0)
        
    return array_step(step, logp_d_dict, vars)
                
    
def K (cov, x):
    return .5 * dot(x,dot(cov, x))
