'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor, concatenate, exp, zeros, diag, real
from numpy.linalg import inv, cholesky, eig


from utils import *
from ..core import * 
    
def split_hmc_step(model, vars, C, approx_loc, approx_C, step_size_scaling = .25, trajectory_length = 2. ):
    
    mapp = DASpaceMap( vars)
    approx_loc = mapp.project(approx_loc)
    
    n = C.shape[0]
    
    logp_d_dict = model_logp_dlogp(model, vars)
    
    step_size = step_size_scaling * n**(1/4.)
    
    invC = inv(C)
    cholInvC = cholesky(invC)
    
    A = zeros((2*n,2*n))
    A[:n,n:] = C
    A[n:,:n] = -approx_C
    
    D, gamma = eig(A)
        
    e = step_size
    R = real(gamma.dot(diag(exp(D* e))).dot(gamma.T))
    def step(logp_d, state, q0):
        
        if state is None: 
            state = SamplerHist()

        nstep = int(floor(trajectory_length / step_size))
        
        q = q0
        logp0, dlogp = logp_d(q)
        dlogp = dlogp + dot(approx_C, q - approx_loc)
        logp = logp0
            
        # momentum scale proportional to inverse of parameter scale (basically sqrt(covariance))
        p = p0 = cholesky_normal( q.shape, cholInvC)
        
        #use the leapfrog method
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            p = p - (e/2) * -dlogp # half momentum update
            
            x = concatenate((q - approx_loc, p))
            x = dot(R, x)
            q = x[:n] + approx_loc
             
            logp, dlogp = logp_d(q)
            dlogp = dlogp + dot(approx_C, q - approx_loc)
            
            p = p - (e/2) * -dlogp  # do a half step momentum update to finish off
             
        
        p = -p 
            
        mr = logp - logp0 + K(C, p0) - K(C, p)
        state.metrops.append(mr)
        
        return state, metrop_select(mr, q, q0)
        
    return array_step(step, logp_d_dict, vars)
                
    
def K (cov, x):
    return .5 * dot(x,dot(cov, x))


