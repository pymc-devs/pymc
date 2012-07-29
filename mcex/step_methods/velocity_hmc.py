'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
import random
from __builtin__ import sum as sumb

from quadpotential import *

from utils import *
from ..core import * 

def velocity_hmc_step(model, vars, m, step_size_scaling = .25, trajectory_length = 2.):
    n = sumb(v.dsize for v in vars)
    
    logp_d_dict = model_logp_dlogp(model, vars)
    
    step_size = step_size_scaling / n**(1/4.)
    def step(logp_d, state, q0):
        
        if state is None:
            state = SamplerHist()
            state.previous = []
            state. i = 0 
            state.q_tmp = None
            
        e = uniform(.85, 1.15)  * step_size
        nstep = int(floor(trajectory_length / step_size)) 
        
        # draw a random sample
        
        f = np.minimum(m, len(state.previous))
        directions = random.sample(state.previous, f)
        
        C = 1e-6 *np.eye(n)
        if len(directions) > 3:
            C = np.cov(np.array(directions).T) + C
        
        v = v0 =  np.dot(np.linalg.cholesky(C), np.random.normal(size =n))
        
        
        
        q = q0
        logp0, dlogp = logp_d(q)
        logp = logp0
        
        
        
        #use the leapfrog method
        def Cdot(g):
            return np.dot(C, g)
        v = v - (e/2) * Cdot(-dlogp) 
        
        
                    
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + e * v
             
            logp, dlogp = logp_d(q)
            
            if i != nstep - 1:
                v = v - e * Cdot(-dlogp)
             
        v = v - (e/2) * Cdot(-dlogp)  # do a half step momentum update to finish off
        
        v = -v 
            
        # - H(q*, p*) + H(q, p) = -H(q, p) + H(q0, p0) = -(- logp(q) + K(p)) + (-logp(q0) + K(p0))
        mr = (-logp0) + .5*dot(v0, Cdot(v0)) - ((-logp)  + .5*dot(v,Cdot(v)))
        q = metrop_select(mr, q, q0)
        
        state.metrops.append(mr)
        
        if state.q_tmp is not None:
            state.previous.append(state.q_tmp)
        
        state.q_tmp = q 
        state.i += 1
        
        if len(state.previous)> state.i/2 : 
            state.previous.pop(0)

        return state, q
        
    return array_step(step, logp_d_dict, vars)

