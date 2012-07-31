'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from quadpotential import *
from utils import *
from ..core import * 
s

def hmc_step(model, vars, C, step_size_scaling = .25, trajectory_length = 2., is_cov = False, elow = .85, ehigh = 1.15, a = .1):
    """
    is_cov : treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
    """
    n = C.shape[0]
    
    logp_dict = model_logp(model)
    dlogp_dict = model_dlogp(model, vars)
    
    step_size = step_size_scaling / n**(1/4.)
    
    pot = quad_potential(C, is_cov)

    def step(state, q0, logp, dlogp):
        
        if state is None:
            state = SamplerHist()
            
        #randomize step size
        e = uniform(elow, ehigh) * step_size
        nstep = int(floor(trajectory_length / step_size))
        
        p0 = state.p
        p0 = a* p0 + (1-a**2)**.5* pot.random()
        
        #use the leapfrog method
        def leap(q, p):
            p = p - (e/2) * -dlogp(q) # half momentum update
            
            for i in range(nstep): 
                #alternate full variable and momentum updates
                q = q + e * pot.velocity(p)
                if i != nstep - 1:
                    p = p - e * -dlogp(q)
                 
            p = p - (e/2) * -dlogp(q)  # do a half step momentum update to finish off
            return q, p
        
        def H(q, p):
            return -logp(q) + pot.energy(p)
        H0 = H(q0, p0)
        def dH(q, p):
            return H(q, p) - H0
        
        def metrop(H):
            return np.minimum(1, np.exp(H))
        
        q, p = leap(q0, p0)
        pleap = metrop(dH(q, p))
        
        r = uniform()
        if r < pleap:
            pass
        else:
            q, p = leap(-p0, q0)
            
            pflip = np.maximum(0, metrop(dH(q, p)) - pleap)
            if r < pflip:
                pass
            else: 
                q, p = q0, p0
                
        state.p = p
        return state, q
                
        
    return array_step(step, vars, [logp_dict, dlogp_dict])
        