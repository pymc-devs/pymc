'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from quadpotential import *
from utils import *
from ..core import * 

def hmc_lowflip_step(model, vars, C,is_cov = False, step_size = .1, nstep = 1,  a = .9):
    """
    is_cov : treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
    """ 
    logp_dict = model_logp(model)
    dlogp_dict = model_dlogp(model, vars)
    
    e = step_size
    
    pot = quad_potential(C, is_cov)

    def step(state, q0, logp, dlogp):
        
        if state is None:
            state = SamplerHist()
            state.p = pot.random()
            
        p0 = state.p
        p0 = a* p0 + (1-a**2)**.5* pot.random()
        
        def leap(q, p):
            p = p - (e/2) * -dlogp(q) # half momentum update
            
            for i in range(nstep): 
                q = q + e * pot.velocity(p)
                if i != nstep - 1:
                    p = p - e * -dlogp(q)
                 
            p = p - (e/2) * -dlogp(q)  
            return q, p
        
        def H(q, p):
            return -logp(q) + pot.energy(p)
        
        H0 = H(q0, p0)
        
        def metrop(q, p):
            return np.minimum(1, np.exp(H(q, p) - H0))
        
        q, p = leap(q0, p0)
        pleap = metrop(q, p)
        
        r = uniform()
        if r > pleap:
            q, p = leap(-p0, q0)
            
            pflip = np.maximum(0, metrop(q, p) - pleap)
            
            if r > pflip:
                q, p = q0, p0
                
        state.p = p
        return state, q
                
        
    return array_step(step, vars, [logp_dict, dlogp_dict])
        