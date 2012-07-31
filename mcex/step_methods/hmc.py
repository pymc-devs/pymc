'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from quadpotential import *
from utils import *
from ..core import * 


# todo : 
#add constraint handling via page 37 of Radford's http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html
#allow users to pass Hamiltonian splitting functions

def hmc_step(model, vars, C, step_size_scaling = .25, trajectory_length = 2., is_cov = False, elow = .85, ehigh = 1.15):
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
        
        q = q0 
        p = p0 = pot.random()
        
        #use the leapfrog method
        p = p - (e/2) * -dlogp(q) # half momentum update
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + e * pot.velocity(p)
            if i != nstep - 1:
                p = p - e * -dlogp(q)
             
        p = p - (e/2) * -dlogp(q)  # do a half step momentum update to finish off
        
        p = -p 
            
        # - H(q*, p*) + H(q, p) = -H(q, p) + H(q0, p0) = -(- logp(q) + K(p)) + (-logp(q0) + K(p0))
        mr = (-logp(q0)) + pot.energy(p0) - ((-logp(q))  + pot.energy(p))
        state.metrops.append(mr)
        
        return state, metrop_select(mr, q, q0)
        
    return array_step(step, vars, [logp_dict, dlogp_dict])
        