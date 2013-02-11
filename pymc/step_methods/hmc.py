'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from quadpotential import *
from utils import *
from ..core import * 


#TODO: 
#add constraint handling via page 37 of Radford's http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html

def unif(step_size, elow = .85, ehigh = 1.15):
    return uniform(elow, ehigh) * step_size

class hmc_step(array_step):
    def __init__(self, model, vars, C, step_size_scaling = .25, trajectory_length = 2., is_cov = False, step_rand = unif):
        """
        is_cov : treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
        """
        n = C.shape[0]
        
        self.step_size = step_size_scaling * n**(1/4.)
        
        self.pot = quad_potential(C, is_cov)
        self.trajectory_length = trajectory_length
        self.step_rand = step_rand

        super(hmc_step, self).__init__(vars, [model.logp(), model.dlogp(vars)] )

    def astep(self, state, q0, logp, dlogp):
        
        if state is None:
            state = SamplerHist()
            
        #randomize step size
        e = self.step_rand(self.step_size) 
        nstep = int(floor(self.trajectory_length / self.step_size))
        
        q = q0 
        p = p0 = self.pot.random()
        
        #use the leapfrog method
        p = p - (e/2) * -dlogp(q) # half momentum update
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + e * self.pot.velocity(p)
            if i != nstep - 1:
                p = p - e * -dlogp(q)
             
        p = p - (e/2) * -dlogp(q)  # do a half step momentum update to finish off
        
        p = -p 
            
        # - H(q*, p*) + H(q, p) = -H(q, p) + H(q0, p0) = -(- logp(q) + K(p)) + (-logp(q0) + K(p0))
        mr = (-logp(q0)) + self.pot.energy(p0) - ((-logp(q))  + self.pot.energy(p))
        state.metrops.append(mr)
        
        return state, metrop_select(mr, q, q0)
        
