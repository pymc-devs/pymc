'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from utils import *
from ..core import * 
from lbfgs import *
from __builtin__ import sum as bsum

# todo : 
#make step method use separate gradient and logp functions
#add constraint handling via page 37 of Radford's http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html
#allow users to pass Hamiltonian splitting functions

def lbfgs_hmc_step(model, vars, approx_n, step_size_scaling = .25, trajectory_length = 2., is_cov = False):
    n = bsum(v.dsize for v in vars)
    
    logp_dict = model_logp(model)
    dlogp_dict = model_dlogp(model, vars)
    
    step_size = step_size_scaling / n**(1/4.)
    
    def step(state, q0, logp, dlogp):
        
        if state is None:
            state = SamplerHist()
            state.hessgen = HessApproxGen(approx_n)
            state.hess = LBFGS( 1e-8)
            
        #randomize step size
        e = uniform(.85, 1.15) * step_size
        nstep = int(floor(trajectory_length / step_size))
        
        q = q0 
        z = np.random.normal(size = n)
        p = p0 = state.hess.C.dot(z)
        #use the leapfrog method
        p = p - (e/2) * dlogp(q) # half momentum update
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + -e * state.hess.Hdot(p)
            if i != nstep - 1:
                p = p - e *dlogp(q)
             
        p = p - (e/2) * dlogp(q)  # do a half step momentum update to finish off
        
        p = -p 
            
        def energy(d):
            Sd = state.hess.S.Tdot(d)
            return .5 * dot(Sd, Sd)
        
        
        mr = (-logp(q0)) + energy(p0) - ((-logp(q))  + energy(p))
        q = metrop_select(mr, q, q0)
       
        state.hess = state.hessgen.update(q,-logp(q), -dlogp(q))
        state.metrops.append(mr)
        
        return state, q
        
    return array_step(step, vars, [logp_dict, dlogp_dict])