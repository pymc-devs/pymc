'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from numpy.linalg import solve
from scipy.linalg import cholesky, cho_solve


from utils import *
from ..core import * 
from lbfgs import *
from __builtin__ import sum as bsum

# todo : 
#make step method use separate gradient and logp functions
#add constraint handling via page 37 of Radford's http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html
#allow users to pass Hamiltonian splitting functions

def vlbfgs_hmc_step(model, vars, approx_n, step_size_scaling = .25, trajectory_length = 2., is_cov = False):
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
        v = v0 = state.hess.C.dot(z)
        #use the leapfrog method
        v = v - (e/2) * -state.hess.Bdot(dlogp(q)) # half momentum update
        
        for i in range(nstep): 
            #alternate full variable and momentum updates
            q = q + e * v
            if i != nstep - 1:
                v = v - e * -state.hess.Bdot(dlogp(q))
             
        v = v - (e/2) * -state.hess.Bdot(dlogp(q))  # do a half step momentum update to finish off
        
        v = -v
            
        def energy(d):
            return .5 * dot(d, d)
        
        
        mr = (-logp(q0)) + energy(v0) - ((-logp(q))  + energy(v))
        q = metrop_select(mr, q, q0)
       
        state.hess = state.hessgen.update(q,-logp(q), -dlogp(q))
        state.metrops.append(mr)
        
        return state, q
        
    return array_step(step, vars, [logp_dict, dlogp_dict])