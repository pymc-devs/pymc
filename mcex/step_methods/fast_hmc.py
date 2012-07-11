'''
Created on Mar 7, 2011

@author: johnsalvatier
''' 
from numpy import floor
from numpy.linalg import solve
from scipy.linalg import cholesky, cho_solve
import random
from __builtin__ import sum as sumb

from utils import *
from ..core import * 

def fast_hmc_step(model, vars, m, step_size_scaling = .25, trajectory_length = 2.):
    n = sumb(v.dsize for v in vars)
    
    logp_d_dict = model_logp_dlogp(model, vars)
    
    step_size = step_size_scaling * n**(1/4.)
    def step(logp_d, state, q0):
        
        if state is None:
            state = SamplerHist()
            state.previous = []
            state. i = 0 
            state.q_tmp = None
            
        e = uniform(.85, 1.15)  * step_size
        nstep = int(floor(trajectory_length / step_size)) 
        
        # draw a random sample
        
        v = v0 = (1e-5) **.5 *np.random.normal(size =n)
        
        f = np.minimum(m, len(state.previous))
        directions = random.sample(state.previous, f)
        
        if len(directions):
            u = np.mean(directions, 0)
            directions = [x - u for x in directions]
            
    
            for x in directions:
                v = v + x*np.random.normal()
        
        #if len(directions) > 2:        
        #    c = np.cov(np.array(state.previous[1:]).T) + 1e-5 * np.eye(n)
        
        
        
        q = q0
        logp0, dlogp = logp_d(q)
        logp = logp0
        
        
        
        #use the leapfrog method
        def Cdot(g):
            Xd = np.ones_like(g) * 1e-5
            ps = [dot(d, g) for d in directions]
            for d, p in zip(directions, ps):
                Xd = Xd + p*d
            return Xd
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



def quad_potential(C, is_cov):
    if is_cov:
        return QuadPotential(C)
    else :
        return QuadPotential_Inv(C) 

class QuadPotential_Inv(object):
    def __init__(self, A):
        self.L = cholesky(A, lower = True)
        
    def velocity(self, x ):
        return cho_solve((self.L, True), x)
        
    def random(self):
        n = normal(size = self.L.shape[0])
        return dot(self.L, n)
    
    def energy(self, x):
        L1x = solve(self.L, x)
        return .5 * dot(L1x.T, L1x)


class QuadPotential(object):
    def __init__(self, A):
        self.A = A
        self.L = cholesky(A, lower = True)
    
    def velocity(self, x):
        return dot(self.A, x)
    
    def random(self):
        n = normal(size = self.L.shape[0])
        return solve(self.L.T, n)
    
    def energy(self, x):
        return .5 * dot(x, dot(self.A, x))
        