from PyMC2 import SamplingMethod, OneAtATimeMetropolis
from Realization import Realization
from numpy import reshape, asarray
from numpy.random import normal



class GPMetropolis(OneAtATimeMetropolis):
    """
    Just need to override propose(), I think.
    Everything else should be fine, including
    proposing from the prior if there are no 
    children.
    """
    def __init__(self, f, M, C, scale=.1, dist=None):
        SamplingMethod.__init__(self, pymc_objects=[f])
        self.parameter = f
        self._id = 'GPMetropolis_'+self.parameter.__name__
        self.scale = scale
        
        self.M = M
        self.C = C
        self.length = self.C.value.base_reshape.shape[0]
        
        self.base_mesh = self.C.value.base_mesh

        # If self's extended children is the empty set (eg, if
        # self's parameter is a posterior predictive quantity of
        # interest), proposing from the prior is best.
        if len(self.children) == 0:
            self.step = self.prior_proposal_step
    
    def propose(self):
        # Draw a zero-mean realization from C, with out the Realization __new__ overhead
        dev = reshape(asarray(self.C.value.S.T * normal(size = self.length)), self.base_mesh.shape)
        
        # Scale it and add it to the current value
        f_p = dev * self.scale * self._asf + self.parameter.value
        
        # Create a new realization, forcing it to that value.
        self.parameter.value = Realization(self.M.value, self.C.value, init_base_array = f_p)



class ObservedGPGibbs(SamplingMethod):
    """
    S = ObservedGPGibbs(f, M, C, obs_mesh, obs_taus, obs_vals)
    
    Causes Gaussian process-valued parameter f to take a Gibbs step.
    Applies to the following submodel:
    
    obs_vals ~ N(f(obs_mesh), obs_taus)
    f ~ GP(M,C)
    """
    
    def __init__(self, f, M, C, obs_mesh, obs_taus, obs_vals):

        SamplingMethod.__init__(self, pymc_objects = [f])
        self.f = f
        self._id = 'GPGibbs_'+self.f.__name__
        
        # This local node is valued as the covariance of f's conditional
        # distribution.
        @node
        def C_local(C_real = C,
                    base_mesh = C.base_mesh, 
                    obs_mesh = obs_mesh, 
                    obs_taus = obs_taus):

            val = C_real.copy()
            observe_cov(val, obs_mesh, obs_taus = array([obs_tau, obs_tau]))
            return val
        
        # This local node is valued as the mean of f's conditional distribution.    
        @node
        def M_local(M_real = M
                    base_mesh = C.base_mesh, 
                    C = C_local,
                    obs_vals = obs_vals):

            val=M_real.copy()
            observe_mean_from_cov(val, C, obs_vals)
            return val
            
        self._M = M_local
        self._C = C_local


    def step(self):
        # It's now really easy to take Gibbs steps, since the local nodes
        # take care of computing the conditional mean and covariance.
        self.f.value = Realization(self._M.value, self._C.value)