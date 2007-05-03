###
# Adaptative Metropolis Algorithm
# Author: David Huard
# Date: April 23, 2007
# Reference: Haario, H., E. Saksman and J. Tamminen, An adaptative Metropolis algorithm, Bernouilli, vol. 7 (2), pp. 223-242, 2001.
###
from PyMC2.utils import msqrt, extend_children, check_type, round_array, extend_parents
from PyMC2 import SamplingMethod
from PyMC2.flib import fill_stdnormal
from numpy import ndarray, concatenate, squeeze, eye, zeros, asmatrix, inner,\
    reshape, shape, log, asarray, dot
from numpy.random import randint, random
from PyMC2.PyMCBase import ZeroProbability

class AdaptativeMetropolis(SamplingMethod):
    """
    S = AdaptativeMetropolis(pymc_objects, delay=1000, rate=1, scale={})

    Applies the Metropolis-Hastings algorithm to several parameters
    together. Jumping density is a multivariate normal distribution
    with mean zero and covariance equal to the empirical covariance
    of the parameters.
    
    :Parameters:
      - delay : int
          Number of iterations before the empirical covariance is computed. 
      - rate : int
          Interval between covariance updates.
      - scale : dict
          Dictionary containing the scale for each parameter keyed by name.
          The scales are used to define an initial covariance matrix used 
          until delay is reached.  

    """
    def __init__(self, pymc_objects=None, delay=1000, rate=1, scales={}):
        
        SamplingMethod.__init__(self,pymc_objects)
        self.check_type()
        self.delay = delay
        self.rate = rate
        
        self.dimension()
        self.scales(scales)    
        self.C_0 = eye(self.dim)*self._init_scale
        self._sig = msqrt(self.C_0)
        self.scaling_parameter = (2.4)**2/self.dim # Gelman et al. 1996.
        
        self._last_trace_index = 0
        self._proposal_deviate = zeros(self.dim)
        self.C = zeros((self.dim,self.dim))
        self.chain_mean = asmatrix(zeros(self.dim))
        self._trace = []
        
        # State variables used to restore the state in a latter session. 
        self._id = 'AdaptativeMetropolis_'+'_'.join([p.__name__ for p in self.parameters])
        self._state += ['last_trace_index', '_cov', '_sig',
        '_proposal_deviate', '_trace']

               
    def check_type(self):
        """Make sure each parameter has a correct type, and identify discrete parameters."""
        self.isdiscrete = {}
        for parameter in self.parameters:
            type_now = check_type(parameter)[0]
            if not type_now is float and not type_now is int:
                raise TypeError,    'Parameter ' + parameter.__name__ + "'s value must be numeric"+\
                                    'or ndarray with numeric dtype for JointMetropolis to be applied.'
            elif type_now is int:
                self.isdiscrete[parameter] = True
            else:
                self.isdiscrete[parameter] = False
                
                
    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each parameter.
        """
        self.dim = 0
        self._slices = {}
        for parameter in self.parameters:
            if isinstance(parameter.value, ndarray):
                p_len = len(parameter.value.ravel())
            else:
                p_len = 1
            self._slices[parameter] = slice(self.dim, self.dim + p_len)
            self.dim += p_len
            
            
    def scales(self, init_scales):
        """Define an array of scales to build the initial covariance.
        If init_scales is None, the scale is taken to be the initial value of 
        the parameters.
        """
        s = []
        for parameter in self.parameters:
            try:
                s.append(init_scales[parameter])
            except KeyError:
                s.append(parameter.value.ravel())
                
        self._init_scale = concatenate(s)
        if squeeze(self._init_scale.shape) != self.dim:
            raise "Improper initial scales, dimension don't match", \
                (self._init_scale.shape, self.dim)
        
    def covariance(self):
        """
        Return the covariance of the chain. 
        """
        if self._model._current_iter < self.delay:
            return self.C_0
        else:
            return self.C
            
    def update_cov(self):
        """Return the updated covariance.
        
        .. math::
          \Sigma_i = \frac{k}{i} C_k + ...
        
        where i is the current index and k is the index from the last time
        the covariance was computed. 
        """
        
        epsilon = 1.0e-6
        i0 = self._last_trace_index
        i = self._model._cur_trace_index
        s_d = self.scaling_parameter
        n = i - i0
        chain = asarray(self._trace)
        
        t1 = i0/i * self.covariance()
        t2 =  (i0+1) * dot(self.chain_mean.T, self.chain_mean)
        self.update_mean(chain)
        t3 = (i+1) * dot(self.chain_mean.T, self.chain_mean)
        t4 = dot(chain.T, chain)
        t5 = epsilon * eye(self.dim)
        
        self.C = t1 + s_d/i * (t2 - t3 + t4 + t5)
        self._sig = msqrt(self.C)
        self._trace = []
              
    def update_mean(self, chain):
        """Update the chain mean"""
        self.chain_mean = 1./self._model._current_iter * \
        ( self._last_trace_index * self.chain_mean + chain.sum(0))
        
    def trace2array(i0,i1):
        """Return an array with the trace of all parameters from index i0 to i1."""
        chain = []
        for parameter in self.parameters:
            chain.append(ravel(parameter.trace.gettrace(slicing=slice(i0,i1))))
        return concatenate(chain)

    def propose(self):
        """
        This method proposes values for self's parameters based on the empirical
        covariance :math:`\Sigma`.
        
        The proposal jumps X are drawn from a multivariate normal distribution
        by the following method:
          1. Compute `A`, the Cholesky decomposition (matrix square root) of the
             covariance matrix. :math:`AA^T = \Sigma`
          2. Draw Z, a vector of n independent standard normal variates.
          3. Compute X = AZ.
        
        """
        # 1. Done in compute_sig. A = self._sig
        
        # 2. Draw Z. Fill in place with normal standard variates
        fill_stdnormal(self._proposal_deviate)
        
        # 3. Compute multivariate jump.
        arrayjump = inner(self._proposal_deviate, self._sig)
                
        # 4. Update each parameter individually.
        # TODO: test inplace jump, ie parameter.value += jump
        for parameter in self.parameters:
            jump = reshape(arrayjump[self._slices[parameter]],shape(parameter.value))
            if self.isdiscrete[parameter]:
                parameter.value = parameter.value + round_array(jump)
            else:
                parameter.value = parameter.value + jump
                
        # 5. Store the trace internally. 
        self._arrayjump = arrayjump

    def step(self):
        """
        If the empirical covariance hasn't been computed yet, the step() call
        is passed along to the OneAtATimeMetropolis instances that handle self's
        parameters before the end of the first epoch.
        
        If the empirical covariance has been computed, values for self's parameters
        are proposed and tested simultaneously.
        """

        # Probability and likelihood for parameter's current value:
        logp = sum([parameter.logp for parameter in self.parameters])
        loglike = self.loglike

        # Sample a candidate value
        self.propose()
        
        try:
            # Probability and likelihood for parameter's proposed value:
            logp_p = sum([parameter.logp for parameter in self.parameters])
            loglike_p = self.loglike
            
            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                raise 'Rejected'
            else:
                self._accepted += 1
                self._trace.append(self._arrayjump)
                
        except (ZeroProbability, 'Rejected'):
            self._rejected += 1
            
            if self._model._current_iter > self.delay: 
                self._trace.append(self._arrayjump)
                
            for parameter in self.parameters:
                parameter.value = parameter.last_value
            return

        if self._model._current_iter>self.delay and \
            self._model._current_iter%self.rate==0:
           self.update_cov()
           self.last_trace_index = self._model._current_iter
           
   
if __name__=='__main__':
    from numpy.testing import *
    from PyMC2 import Sampler, JointMetropolis
    from PyMC2 import parameter, data, JointMetropolis
    from numpy import array, eye, ones
    from PyMC2.distributions import multivariate_normal_like
    class AMmodel:
        mu_A = array([0.,0.])
        tau_A = eye(2)
        @parameter
        def A(value = ones(2,dtype=float), mu=mu_A, tau = tau_A):
            return multivariate_normal_like(value,mu,tau)
        
        tau_B = eye(2) * 100.          
        @parameter
        def B(value = ones(2,dtype=float), mu = A, tau = tau_B):
            return multivariate_normal_like(value,mu,tau)
    
        AM = AdaptativeMetropolis([A,B])
    
    class test_AdaptativeMetropolis(NumpyTestCase):
        S = Sampler(AMmodel, 'ram')
        S.sample(10000)
        print S.A.trace(burn=5000, thin=50)

    NumpyTest().run
