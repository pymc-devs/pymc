###
# Adaptive Metropolis Algorithm
# Author: David Huard
# Date: April 23, 2007
# Reference: Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm, Bernouilli, vol. 7 (2), pp. 223-242, 2001.
###
import PyMC
from PyMC.utils import msqrt, check_type, round_array
from PyMC import StepMethod
from PyMC.flib import fill_stdnormal
from numpy import ndarray, concatenate, squeeze, eye, zeros, asmatrix, inner,\
    reshape, shape, log, asarray, dot
from numpy.random import randint, random
from PyMC.Node import ZeroProbability

class AdaptiveMetropolis(StepMethod):
    """
    S = AdaptiveMetropolis(self, stoch, cov, delay=1000, scale={})

    Applies the Metropolis-Hastings algorithm to several stochs
    together. Jumping density is a multivariate normal distribution
    with mean zero and covariance equal to the empirical covariance
    of the stochs.
    
    :Parameters:
    
      - stoch : PyMC objects
            These objects are to be handled using the AdaptativeMetropolis step 
            method.
            
      - cov : array
            Initial guess for the covariance matrix C_0. 
            
      - delay : int
          Number of iterations before the empirical covariance is computed.
          Equivalent to t_0 in Haario et al. (2001).
        
      - interval : int
          Interval between covariance updates.
          
      - scale : dict
          Dictionary containing the scale for each stoch keyed by name.
          The scales are used to define an initial covariance matrix used 
          until delay is reached. If it not given, and cov is None, a first 
          guess is estimated using the current objects value. 

    """
    def __init__(self, stoch, cov=None, epoch=1000, delay=1000, scales=None, interval=100, verbose=0):
        
        self.verbose = verbose
        
        if getattr(stoch, '__class__') is PyMC.PyMCObjects.Stochastic:
            stoch = [stoch] 
    
        StepMethod.__init__(self,stoch, verbose)
        
        
        self.epoch = epoch
        self.delay = delay
        self.scales = scales
        self.isdiscrete = {}
        self.interval = interval

        self._ready = False
        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochs])
        
        self.check_type()
        self.dimension()
        
        ord_sc = self.order_scales(scales)    
        if cov is None:
            self.C_0 = eye(self.dim)*ord_sc
        else:
            self.C_0 = cov
        self._sig = msqrt(self.C_0)
        self.scaling_stoch = (2.4)**2/self.dim # Gelman et al. 1996.
        
        self._last_trace_index = 0
        self._current_iter = 0
        self._proposal_deviate = zeros(self.dim)
        self.C = zeros((self.dim,self.dim))
        self.chain_mean = asmatrix(zeros(self.dim))
        self._trace = []
        
        # State variables used to restore the state in a latter session. 
        self._state += ['_last_trace_index', '_current_iter', 'C', '_sig',
        '_proposal_deviate', '_trace']

               
    @staticmethod
    def competence(stoch):
        """
        The competence function for AdaptiveMetropolis.
        """

        if isinstance(stoch, BinaryStochastic):
            return 0

        else:
            _type = check_type(stoch)[0]
            if _type in [float, int]:
                return 1
            else:
                return 2
                
                
    def check_type(self):
        """Make sure each stoch has a correct type, and identify discrete stochs."""
        self.isdiscrete = {}
        for stoch in self.stochs:
            type_now = check_type(stoch)[0]
            if not type_now is float and not type_now is int:
                raise TypeError,    'Stochastic ' + stoch.__name__ + "'s value must be numeric"+\
                                    'or ndarray with numeric dtype for JointMetropolis to be applied.'
            elif type_now is int:
                self.isdiscrete[stoch] = True
            else:
                self.isdiscrete[stoch] = False
                
                
    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stoch.
        """
        self.dim = 0
        self._slices = {}
        for stoch in self.stochs:
            if isinstance(stoch.value, ndarray):
                p_len = len(stoch.value.ravel())
            else:
                p_len = 1
            self._slices[stoch] = slice(self.dim, self.dim + p_len)
            self.dim += p_len
            
            
    def order_scales(self, scales):
        """Define an array of scales to build the initial covariance.
        If init_scales is None, the scale is taken to be the initial value of 
        the stochs.
        """
        ord_sc = []
        for stoch in self.stochs:
            try:
                ord_sc.append(scales[stoch])
            except (TypeError, KeyError):
                ord_sc.append(stoch.value.ravel())
        ord_sc = concatenate(ord_sc)
        
        if squeeze(ord_sc.shape) != self.dim:
            raise "Improper initial scales, dimension don't match", \
                (ord_sc, self.dim)
        if self.verbose >= 1:
            print "Ordered scales : ", ord_sc
        return ord_sc
        
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
        i = self._current_iter
        s_d = self.scaling_stoch
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
        self.chain_mean = 1./self._current_iter * \
        ( self._last_trace_index * self.chain_mean + chain.sum(0))
        
    def trace2array(i0,i1):
        """Return an array with the trace of all stochs from index i0 to i1."""
        chain = []
        for stoch in self.stochs:
            chain.append(ravel(stoch.trace.gettrace(slicing=slice(i0,i1))))
        return concatenate(chain)

    def propose(self):
        """
        This method proposes values for self's stochs based on the empirical
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
                
        # 4. Update each stoch individually.
        # TODO: test inplace jump, ie stoch.value += jump
        # This won't work - in-place stoch value updates aren't allowed.

        for stoch in self.stochs:
            jump = reshape(arrayjump[self._slices[stoch]],shape(stoch.value))
            if self.isdiscrete[stoch]:
                stoch.value = stoch.value + round_array(jump)
            else:
                stoch.value = stoch.value + jump
                
        # 5. Store the trace internally. 
        self._arrayjump = arrayjump

    def step(self):
        """
        If the empirical covariance hasn't been computed yet, the step() call
        is passed along to the OneAtATimeMetropolis instances that handle self's
        stochs before the end of the first epoch.
        
        If the empirical covariance has been computed, values for self's stochs
        are proposed and tested simultaneously.
        """

        # Probability and likelihood for stoch's current value:
        logp = sum([stoch.logp for stoch in self.stochs])
        loglike = self.loglike

        # Sample a candidate value
        self.propose()
        
        try:
            # Probability and likelihood for stoch's proposed value:
            logp_p = sum([stoch.logp for stoch in self.stochs])
            loglike_p = self.loglike
            
            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                raise 'Rejected'
            else:
                self._accepted += 1
                self._trace.append(self._arrayjump)
                
        except (ZeroProbability, 'Rejected'):
            self._rejected += 1
            
            if self._current_iter > self.delay: 
                self._trace.append(self._arrayjump)
                
            for stoch in self.stochs:
                stoch.value = stoch.last_value
            return

        if self._current_iter>self.delay and \
            self._current_iter%self.interval==0:
           self.update_cov()
           self.last_trace_index = self._current_iter
           
    def tune(self, verbose):
        return True
   
if __name__=='__main__':
    from numpy.testing import *
    from PyMC import Sampler, JointMetropolis
    from PyMC import stoch, data, JointMetropolis
    from numpy import array, eye, ones
    from PyMC.distributions import multivariate_normal_like
    class AMmodel:
        mu_A = array([0.,0.])
        tau_A = eye(2)
        @stoch
        def A(value = ones(2,dtype=float), mu=mu_A, tau = tau_A):
            return multivariate_normal_like(value,mu,tau)
        
        tau_B = eye(2) * 100.          
        @stoch
        def B(value = ones(2,dtype=float), mu = A, tau = tau_B):
            return multivariate_normal_like(value,mu,tau)
    
        AM = AdaptativeMetropolis([A,B])
    
    class test_AdaptativeMetropolis(NumpyTestCase):
        S = Sampler(AMmodel, 'ram')
        S.sample(10000)
        print S.A.trace(burn=5000, thin=50)

    NumpyTest().run
