###
# Adaptive Metropolis Algorithm
# Author: David Huard
# Date: April 23, 2007
# Reference: Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm, Bernouilli, vol. 7 (2), pp. 223-242, 2001.
### 

### Changeset
# Dec. 4, 2007 -- Fixed a slew of bugs. Refactored the code.
# Dec. 5, 2007 -- 

from __future__ import division
import PyMC
from PyMC.utils import msqrt, check_type, round_array
from PyMC import StepMethod, Metropolis, rmvnormal
from PyMC.flib import fill_stdnormal
import numpy as np
from numpy import ndarray, squeeze, eye, zeros, asmatrix, inner,\
    reshape, shape, log, asarray, dot
from numpy.random import randint, random
from PyMC.Node import ZeroProbability
from PyMC.PyMCObjects import BinaryStochastic

class AdaptiveMetropolis(StepMethod):
    """
    S = AdaptiveMetropolis(self, stoch, cov, delay=1000, interval=100, scale={})

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
    def __init__(self, stoch, cov=None, delay=1000, scales=None, interval=100, greedy=True,verbose=0):
        
        self.verbose = verbose
        
        if getattr(stoch, '__class__') is PyMC.PyMCObjects.Stochastic:
            stoch = [stoch] 
    
        StepMethod.__init__(self, stoch, verbose)
        
        
        self.delay = delay
        self.scales = scales
        self.isdiscrete = {}
        self.interval = interval
        self.greedy = greedy
        
        self._ready = False
        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochs])
        
        self.check_type()
        self.dimension()
                   
        ord_sc = self.order_scales(scales)    
        if cov is None:
            self.C_0 = eye(self.dim)*ord_sc/20.
        else:
            self.C_0 = cov
            
        #self._sig = msqrt(self.C_0)
        self._sig = np.linalg.inv(self.C_0)
        
        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy 
        # sampling is done during warm-up. 
        self._trace_count = 0 
        
        self._current_iter = 0
        
        self._proposal_deviate = zeros(self.dim)
        self.C = self.C_0.copy()
        self.chain_mean = asmatrix(zeros(self.dim))
        self._trace = []
        
        if self.verbose >= 1:
            print "Initialization..."
            print 'Dimension: ', self.dim
            print "Ordered scales: ", ord_sc
            print "C_0: ", self.C_0
            print "Sigma: ", self._sig

        # State variables used to restore the state in a latter session. 
        self._state += ['_trace_count', '_current_iter', 'C', '_sig',
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
        ord_sc = np.concatenate(ord_sc)
        
        if squeeze(ord_sc.shape) != self.dim:
            raise "Improper initial scales, dimension don't match", \
                (ord_sc, self.dim)
        return ord_sc
                
    def update_cov(self):
        """Return the updated covariance.
        
        .. math::
          \Sigma_i = \frac{k}{i} C_k + ...
        
        where i is the current index and k is the index from the last time
        the covariance was computed. 
        """
        
        scaling = (2.4)**2/self.dim # Gelman et al. 1996.
        epsilon = 1.0e-5
        chain = asarray(self._trace)
        
        # Recursively compute the chain mean 
        cov, mean = self.recursive_cov(self.C, self._trace_count, 
            self.chain_mean, chain, scaling=scaling, epsilon=epsilon)
        
        if self.verbose > 0:
            print "\tUpdating covariance ...\n", cov
            print "\tUpdating mean ... ", mean
        
        # Update state
        self.C = cov
        #self._sig = msqrt(self.C)
        self._sig = np.linalg.inv(cov)
        self.chain_mean = mean
        self._trace_count += len(self._trace)
        self._trace = []        
        
        
              
              
    def recursive_cov(self, cov, length, mean, chain, scaling=1, epsilon=0):
        r"""Compute the covariance recursively.
        
        Return the new covariance and the new mean. 
        
        .. math::
            C_k & = \frac{1}{k-1} (\sum_{i=1}^k x_i x_i^T - k\bar{x_k}\bar{x_k}^T)
            C_n & = \frac{1}{n-1} (\sum_{i=1}^k x_i x_i^T + \sum_{i=k+1}^n x_i x_i^T - k\bar{x_n}\bar{x_n}^T)
                & = \frac{1}{n-1} ((k-1)C_k + k\bar{x_k}\bar{x_k}^T + \sum_{i=k+1}^n x_i x_i^T - k\bar{x_n}\bar{x_n}^T)
                
        :Parameters:
            -  cov : matrix
                Previous covariance matrix.
            -  length : int
                Length of chain used to compute the previous covariance.
            -  mean : array
                Previous mean. 
            -  chain : array
                Sample used to update covariance.
            -  scaling : float
                Scaling parameter
            -  epsilon : float
                Set to a small value to avoid singular matrices.
            
        """
        n = length + len(chain)
        k = length
        new_mean = self.recursive_mean(mean, length, chain)
        
        t0 = (k-1) * cov
        t2 = k * np.outer(mean, mean)
        t3 = np.dot(chain.T, chain)
        t4 = n*np.outer(new_mean, new_mean)
        t5 = epsilon * np.eye(cov.ndim)
        
        new_cov =  (k-1)/(n-1)*cov + scaling/(n-1.  ) * (t2 + t3 - t4 + t5)
        return new_cov, new_mean
        
    def recursive_mean(self, mean, length, chain):
        r"""Compute the chain mean recursively.
        
        Instead of computing the mean :math:`\bar{x_n}` of the entire chain, 
        use the last computed mean :math:`bar{x_j}` and the tail of the chain 
        to recursively estimate the mean. 
        
        .. math::
            \bar{x_n} & = \frac{1}{n} \sum_{i=1]^n x_i
                      & = \frac{1}{n} (\sum_{i=1]^j x_i + \sum_{i=j+1]^n x_i)
                      & = \frac{j\bar{x_j}}{n} + \frac{\sum_{i=j+1]^n x_i}{n}
        
        :Parameters:
            -  mean : array
                Previous mean.
            -  length : int
                Length of chain used to compute the previous mean.
            -  chain : array
                Sample used to update mean.
        """      
        n = length + len(chain)
        return length * mean / n + chain.sum(0)/n
        

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
        # 1. Done in self.update_cov. A = self._sig
        
        # 2. Draw Z. Fill in place with normal standard variates
        #fill_stdnormal(self._proposal_deviate)
        
        # 3. Compute multivariate jump.
        #arrayjump = inner(self._proposal_deviate, self._sig)
        arrayjump = rmvnormal(np.zeros(self.dim), self._sig)
        
        # 4. Update each stoch individually.
        for stoch in self.stochs:
            jump = reshape(arrayjump[self._slices[stoch]],shape(stoch.value))
            if self.isdiscrete[stoch]:
                stoch.value = stoch.value + round_array(jump)
            else:
                stoch.value = stoch.value + jump
                
    def step(self):
        """
        If the empirical covariance hasn't been computed yet, the step() call
        is passed along to the OneAtATimeMetropolis instances that handle self's
        stochs before the end of the first epoch.
        
        If the empirical covariance has been computed, values for self's stochs
        are proposed and tested simultaneously.
        
        The algorithm is greedy until delay is reached. That is, only accepted
        jumps are stored in the local trace in order to avoid singular 
        covariance matrices. 
        """

        # Probability and likelihood for stoch's current value:
        logp = sum([stoch.logp for stoch in self.stochs])
        loglike = self.loglike

        # Sample a candidate value              
        self.propose()
        
        # Test
        accept = False
        try:
            # Probability and likelihood for stoch's proposed value:
            logp_p = sum([stoch.logp for stoch in self.stochs])
            loglike_p = self.loglike
            if log(random()) < logp_p + loglike_p - logp - loglike:
                accept = True
                self._accepted += 1
            else:
                self._rejected += 1
        except ZeroProbability:
            self._rejected += 1
            
        if self.verbose > 1:
            print "Step ", self._current_iter
            print "\tLogprobability (current, proposed): ", logp, logp_p
            print "\tloglike (current, proposed):      : ", loglike, loglike_p
            for stoch in self.stochs:
                print "\t", stoch.__name__, stoch.last_value, stoch.value
            if accept:
                print "\tAccepted\t*******\n"
            else: 
                print "\tRejected\n"
            print "\tAcceptance ratio: ", self._accepted/(self._accepted+self._rejected)
            
        if self._current_iter == self.delay: 
            self.greedy = False
            
        if not accept:
            for stoch in self.stochs:
                stoch.value = stoch.last_value
        
        if accept or not self.greedy:
            self.internal_tally()

        if self._current_iter>self.delay and self._current_iter%self.interval==0:
           self.update_cov()
    
        self._current_iter += 1
    
    def internal_tally(self):
        """Store the trace of stochs for the computation of the covariance.
        This is completely independent from the backend used by the sampler to 
        store the samples."""
        chain = []
        for stoch in self.stochs:
            chain.append(stoch.value.ravel())
        self._trace.append(np.concatenate(chain))
        
    
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
