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

import numpy as np
from numpy.random import randint, random

import PyMC
from PyMC.utils import msqrt, check_type, round_array
from PyMC import StepMethod, Metropolis, rmvnormal
from PyMC.flib import fill_stdnormal
from PyMC.Node import ZeroProbability
from PyMC.PyMCObjects import BinaryStochastic

class AdaptiveMetropolis(StepMethod):
    """
    S = AdaptiveMetropolis(self, stoch, cov, delay=1000, interval=100, scale={})

    The AdaptativeMetropolis (AM) sampling algorithm works like a regular 
    Metropolis, with the exception that stochastic parameters are block-updated 
    using a multivariate jump distribution whose covariance is tuned during 
    sampling. Although the chain is non-Markovian, i.e. the proposal 
    distribution is asymetric, it has correct ergodic properties. See
    (Haario et al., 2001) for details. 
    
    :Parameters:
      - stoch : PyMC objects
            Stochastic objects to be handled by the AM algorith,
            
      - cov : array
            Initial guess for the covariance matrix C_0. 
            
      - delay : int
          Number of iterations before the empirical covariance is computed.
        
      - interval : int
          Interval between covariance updates.
          
      - scale : dict
          Dictionary containing the scale for each stoch keyed by name.
          If cov is None, those scales are used to define an initial covariance
          C_0. If neither cov nor scale is given, the initial covariance is 
          guessed from the objects value (or trace if available).

    """
    def __init__(self, stoch, cov=None, delay=1000, scales=None, interval=100, greedy=True,verbose=0):
        
        self.verbose = verbose
        
        if getattr(stoch, '__class__') is PyMC.PyMCObjects.Stochastic:
            stoch = [stoch] 
        StepMethod.__init__(self, stoch, verbose)
        
        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochs])
        
        # State variables used to restore the state in a latter session. 
        self._state += ['_trace_count', '_current_iter', 'C', '_sig',
        '_proposal_deviate', '_trace']
        
        self.delay = delay
        self.isdiscrete = {}
        self.interval = interval
        self.greedy = greedy
        
        self.check_type()
        self.dimension()
        self.C_0 = self.initialize_cov(cov, scales)           
        
        #self._sig = msqrt(self.C_0)
        self._sig = np.linalg.inv(self.C_0)
        
        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy 
        # sampling can be done during warm-up period.
        self._trace_count = 0 
        self._current_iter = 0
        
        self._proposal_deviate = np.zeros(self.dim)
        self.C = self.C_0.copy()
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []
        
        if self.verbose >= 1:
            print "Initialization..."
            print 'Dimension: ', self.dim
            print "C_0: ", self.C_0
            print "Sigma: ", self._sig
              
    @staticmethod
    def competence(stoch):
        """
        The competence function for AdaptiveMetropolis.
        The AM algorithm is well suited to deal with multivariate
        parameters. 
        """
        if isinstance(stoch, BinaryStochastic):
            return 0
        elif np.iterable(stoch):
            return 2
                
                
    def initialize_cov(self, cov=None, scales=None, scaling=20):
        """Define C_0, the initial jump distributioin covariance matrix.
        
        Return:
            - cov,  if cov != None
            - covariance matrix built from the scales dictionary if scales!=None
            - covariance matrix estimated from the stochs trace
            - covariance matrix estimated from the stochs value, scaled by 
                scaling parameter.
        """
        if cov:
            return cov
        elif scales:
            ord_sc = self.order_scales(scales)    
            return np.eye(self.dim)*ord_sc
        else:
            try:
                a = self.trace2array(-2000, -1)
                nz = a[:, 0]!=0
                return np.cov(a[nz, :], rowvar=0)
            except:
                ord_sc = []
                for stoch in self.stochs:
                    ord_sc.append(stoch.value.ravel())
                return np.eye(self.dim)*ord_sc/scaling
            
        
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
            if isinstance(stoch.value, np.ndarray):
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
            ord_sc.append(scales[stoch])
        ord_sc = np.concatenate(ord_sc)
        
        if np.squeeze(ord_sc.shape) != self.dim:
            raise "Improper initial scales, dimension don't match", \
                (ord_sc, self.dim)
        return ord_sc
                
    def update_cov(self):
        """Recursively compute the covariance matrix for the multivariate normal 
        proposal distribution.
        
        This method is called every self.interval once self.delay iterations 
        have been performed.
        """
        
        scaling = (2.4)**2/self.dim # Gelman et al. 1996.
        epsilon = 1.0e-5
        chain = np.asarray(self._trace)
        
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
        This method proposes values for stochs based on the empirical
        covariance of the values sampled so far.
        
        The proposal jumps are drawn from a multivariate normal distribution.        
        """
                
        #fill_stdnormal(self._proposal_deviate)
        #arrayjump = np.inner(self._proposal_deviate, self._sig)
        
        arrayjump = rmvnormal(np.zeros(self.dim), self._sig)
        
        # 4. Update each stoch individually.
        for stoch in self.stochs:
            jump = np.reshape(arrayjump[self._slices[stoch]],np.shape(stoch.value))
            if self.isdiscrete[stoch]:
                stoch.value = stoch.value + round_array(jump)
            else:
                stoch.value = stoch.value + jump
                
    def step(self):
        """
        Perform a Metropolis step. 
        
        Stochastic parameters are block-updated using a multivariate normal 
        distribution whose covariance is updated every self.interval once 
        self.delay steps have been performed. 
        
        The AM instance keeps a local copy of the stochastic parameter's trace.
        This trace is used to computed the empirical covariance, and is 
        completely independent from the Database backend.

        If self.greedy is True and the number of iterations is smaller than 
        self.delay, only accepted jumps are stored in the internal 
        trace to avoid computing singular covariance matrices. 
        """

        # Probability and likelihood for stoch's current value:
        logp = sum([stoch.logp for stoch in self.stochs])
        loglike = self.loglike

        # Sample a candidate value              
        self.propose()
        
        # Metropolis acception/rejection test
        accept = False
        try:
            # Probability and likelihood for stoch's proposed value:
            logp_p = sum([stoch.logp for stoch in self.stochs])
            loglike_p = self.loglike
            if np.log(random()) < logp_p + loglike_p - logp - loglike:
                accept = True
                self._accepted += 1
            else:
                self._rejected += 1
        except ZeroProbability:
            self._rejected += 1
            
        if self.verbose > 2:
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
        This trace is completely independent from the backend used by the 
        sampler to store the samples."""
        chain = []
        for stoch in self.stochs:
            chain.append(stoch.value.ravel())
        self._trace.append(np.concatenate(chain))
        
    def trace2array(i0,i1):
        """Return an array with the trace of all stochs from index i0 to i1."""
        chain = []
        for stoch in self.stochs:
            chain.append(ravel(stoch.trace.gettrace(slicing=slice(i0,i1))))
        return concatenate(chain)
        
    def tune(self, verbose):
        """Tuning is done during the entire run, independently from the Sampler 
        tuning specifications. """
        return False
   
if __name__=='__main__':
    from numpy.testing import *
    from PyMC import Sampler, JointMetropolis
    from PyMC import stoch, data, JointMetropolis
    from numpy import array,  ones
    from PyMC.distributions import multivariate_normal_like
    class AMmodel:
        mu_A = array([0.,0.])
        tau_A = np.eye(2)
        @stoch
        def A(value = ones(2,dtype=float), mu=mu_A, tau = tau_A):
            return multivariate_normal_like(value,mu,tau)
        
        tau_B = np.eye(2) * 100.          
        @stoch
        def B(value = ones(2,dtype=float), mu = A, tau = tau_B):
            return multivariate_normal_like(value,mu,tau)
    
        AM = AdaptativeMetropolis([A,B])
    
    class test_AdaptativeMetropolis(NumpyTestCase):
        S = Sampler(AMmodel, 'ram')
        S.sample(10000)
        print S.A.trace(burn=5000, thin=50)

    NumpyTest().run
