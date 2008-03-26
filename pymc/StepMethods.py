from __future__ import division

import numpy as np
from utils import msqrt, check_type, round_array, float_dtypes, integer_dtypes, bool_dtypes, safe_len
from numpy import ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any, array, all, abs, exp, where, isscalar
from numpy.linalg.linalg import LinAlgError
from numpy.random import randint, random
from numpy.random import normal as rnormal
from flib import fill_stdnormal
from PyMCObjects import Stochastic, Potential, Deterministic
from Container import Container
from Node import ZeroProbability, Node, Variable, StochasticBase
from     pymc.decorators import prop
from copy import copy
from InstantiationDecorators import deterministic
import pdb, warnings

__docformat__='reStructuredText'

# Changeset history
# 22/03/2007 -DH- Added a _state attribute containing the name of the attributes that make up the state of the step method, and a method to return that state in a dict. Added an id.
# TODO: Test cases for binary and discrete Metropolises.

conjugate_Gibbs_competence = 0
nonconjugate_Gibbs_competence = 0

__all__=['DiscreteMetropolis', 'Metropolis', 'StepMethod', 'assign_method',  'pick_best_methods', 'StepMethodRegistry', 'NoStepper', 'BinaryMetropolis', 'AdaptiveMetropolis','Gibbs','conjugate_Gibbs_competence', 'nonconjugate_Gibbs_competence']

StepMethodRegistry = []

def pick_best_methods(stochastic):
    """
    Picks the StepMethods best suited to handle
    a stochastic variable.
    """
    
    # Keep track of most competent methohd
    max_competence = 0
    # Empty set of appropriate StepMethods
    best_candidates = set([])
    
    # Loop over StepMethodRegistry
    for method in StepMethodRegistry:
        
        # Parse method and its associated competence
        competence = method.competence(stochastic)
        
        # If better than current best method, promote it
        if competence > max_competence:
            best_candidates = set([method])
            max_competence = competence
        
        # If same competence, add it to the set of best methods
        elif competence == max_competence:
            best_candidates.add(method)
    
    if max_competence<=0:
        raise ValueError, 'Maximum competence reported for stochastic %s is <= 0... you may need to write a custom step method class.' % stochastic.__name__
    
    # print s.__name__ + ': ', best_candidates, ' ', max_competence
    return best_candidates

def assign_method(stochastic, scale=None):
    """
    Returns a step method instance to handle a
    variable. If several methods have the same competence,
    it picks one arbitrarily (using set.pop()).
    """
    
    # Retrieve set of best candidates
    best_candidates = pick_best_methods(stochastic)
    
    # Randomly grab and appropriate method
    method = best_candidates.pop()
        
    if scale:
        return method(stochastic, scale = scale)
    
    return method(stochastic)


class StepMethodMeta(type):
    """
    Automatically registers new step methods.
    """
    def __init__(cls, name, bases, dict):
        type.__init__(cls)
        StepMethodRegistry.append(cls)
        
class StepMethod(object):
    """
    This object knows how to make Stochastics take single MCMC steps.
    It's sample() method will be called by Model at every MCMC iteration.
    
    :Parameters:
          -variables : list, array or set
            Collection of PyMCObjects
          
          - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
    
    Externally-accessible attributes:
      stochastics:   The Stochastics over which self has jurisdiction which have isdata = False.
      children: The combined children of all Variables over which self has jurisdiction.
      parents:  The combined parents of all Nodes over which self has jurisdiction, as a set.
      loglike:  The summed log-probability of self's children conditional on all of self's 
                  Variables' current values. These will be recomputed only as necessary. 
                  This descriptor should eventually be written in C.
    
    Externally accesible methods:
      sample(): A single MCMC step for all the Stochastics over which self has 
        jurisdiction. Must be overridden in subclasses.
      tune(): Tunes proposal distribution widths for all self's Stochastics.
      competence(s): Examines Stochastic instance s and returns self's 
        competence to handle it, on a scale of 0 to 3.
    
    To instantiate a StepMethod called S with jurisdiction over a
    sequence/set N of Nodes:
      
      >>> S = StepMethod(N)
    
    :SeeAlso: Metropolis, Sampler.
    """
    
    __metaclass__ = StepMethodMeta
    
    def __init__(self, variables, verbose=0):
        # StepMethod initialization
        
        if not hasattr(variables, '__iter__'):
            variables = [variables]
        
        self.stochastics = set()
        self.children = set()
        self.parents = set()
        
        # Initialize hidden attributes
        self._asf = 1.
        self._accepted = 0.
        self._rejected = 0.
        self._state = ['_rejected', '_accepted', '_asf']
        self.verbose = verbose
        
        # File away the variables
        for variable in variables:
            
            # Sort.
            if isinstance(variable,Stochastic):
                if not variable.isdata:
                    self.stochastics.add(variable)
        
        if len(self.stochastics)==0:
            raise ValueError, 'No stochastics provided.'
        
        # Find children, no need to find parents; each variable takes care of those.
        for variable in variables:
            self.children |= variable.children
            for parent in variable.parents.itervalues():
                if isinstance(parent, Variable):
                    self.parents.add(parent)
        
        self.children = set([])
        self.parents = set([])
        for s in self.stochastics:
            self.children |= s.extended_children
            self.parents |= s.extended_parents
        
        # Remove own stochastics from children and parents.
        self.children -= self.stochastics
        self.parents -= self.stochastics
        
        # ID string for verbose feedback
        self._id = 'To define in subclasses'
    
    def step(self):
        """
        Specifies single step of step method.
        Must be overridden in subclasses.
        """
        pass

    @staticmethod
    def competence(s):
        """
        This function is used by Sampler to determine which step method class
        should be used to handle stochastic variables.
        
        Return value should be a competence
        score from 0 to 3, assigned as follows:

        0:  I can't handle that variable.
        1:  I can handle that variable, but I'm a generalist and
            probably shouldn't be your top choice (Metropolis
            and friends fall into this category).
        2:  I'm designed for this type of situation, but I could be
            more specialized.
        3:  I was made for this situation, let me handle the variable.

        In order to be eligible for inclusion in the registry, a sampling
        method's init method must work with just a single argument, a
        Stochastic object.
        
        If you want to exclude a particular step method from
        consideration for handling a variable, do this:

        Competence functions MUST be called 'competence' and be decorated by the 
        '@staticmethod' decorator. Example:
        
            @staticmethod
            def competence(s):
                if isinstance(s, MyStochasticSubclass):
                    return 2
                else:
                    return 0
        
        :SeeAlso: pick_best_methods, assign_method
        """
        return 0
    
    
    def tune(self, divergence_threshold=1e10, verbose=0):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate of the last k proposals:
        
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        
        This method is called exclusively during the burn-in period of the
        sampling algorithm.
        
        May be overridden in subclasses.
        """
        
        # Verbose feedback
        if verbose > 0 or self.verbose > 0:
            print '\t%s tuning:' % self._id
        
        # Flag for tuning state
        tuning = True        
        
        # Calculate recent acceptance rate
        if not (self._accepted + self._rejected): return tuning
        acc_rate = self._accepted / (self._accepted + self._rejected)
        
        
        # Switch statement
        if acc_rate<0.001:
            # reduce by 90 percent
            self._asf *= 0.1
        elif acc_rate<0.05:
            # reduce by 50 percent
            self._asf *= 0.5
        elif acc_rate<0.2:
            # reduce by ten percent
            self._asf *= 0.9
        elif acc_rate>0.95:
            # increase by factor of ten
            self._asf *= 10.0
        elif acc_rate>0.75:
            # increase by double
            self._asf *= 2.0
        elif acc_rate>0.5:
            # increase by ten percent
            self._asf *= 1.1
        else:
            tuning = False
        
        # Re-initialize rejection count
        self._rejected = 0.
        self._accepted = 0.
        
        # More verbose feedback, if requested
        # Warning: self.stochastic is not defined above. The following assumes
        # that the class has created this value, which is a bit fragile. DH
        if verbose > 0 or self.verbose > 0:
            print '\t\tvalue:', self.stochastic.value
            print '\t\tacceptance rate:', acc_rate
            print '\t\tadaptive scale factor:', self._asf
            print
        
        return tuning
    
    def _get_loglike(self):
        # Fetch log-probability (as sum of childrens' log probability)

        # Initialize sum
        sum = 0.
        
        # Loop over children
        for child in self.children:
            
            # Verbose feedback
            if self.verbose > 1:
                print '\t'+self._id+' Getting log-probability from child ' + child.__name__
            
            # Increment sum
            sum += child.logp
        if self.verbose > 1:
            print '\t' + self._id + ' Current log-likelihood ', sum
        return sum
    
    # Make get property for retrieving log-probability
    loglike = property(fget = _get_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochastics, with self.stochastics removed.")
    
    def current_state(self):
        """Return a dictionary with the current value of the variables defining
        the state of the step method."""
        state = {}
        for s in self._state:
            state[s] = getattr(self, s)
        return state

    @prop
    def ratio():
        """Acceptance ratio"""
        def fget(self):
            return self._accepted/(self._accepted + self._rejected)
        return locals()
        
class NoStepper(StepMethod):
    """
    Step and tune methods do nothing.
    
    Useful for holding stochastics constant without setting isdata=True.
    """
    def step(self):
        pass
    def tune(self):
        pass

# The default StepMethod, which Model uses to handle singleton stochastics.
class Metropolis(StepMethod):
    """
    The default StepMethod, which Model uses to handle singleton, continuous variables.
    
    Applies the one-at-a-time Metropolis-Hastings algorithm to the Stochastic over which self has jurisdiction.
    
    To instantiate a Metropolis called M with jurisdiction over a Stochastic P:
      
      >>> M = Metropolis(P, scale=1, dist=None)
    
    :Arguments:
    - s : Stochastic
            The variable over which self has jurisdiction.
    
    - scale (optional) : number
            The proposal jump width is set to scale * variable.value.
    
    - dist (optional) : string
            The proposal distribution. May be 'Normal', 'RoundedNormal', 'Bernoulli',
            'Prior' or None. If None is provided, a proposal distribution is
            chosen by examining P.value's type.
    
    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
    
    :SeeAlso: StepMethod, Sampler.
    """
    
    def __init__(self, stochastic, scale=1., sig=None, dist=None, verbose=0):
        # Metropolis class initialization
        
        # Initialize superclass
        StepMethod.__init__(self, [stochastic], verbose=verbose)
        
        # Set public attributes
        self.stochastic = stochastic
        self.verbose = verbose
        
        # Add _sig to state
        self._state += ['_sig', '_dist']
        
        # Avoid zeros when setting proposal variance
        if sig is not None:
            self._sig = sig
        else:
            if all(self.stochastic.value != 0.):
                self._sig = ones(shape(self.stochastic.value)) * abs(self.stochastic.value) * scale
            else:
                self._sig = ones(shape(self.stochastic.value)) * scale
        
        # Initialize proposal deviate with array of zeros
        self.proposal_deviate = zeros(shape(self.stochastic.value), dtype=float)
        
        # Initialize verbose feedback string
        self._id = stochastic.__name__
        
        # Determine size of stochastic
        if isinstance(self.stochastic.value, ndarray):
            self._len = len(self.stochastic.value.ravel())
        else:
            self._len = 1
        
        # If no dist argument is provided, assign a proposal distribution automatically.
        if not dist:
            
            # Pick Gaussian, just because
            self._dist = "Normal"
            
            # If self's extended children has no stochastics, proposing from the prior is best.
            if sum([isinstance(child, StochasticBase) for child in self.children]) == 0:
                try:
                    self.stochastic.random()
                    self._dist = "Prior"
                except:
                    pass

        else:
            self._dist = dist
    
    @staticmethod
    def competence(s):
        """
        The competence function for Metropolis
        """
        # If no stochastics depend on this stochastic, I'll just propose it from its conditional prior.
        # This is the best possible step method for this stochastic.
        if len(s.extended_children)==0:
            try:
                s.rand()
                return 3                
            except:
                pass
                
        if s.dtype is None:
            return .5

        if not s.dtype in float_dtypes:
            # If the stochastic's binary or discrete, I can't do it.
            return 0
        else:
            return 1
    
    def hastings_factor(self):
        """
        If this is a Metropolis-Hastings method (proposal is not symmetric random walk), 
        this method should return log(back_proposal) - log(forward_proposal).
        """
        return 0.
    
    def step(self):
        """
        The default step method applies if the variable is floating-point
        valued, and is not being proposed from its prior.
        """
        
        # Probability and likelihood for s's current value:
        
        if self.verbose > 0:
            print
            print self._id + ' getting initial prior.'
        
        if self._dist == "Prior":
            # No children
            logp = 0.
        else:
            logp = self.stochastic.logp
        
        if self.verbose > 0:
            print self._id + ' getting initial likelihood.'
        loglike = self.loglike
        
        if self.verbose > 0:
            print self._id + ' proposing.'
        
        # Sample a candidate value
        self.propose()
        
        # Probability and likelihood for s's proposed value:
        try:
            if self._dist == "Prior":
                logp_p = 0.
                # Check for weirdness before accepting jump
                self.stochastic.logp
            else:
                logp_p = self.stochastic.logp
            loglike_p = self.loglike
        
        except ZeroProbability:
            
            # Reject proposal
            if self.verbose > 0:
                print self._id + ' rejecting due to ZeroProbability.'
            self.reject()
            
            # Increment rejected count
            self._rejected += 1
            
            if self.verbose > 1:
                print self._id + ' returning.'
            return
        
        if self.verbose > 1:
            print 'logp_p - logp: ', logp_p - logp
            print 'loglike_p - loglike: ', loglike_p - loglike
        
        HF = self.hastings_factor()
        
        # Evaluate acceptance ratio
        if log(random()) > logp_p + loglike_p - logp - loglike + HF:
            
            # Revert s if fail
            self.reject()
            
            # Increment rejected count
            self._rejected += 1
            if self.verbose > 0:
                print self._id + ' rejecting'
        else:
            # Increment accepted count
            self._accepted += 1
            if self.verbose > 0:
                print self._id + ' accepting'
        
        if self.verbose > 1:
            print self._id + ' returning.'
    
    def reject(self):
        # Sets current s value to the last accepted value
        self.stochastic.value = self.stochastic.last_value
    
    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self._dist is "Normal" (i.e. no proposal specified).
        """
        if self._dist == "Normal":
            self.stochastic.value = rnormal(self.stochastic.value, self._asf * self._sig)
        elif self._dist == "Prior":
            self.stochastic.random()

class Gibbs(Metropolis):
    """
    Base class for the Gibbs step methods
    """
    def __init__(self, stochastic, verbose=0):
        Metropolis.__init__(self, stochastic, verbose=verbose)

    # Override Metropolis's competence.
    competence = staticmethod(StepMethod.competence)

    def step(self):
        if not self.conjugate:
            logp = self.stochastic.logp

        self.propose()

        if not self.conjugate:

            try:
                logp_p = self.stochastic.logp
            except ZeroProbability:
                self.reject()

            if log(np.random.random()) > logp_p - logp:
                self.reject()

    def tune(self, verbose):
        return False

    def propose(self):
        raise NotImplementedError, 'The Gibbs class has to be subclassed, it is not usable directly.'


class NoStepper(StepMethod):
    """
    Step and tune methods do nothing.
    
    Useful for holding stochastics constant without setting isdata=True.
    """
    def step(self):
        pass
    def tune(self):
        pass

class DiscreteMetropolis(Metropolis):
    """
    Just like Metropolis, but rounds the variable's value.
    Good for discrete stochastics.
    """
    
    def __init__(self, stochastic, scale=1., sig=None, dist=None):
        # DiscreteMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, stochastic, scale=scale, sig=sig, dist=dist)
        
        # Initialize verbose feedback string
        self._id = stochastic.__name__
    
    @staticmethod
    def competence(stochastic):
        """
        The competence function for DiscreteMetropolis.
        """
        if stochastic.dtype in integer_dtypes:
            return 1
        else:
            return 0
    
    
    def propose(self):
        # Propose new values using normal distribution
        
        if self._dist == "Normal":
            new_val = rnormal(self.stochastic.value,self._asf * self._sig)
            self.stochastic.value = round_array(new_val)
        elif self._dist == "Prior":
            self.stochastic.random()


class BinaryMetropolis(Metropolis):
    """
    Like Metropolis, but with a modified step() method.
    Good for binary variables.
    
    """
    
    def __init__(self, stochastic, p_jump=.1, dist=None, verbose=0):
        # BinaryMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, stochastic, dist=dist, verbose=verbose)
        
        self.state.pop('_sig')
        
        # Initialize verbose feedback string
        self._id = stochastic.__name__
        
        # _asf controls the jump probability
        self._asf = log(1.-p_jump) / log(.5)
        
    @staticmethod
    def competence(stochastic):
        """
        The competence function for Binary One-At-A-Time Metropolis
        """
        if stochastic.dtype in bool_dtypes:
            return 1
        else:
            return 0

    def step(self):
        if not isscalar(self.stochastic.value):
            Metropolis.step(self)
        else:
            
            # See what log-probability of True is.
            self.stochastic.value = True
        
            try:
                logp_true = self.stochastic.logp
                loglike_true = self.loglike
            except ZeroProbability:
                self.stochastic.value = False
                return
        
            # See what log-probability of False is.
            self.stochastic.value = False
        
            try:
                logp_false = self.stochastic.logp
                loglike_false = self.loglike
            except ZeroProbability:
                self.stochastic.value = True
                return
        
            # Test
            p_true = exp(logp_true + loglike_true)
            p_false = exp(logp_false + loglike_false)
        
            # Stochastically set value according to relative
            # probabilities of True and False
            if random() > p_true / (p_true + p_false):
                self.stochastic.value = True
        
    
    def propose(self):
        # Propose new values

        if self._dist == 'Prior':
            self.stochastic.random()
        else:
            # Convert _asf to a jump probability
            p_jump = 1.-.5**self._asf
        
            rand_array = random(size=shape(self.stochastic.value))
            new_value = copy(self.stochastic.value)
            switch_locs = where(rand_array<p_jump)
            new_value[switch_locs] = True - new_value[switch_locs]
            # print switch_locs, rand_array, new_value, self.stochastic.value
            self.stochastic.value = new_value


class AdaptiveMetropolis(StepMethod):
    """
    The AdaptativeMetropolis (AM) sampling algorithm works like a regular 
    Metropolis, with the exception that stochastic parameters are block-updated 
    using a multivariate jump distribution whose covariance is tuned during 
    sampling. Although the chain is non-Markovian, i.e. the proposal 
    distribution is asymmetric, it has correct ergodic properties. See
    (Haario et al., 2001) for details. 
    
    :Parameters:
      - stochastic : PyMC objects
            Stochastic objects to be handled by the AM algorith,
            
      - cov : array
            Initial guess for the covariance matrix C. 
            
      - delay : int
          Number of steps before the empirical covariance is computed. If greedy 
          is True, the algorithm waits for delay accepted steps before computing
          the covariance.
        
      - scales : dict
          Dictionary containing the scale for each stochastic keyed by name.
          If cov is None, those scales are used to define an initial covariance
          matrix. If neither cov nor scale is given, the initial covariance is 
          guessed from the objects value (or trace if available).
    
      - interval : int
          Interval between covariance updates.
      
      - greedy : bool
          If True, only the accepted jumps are tallied in the internal trace 
          until delay is reached. This is useful to make sure that the empirical 
          covariance has a sensible structure. 
      
      - verbose : int
          Controls the verbosity level. 
   
  
    :Reference: 
      Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm,
          Bernouilli, vol. 7 (2), pp. 223-242, 2001.
    """
    def __init__(self, stochastic, cov=None, delay=1000, scales=None, interval=200, greedy=True,verbose=0):
        
        # Verbosity flag
        self.verbose = verbose
        
        if isinstance(stochastic, Stochastic):
            stochastic = [stochastic] 
        # Initialize superclass
        StepMethod.__init__(self, stochastic, verbose)
        
        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session. 
        self._state += ['_trace_count', '_current_iter', 'C', '_sig',
        '_proposal_deviate', '_trace']
        
        self._sig = None
        
        # Number of successful steps before the empirical covariance is computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy
        
        # Call methods to initialize
        self.check_type()
        self.dimension()
        self.set_cov(cov, scales)     
        self.update_sig()
        
        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy 
        # sampling can be done during warm-up period.
        self._trace_count = 0 
        self._current_iter = 0
        
        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []
        
        if self.verbose >= 1:
            print "Initialization..."
            print 'Dimension: ', self.dim
            print "C_0: ", self.C
            print "Sigma: ", self._sig

      
    @staticmethod
    def competence(stochastic):
        """
        The competence function for AdaptiveMetropolis.
        The AM algorithm is well suited to deal with multivariate
        parameters. 
        """
        if not stochastic.dtype in float_dtypes and not stochastic.dtype in integer_dtypes:
            return 0
        if np.iterable(stochastic.value):
            return 2
        else:
            return 0
                
                
    def set_cov(self, cov=None, scales={}, trace=2000, scaling=50):
        """Define C, the jump distributioin covariance matrix.
        
        Return:
            - cov,  if cov != None
            - covariance matrix built from the scales dictionary if scales!=None
            - covariance matrix estimated from the stochastics last trace values.
            - covariance matrix estimated from the stochastics value, scaled by 
                scaling parameter.
        """
        
        if cov:
            return cov
        elif scales:
            # Get array of scales
            ord_sc = self.order_scales(scales)    
            # Scale identity matrix
            self.C = np.eye(self.dim)*ord_sc
        else:
            try:
                a = self.trace2array(-trace, -1)
                nz = a[:, 0]!=0
                self.C = np.cov(a[nz, :], rowvar=0)
            except:
                ord_sc = []
                for s in self.stochastics:
                    this_value = abs(np.ravel(s.value))
                    if not this_value.any():
                        this_value = [1.]
                    for elem in this_value:
                        ord_sc.append(elem)
                # print len(ord_sc), self.dim
                self.C = np.eye(self.dim)*ord_sc/scaling
            
        
    def check_type(self):
        """Make sure each stochastic has a correct type, and identify discrete stochastics."""
        self.isdiscrete = {}
        for stochastic in self.stochastics:
            if stochastic.dtype in integer_dtypes:
                self.isdiscrete[stochastic] = True
            elif stochastic.dtype in bool_dtypes:
                raise 'Binary stochastics not supported by AdaptativeMetropolis.'
            else:
                self.isdiscrete[stochastic] = False
                
                
    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stochastic.
        """
        self.dim = 0
        self._slices = {}
        for stochastic in self.stochastics:
            if isinstance(stochastic.value, np.ndarray):
                p_len = len(stochastic.value.ravel())
            else:
                p_len = 1
            self._slices[stochastic] = slice(self.dim, self.dim + p_len)
            self.dim += p_len
            
            
    def order_scales(self, scales):
        """Define an array of scales to build the initial covariance.
        If init_scales is None, the scale is taken to be the initial value of 
        the stochastics.
        """
        ord_sc = []
        for stochastic in self.stochastics:
            ord_sc.append(scales[stochastic])
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
        self.C, self.chain_mean = self.recursive_cov(self.C, self._trace_count, 
            self.chain_mean, chain, scaling=scaling, epsilon=epsilon)
        
        if self.verbose > 0:
            print "\tUpdating covariance ...\n", self.C
            print "\tUpdating mean ... ", self.chain_mean
        
        # Update state
        try:
            self.update_sig()
        except np.linalg.LinAlgError:
            self.covariance_adjustment(.9)

        self._trace_count += len(self._trace)
        self._trace = []  
        
    def covariance_adjustment(self, f=.9):
        """Multiply self._sig by a factor f. This is useful when the current _sig is too large and all jumps are rejected.
        """
        warnings.warn('covariance was not positive definite. _sig cannot be computed and next jumps will be based on the last valid value. This might mean that no jumps were accepted. In this case, it might be worthwhile to specify an initial covariance matrix with smaller variance. For the moment, _sig will be artificially reduced by a factor .9 each time this happens.')
        self._sig *= f
	
    def update_sig(self):
        """Compute the Cholesky decomposition of self.C."""
        self._sig = np.linalg.cholesky(self.C)
    
              
    def recursive_cov(self, cov, length, mean, chain, scaling=1, epsilon=0):
        r"""Compute the covariance recursively.
        
        Return the new covariance and the new mean. 
        
        .. math::
            C_k & = \frac{1}{k-1} (\sum_{i=1}^k x_i x_i^T - k\bar{x_k}\bar{x_k}^T)
            C_n & = \frac{1}{n-1} (\sum_{i=1}^k x_i x_i^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)
                & = \frac{1}{n-1} ((k-1)C_k + k\bar{x_k}\bar{x_k}^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)
                
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
        t5 = epsilon * np.eye(cov.shape[0])
        
        new_cov =  (k-1)/(n-1.)*cov + scaling/(n-1.) * (t2 + t3 - t4 + t5)
        return new_cov, new_mean
        
    def recursive_mean(self, mean, length, chain):
        r"""Compute the chain mean recursively.
        
        Instead of computing the mean :math:`\bar{x_n}` of the entire chain, 
        use the last computed mean :math:`bar{x_j}` and the tail of the chain 
        to recursively estimate the mean. 
        
        .. math::
            \bar{x_n} & = \frac{1}{n} \sum_{i=1}^n x_i
                      & = \frac{1}{n} (\sum_{i=1}^j x_i + \sum_{i=j+1}^n x_i)
                      & = \frac{j\bar{x_j}}{n} + \frac{\sum_{i=j+1}^n x_i}{n}
        
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
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.
        
        The proposal jumps are drawn from a multivariate normal distribution.        
        """
        
        arrayjump = np.dot(self._sig, np.random.normal(size=self._sig.shape[0]))
        
        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]]
            if np.shape(stochastic.value):
                jump = np.reshape(arrayjump[self._slices[stochastic]],np.shape(stochastic.value))
            if self.isdiscrete[stochastic]:
                stochastic.value = stochastic.value + round_array(jump)
            else:
                stochastic.value = stochastic.value + jump
                
                
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
        
        # Probability and likelihood for stochastic's current value:
        logp = sum([stochastic.logp for stochastic in self.stochastics])
        loglike = self.loglike
        
        # Sample a candidate value              
        self.propose()
        
        # Metropolis acception/rejection test
        accept = False
        try:
            # Probability and likelihood for stochastic's proposed value:
            logp_p = sum([stochastic.logp for stochastic in self.stochastics])
            loglike_p = self.loglike
            if np.log(random()) < logp_p + loglike_p - logp - loglike:
                accept = True
                self._accepted += 1
            else:
                self._rejected += 1
        except ZeroProbability:
            self._rejected += 1
            logp_p = None
            loglike_p = None
            
        if self.verbose > 2:
            print "Step ", self._current_iter
            print "\tLogprobability (current, proposed): ", logp, logp_p
            print "\tloglike (current, proposed):      : ", loglike, loglike_p
            for stochastic in self.stochastics:
                print "\t", stochastic.__name__, stochastic.last_value, stochastic.value
            if accept:
                print "\tAccepted\t*******\n"
            else: 
                print "\tRejected\n"
            print "\tAcceptance ratio: ", self._accepted/(self._accepted+self._rejected)
            
        if self._current_iter == self.delay: 
            self.greedy = False
            
        if not accept:
            self.reject()
        
        if accept or not self.greedy:
            self.internal_tally()

        if self._current_iter>self.delay and self._current_iter%self.interval==0:
           self.update_cov()
    
        self._current_iter += 1
    
    # Please keep reject() factored out- helps RandomRealizations figure out what to do.
    def reject(self):
        for stochastic in self.stochastics:
            stochastic.value = stochastic.last_value
    
    def internal_tally(self):
        """Store the trace of stochastics for the computation of the covariance.
        This trace is completely independent from the backend used by the 
        sampler to store the samples."""
        chain = []
        for stochastic in self.stochastics:
            chain.append(np.ravel(stochastic.value))
        self._trace.append(np.concatenate(chain))
        
    def trace2array(self, i0, i1):
        """Return an array with the trace of all stochastics from index i0 to i1."""
        chain = []
        for stochastic in self.stochastics:
            chain.append(ravel(stochastic.trace.gettrace(slicing=slice(i0,i1))))
        return concatenate(chain)
        
    def tune(self, verbose):
        """Tuning is done during the entire run, independently from the Sampler 
        tuning specifications. """
        return False

# class JointMetropolis(StepMethod):
#     """
#     JointMetropolis will be superseded by AdaptiveMetropolis.
#     
#     S = Joint(variables, epoch=1000, memory=10, delay=1000)
#     
#     Applies the Metropolis-Hastings algorithm to several variables
#     together. Jumping density is a multivariate normal distribution
#     with mean zero and covariance equal to the empirical covariance
#     of the variables, times _asf ** 2.
#     
#     :Arguments:
#     - variables (optional) : list or array
#             A sequence of pymc objects to handle using
#             this StepMethod.
#     
#     - s (optional) : Stochastic
#             Alternatively to variables, a single variable can be passed.
#     
#     - epoch (optional) : integer
#             After epoch values are stored in the internal
#             traces, the covariance is recomputed.
#     
#     - memory (optional) : integer
#             The maximum number of epochs to consider when
#             computing the covariance.
#     
#     - delay (optional) : integer
#             Number of one-at-a-time iterations to do before
#             starting to record values for computing the joint
#             covariance.
#     
#     - dist (optional) : string
#             The proposal distribution. May be 'Normal', 'RoundedNormal', 'Bernoulli',
#             'Prior' or None. If None is provided, a proposal distribution is
#             chosen by examining P.value's type.
#     
#     - scale (optional) : float
#             Scale proposal distribution.
#     
#     - oneatatime_scales (optional) : dict
#             Dictionary of scales for one-at-a-time iterations.
#     
#     - verbose (optional) : integer
#             Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
#     
#     Externally-accessible attributes:
#         
#         variables:      A sequence of sastic variables to handle using
#                         this StepMethod.
#         
#         epoch:          After epoch values are stored in the internal
#                         traces, the covariance is recomputed.
#         
#         memory:         The maximum number of epochs to consider when
#                         computing the covariance.
#         
#         delay:          Number of one-at-a-time iterations to do before
#                         starting to record values for computing the joint
#                         covariance.
#         
#         _asf:           Adaptive scale factor.
#     
#     Externally-accessible methods:
#         
#         step():         Make a Metropolis step. Applies the one-at-a-time
#                         Metropolis algorithm until the first time the
#                         covariance is computed, then applies the joint
#                         Metropolis algorithm.
#         
#         tune():         sets _asf according to a heuristic.
#         
#     Also: don't start joint sampling until all variables have mixed at
#     least a little. Warn if another epoch of one-at-a-time sampling is
#     required.
#     
#     Also: when the covariance is nonsquare,
#     
#     """
#     
#     
#     def __init__(self, variables=None, s=None, epoch=1000, memory=10, delay = 0, scale=.1, oneatatime_scales=None, verbose=0):
#         print "This StepMethod is deprecated and will be removed from future versions, please use AdaptiveMetropolis."
#         self.verbose = verbose
#         
#         if s is not None:
#             variables = [s]
#         StepMethod.__init__(self, variables, verbose=verbose)
#         
#         self.epoch = epoch
#         self.memory = memory
#         self.delay = delay
#         self._id = ''.join([p.__name__ for p in self.stochastics])
#         self.isdiscrete = {}
#         self.scale = scale
#         
#         # Flag indicating whether covariance has been computed
#         self._ready = False
#         
#         # For making sure the covariance isn't recomputed multiple times
#         # on the same trace index
#         self.last_trace_index = 0
#         
#         # Use Metropolis instances to handle independent jumps
#         # before first epoch is complete
#         if self.verbose > 2:
#             print self._id + ': Assigning single-stochastic handlers.'
#         self._single_stoch_handlers = set()
#         
#         for s in self.stochastics:
#             if oneatatime_scales is not None:
#                 scale_now = oneatatime_scales[s]
#             else:
#                 scale_now = None
#             
#             new_method = assign_method(s, scale_now)
#             self._single_stoch_handlers.add(new_method)
#         StepMethodRegistry[JointMetropolis] = Jointcompetence
# 
#         
#         # Allocate memory for internal traces and get stoch slices
#         self._slices = {}
#         self._len = 0
#         for s in self.stochastics:
#             if isinstance(s.value, ndarray):
#                 stoch_len = len(s.value.ravel())
#             else:
#                 stoch_len = 1
#             self._slices[s] = slice(self._len, self._len + stoch_len)
#             self._len += stoch_len
#         
#         self._proposal_deviate = zeros(self._len,dtype=float)
#         
#         self._trace = zeros((self._len, self.memory * self.epoch),dtype=float)
#         
#         # __init__ should also check that each stoch's value is an ndarray or
#         # a numerical type.
#         
#         self._state += ['last_trace_index', '_cov', '_sig',
#         '_proposal_deviate', '_trace']
# 
#     
#     @staticmethod
#     def competence(s):
#         """
#         The competence function for JointMetropolis. Currently returning 0,
#         because JointMetropolis may be buggy.
#         """
#         
#         return 0
#         
#         if isinstance(s, DiscreteStochastic) or isinstance(s, BinaryStochastic):
#             # If the stoch's binary or discrete, I can't do it.
#             return 0
#         elif isinstance(s.value, ndarray):
#             return 0
#         else:
#             return 0
#     
#     
#     def compute_sig(self):
#         """
#         This method computes and stores the matrix square root of the empirical
#         covariance every epoch.
#         """
#         
#         if self.verbose > 1:
#             print 'Joint StepMethod ' + self._id + ' computing covariance.'
#         
#         # Figure out which slice of the traces to use
#         if (self._model._cur_trace_index - self.delay) / self.epoch > self.memory:
#             trace_slice = slice(self._model._cur_trace_index-self.epoch * self.memory,\
#                                 self._model._cur_trace_index)
#             trace_len = self.memory * self.epoch
#         else:
#             trace_slice = slice(self.delay, self._model._cur_trace_index)
#             trace_len = (self._model._cur_trace_index - self.delay)
# 
#         
#         # Store all the stochastics' traces in self._trace
#         for s in self.stochastics:
#             stoch_trace = s.trace(slicing=trace_slice)
#             
#             # If s is an array, ravel each tallied value
#             if isinstance(s.value, ndarray):
#                 for i in range(trace_len):
#                     self._trace[self._slices[s], i] = stoch_trace[i,:].ravel()
#             
#             # If s is a scalar, there's no need.
#             else:
#                 self._trace[self._slices[s], :trace_len] = stoch_trace
#         
#         # Compute matrix square root of covariance of self._trace
#         self._cov = cov(self._trace[: , :trace_len])
#         
#         self._sig = msqrt(self._cov).T * self.scale
#         
#         self._ready = True
# 
#     
#     def tune(self, divergence_threshold = 1e10, verbose=0):
#         """
#         If the empirical covariance hasn't been computed yet (the first
#         epoch isn't over), this method passes the tune() call along to the
#         Metropolis instances handling self's variables. If the
#         empirical covariance has been computed, the Metropolis
#         instances aren't in use anymore so this method does nothing.
#         
#         We may want to make this method do something eventually.
#         """
#         
#         if not self._accepted > 0 or self._rejected > 0:
#             for handler in self._single_stoch_handlers:
#                 handler.tune(divergence_threshold, verbose)
#         # This has to return something... please check this. DH
#         return False 
#     
#     def propose(self):
#         """
#         This method proposes values for self's variables based on the empirical
#         covariance.
#         """
#         fill_stdnormal(self._proposal_deviate)
#         
#         N = self._sig.shape[1]
#         
#         proposed_vals = inner(self._proposal_deviate[:N], self._asf * self._sig)
#         
#         for s in self.stochastics:
#             
#             jump = reshape(proposed_vals[self._slices[s]],shape(s.value))
#             
#             s.value = s.value + jump
#     
#     def reject(self):
#         """Reject a jump."""
#         for s in self.stochastics:
#             s.value = s.last_value
#     
#     def step(self):
#         """
#         If the empirical covariance hasn't been computed yet, the step() call
#         is passed along to the Metropolis instances that handle self's
#         variables before the end of the first epoch.
#         
#         If the empirical covariance has been computed, values for self's variables
#         are proposed and tested simultaneously.
#         """
#         if not self._ready:
#             for handler in self._single_stoch_handlers:
#                 handler.step()
#         else:
#             # Probability and likelihood for s's current value:
#             logp = sum([s.logp for s in self.stochastics])
#             loglike = self.loglike
#             
#             # Sample a candidate value
#             self.propose()
#             
#             # Probability and likelihood for s's proposed value:
#             try:
#                 logp_p = sum([s.logp for s in self.stochastics])
#             except ZeroProbability:
#                 self._rejected += 1
#                 self.reject()
#                 return
#             
#             loglike_p = self.loglike
#             
#             # Test
#             if log(random()) > logp_p + loglike_p - logp - loglike:
#                 # Revert stochastic if fail
#                 self._rejected += 1
#                 self.reject()
#             else:
#                 self._accepted += 1
#         
#         # If an epoch has passed, recompute covariance.
#         if  (float(self._model._cur_trace_index - self.delay)) % self.epoch == 0 \
#             and self._model._cur_trace_index > self.delay \
#             and not self._model._cur_trace_index == self.last_trace_index:
#             
#             # Make sure all the one-at-a-time handlers mixed.
#             if not self._ready:
#                 for handler in self._single_stoch_handlers:
#                     if handler._accepted == 0:
#                         print self._id+ ": Warnining, stochastic " + handler.s.__name__ + " did not mix, continuing one-at-a-time sampling"
#                 
#                 self.compute_sig()
#                 self.last_trace_index = self._model._cur_trace_index
# 
# # JointMetropolis.competence = StepMethod.competence
