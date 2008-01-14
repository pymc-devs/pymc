__docformat__='reStructuredText'
from utils import msqrt, check_type, round_array
from numpy import ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any, array, all, abs, exp, where, isscalar
from numpy.linalg.linalg import LinAlgError
from numpy.random import randint, random
from numpy.random import normal as rnormal
from flib import fill_stdnormal
from PyMCObjects import Stochastic, Deterministic, Node, DiscreteStochastic, BinaryStochastic, Potential
from Node import ZeroProbability, Node, Variable, StepMethodBase, StochasticBase
from PyMC.decorators import prop
from copy import copy
# Changeset history
# 22/03/2007 -DH- Added a _state attribute containing the name of the attributes that make up the state of the step method, and a method to return that state in a dict. Added an id.
# TODO: Test cases for binary and discrete Metropolises.

__all__=['DiscreteMetropolis', 'JointMetropolis', 'Metropolis', 'StepMethod', 'assign_method',  'pick_best_methods', 'StepMethodRegistry', 'NoStepper', 'BinaryMetropolis']

StepMethodRegistry = []

def pick_best_methods(stoch):
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
        competence = method.competence(stoch)
        
        # If better than current best method, promote it
        if competence > max_competence:
            best_candidates = set([method])
            max_competence = competence
        
        # If same competence, add it to the set of best methods
        elif competence == max_competence:
            best_candidates.add(method)
    
    # print stoch.__name__ + ': ', best_candidates, ' ', max_competence
    return best_candidates

def assign_method(stoch, scale=None):
    """
    Returns a step method instance to handle a
    variable. If several methods have the same competence,
    it picks one arbitrarily (using set.pop()).
    """
    
    # Retrieve set of best candidates
    best_candidates = pick_best_methods(stoch)
    
    # Randomly grab and appropriate method
    method = best_candidates.pop()
    
    if scale:
        return method(stoch = stoch, scale = scale)
    
    return method(stoch = stoch)


class StepMethodMeta(type):
    """
    Automatically registers new step methods.
    """
    def __init__(cls, name, bases, dict):
        type.__init__(cls)
        StepMethodRegistry.append(cls)
        
class StepMethod(StepMethodBase):
    """
    This object knows how to make Stochastics take single MCMC steps.
    It's sample() method will be called by Model at every MCMC iteration.
    
    :Parameters:
          -variables : list, array or set
            Collection of PyMCObjects
          
          - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
    
    Externally-accessible attributes:
      dtrms:    The Deterministics over which self has jurisdiction.
      stochs:   The Stochastics over which self has jurisdiction which have isdata = False.
      data:     The Stochastics over which self has jurisdiction which have isdata = True.
      variables:The Deterministics and Stochastics over which self has jurisdiction.
      children: The combined children of all Variables over which self has jurisdiction.
      parents:  The combined parents of all Nodes over which self has jurisdiction, as a set.
      loglike:  The summed log-probability of self's children conditional on all of self's 
                  Variables' current values. These will be recomputed only as necessary. 
                  This descriptor should eventually be written in C.
    
    Externally accesible methods:
      sample(): A single MCMC step for all the Stochastics over which self has 
        jurisdiction. Must be overridden in subclasses.
      tune(): Tunes proposal distribution widths for all self's Stochastics.
      competence(stoch): Examines Stochastic instance stoch and returns self's 
        competence to handle it, on a scale of 0 to 3.
    
    To instantiate a StepMethod called S with jurisdiction over a
    sequence/set N of Nodes:
      
      >>> S = StepMethod(N)
    
    :SeeAlso: Metropolis, Sampler.
    """
    
    __metaclass__ = StepMethodMeta
    
    def __init__(self, variables, verbose=0):
        # StepMethod initialization
        
        # Initialize public attributes
        self.variables = set(variables)
        self.dtrms = set()
        self.stochs = set()
        self.data_stochs = set()
        self.children = set()
        self.parents = set()
        
        # Initialize hidden attributes
        self._asf = 1.
        self._accepted = 0.
        self._rejected = 0.
        self._state = ['_rejected', '_accepted', '_asf']
        self.verbose = verbose
        
        # File away the variables
        for variable in self.variables:
            
            # Sort.
            if isinstance(variable,Deterministic):
                self.dtrms.add(variable)
            elif isinstance(variable,Stochastic):
                if variable.isdata:
                    self.data_stochs.add(variable)
                else:
                    self.stochs.add(variable)
        
        # Find children, no need to find parents; each variable takes care of those.
        for variable in self.variables:
            self.children |= variable.children
            for parent in variable.parents.itervalues():
                if isinstance(parent, Variable):
                    self.parents.add(parent)
        
        self.children = set([])
        self.parents = set([])
        for stoch in self.stochs:
            self.children |= stoch.extended_children
            self.parents |= stoch.extended_parents
        
        # Remove own stochastics from children and parents.
        self.children -= self.stochs
        self.parents -= self.stochs
        
        # ID string for verbose feedback
        self._id = 'To define in subclasses'
    
    def step(self):
        """
        Specifies single step of step method.
        Must be overridden in subclasses.
        """
        pass

    @staticmethod
    def competence(stoch):
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
            def competence(stoch):
                if isinstance(stoch, MyStochasticSubclass):
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
        if verbose > 1 or self.verbose > 1:
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
        # Warning: self.stoch is not defined above. The following assumes
        # that the class has created this value, which is a bit fragile. DH
        if verbose > 1 or self.verbose > 1:
            print '\t\tvalue:', self.stoch.value
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
            if self.verbose > 2:
                print '\t'+self._id+' Getting log-probability from child ' + child.__name__
            
            # Increment sum
            sum += child.logp
        
        return sum
    
    # Make get property for retrieving log-probability
    loglike = property(fget = _get_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochs, with self.stochs removed.")
    
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

# The default StepMethod, which Model uses to handle singleton stochs.
class Metropolis(StepMethod):
    """
    The default StepMethod, which Model uses to handle singleton, continuous variables.
    
    Applies the one-at-a-time Metropolis-Hastings algorithm to the Stochastic over which self has jurisdiction.
    
    To instantiate a Metropolis called M with jurisdiction over a Stochastic P:
      
      >>> M = Metropolis(P, scale=1, dist=None)
    
    :Arguments:
    - stoch : Stochastic
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
    
    def __init__(self, stoch, scale=1., dist=None, verbose=0):
        # Metropolis class initialization
        
        # Initialize superclass
        StepMethod.__init__(self, [stoch], verbose=verbose)
        
        # Set public attributes
        self.stoch = stoch
        self.verbose = verbose
        
        # Avoid zeros when setting proposal variance
        if all(self.stoch.value != 0.):
            self.proposal_sig = ones(shape(self.stoch.value)) * abs(self.stoch.value) * scale
        else:
            self.proposal_sig = ones(shape(self.stoch.value)) * scale
        
        # Initialize proposal deviate with array of zeros
        self.proposal_deviate = zeros(shape(self.stoch.value), dtype=float)
        
        # Initialize verbose feedback string
        self._id = stoch.__name__
        
        # Determine size of stoch
        if isinstance(self.stoch.value, ndarray):
            self._len = len(self.stoch.value.ravel())
        else:
            self._len = 1
        
        # If no dist argument is provided, assign a proposal distribution automatically.
        if not dist:
            
            # Pick Gaussian, just because
            self._dist = "Normal"
            
            # If self's extended children has no stochs, proposing from the prior is best.
            if sum([isinstance(child, StochasticBase) for child in self.children]) == 0:
                try:
                    self.stoch.random()
                    self._dist = "Prior"
                except:
                    pass
        
        else:
            self._dist = dist
    
    @staticmethod
    def competence(stoch):
        """
        The competence function for Metropolis
        """

        if isinstance(stoch, DiscreteStochastic) or isinstance(stoch, BinaryStochastic):
            # If the stoch's binary or discrete, I can't do it.
            return 0

        else:
            # If the stoch's value is an ndarray or a number, I can do it,
            # but not necessarily particularly well.
            _type = check_type(stoch)[0]
            if _type in [float, int]:
                return 1
            else:
                return 0
    
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
        
        # Probability and likelihood for stoch's current value:
        
        if self.verbose > 2:
            print
            print self._id + ' getting initial prior.'
        
        if self._dist == "Prior":
            # No children
            logp = 0.
        else:
            logp = self.stoch.logp
        
        if self.verbose > 2:
            print self._id + ' getting initial likelihood.'
        loglike = self.loglike
        
        if self.verbose > 2:
            print self._id + ' proposing.'
        
        # Sample a candidate value
        self.propose()
        
        # Probability and likelihood for stoch's proposed value:
        try:
            if self._dist == "Prior":
                logp_p = 0.
            else:
                logp_p = self.stoch.logp
            loglike_p = self.loglike
        
        except ZeroProbability:
            
            # Reject proposal
            if self.verbose > 2:
                print self._id + ' rejecting due to ZeroProbability.'
            self.reject()
            
            # Increment rejected count
            self._rejected += 1
            
            if self.verbose > 2:
                print self._id + ' returning.'
            return
        
        if self.verbose > 2:
            print 'logp_p - logp: ', logp_p - logp
            print 'loglike_p - loglike: ', loglike_p - loglike
        
        HF = self.hastings_factor()
        
        # Evaluate acceptance ratio
        if log(random()) > logp_p + loglike_p - logp - loglike + HF:
            
            # Revert stoch if fail
            self.reject()
            
            # Increment rejected count
            self._rejected += 1
            if self.verbose > 2:
                print self._id + ' rejecting'
        else:
            # Increment accepted count
            self._accepted += 1
            if self.verbose > 2:
                print self._id + ' accepting'
        
        if self.verbose > 2:
            print self._id + ' returning.'
    
    def reject(self):
        # Sets current stoch value to the last accepted value
        self.stoch.value = self.stoch.last_value
    
    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self._dist is "Normal" (i.e. no proposal specified).
        """
        if self._dist == "Normal":
            self.stoch.value = rnormal(self.stoch.value, self._asf * self.proposal_sig)
        elif self._dist == "Prior":
            self.stoch.random()

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
    Good for DiscreteStochastics.
    """
    
    def __init__(self, stoch, scale=1., dist=None):
        # DiscreteMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, stoch, scale=scale, dist=dist)
        
        # Initialize verbose feedback string
        self._id = stoch.__name__
    
    @staticmethod
    def competence(stoch):
        """
        The competence function for DiscreteMetropolis.
        """
        if isinstance(stoch, DiscreteStochastic):
            return 1
        else:
            return 0
    
    
    def propose(self):
        # Propose new stoch values using normal distribution
        
        if self._dist == "Normal":
            new_val = rnormal(self.stoch.value,self._asf * self.proposal_sig)
            self.stoch.value = round_array(new_val)
        elif self._dist == "Prior":
            self.stoch.random()


class BinaryMetropolis(Metropolis):
    """
    Like Metropolis, but with a modified step() method.
    Good for binary variables.
    
    NOTE this is not compliant with the Metropolis standard
    yet because it lacks a reject() method.
    (??? But, it is a subclass of Metropolis, which has a reject() method)
    True... but it's never called, this is really a Gibbs sampler since there
    are only 2 states available.
    
    This should be a subclass of Gibbs, not Metropolis.
    """
    
    def __init__(self, stoch, p_jump=.1, dist=None, verbose=0):
        # BinaryMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, stoch, dist=dist, verbose=verbose)
        
        # Initialize verbose feedback string
        self._id = stoch.__name__
        
        # _asf controls the jump probability
        self._asf = log(1.-p_jump) / log(.5)
        
    @staticmethod
    def competence(stoch):
        """
        The competence function for Binary One-At-A-Time Metropolis
        """
        if isinstance(stoch, BinaryStochastic):
            return 1
        else:
            return 0

    def step(self):
        if not isscalar(self.stoch.value):
            Metropolis.step(self)
        else:
            
            # See what log-probability of True is.
            self.stoch.value = True
        
            try:
                logp_true = self.stoch.logp
                loglike_true = self.loglike
            except ZeroProbability:
                self.stoch.value = False
                return
        
            # See what log-probability of False is.
            self.stoch.value = False
        
            try:
                logp_false = self.stoch.logp
                loglike_false = self.loglike
            except ZeroProbability:
                self.stoch.value = True
                return
        
            # Test
            p_true = exp(logp_true + loglike_true)
            p_false = exp(logp_false + loglike_false)
        
            # Stochastically set value according to relative
            # probabilities of True and False
            if random() > p_true / (p_true + p_false):
                self.stoch.value = True
        
    
    def propose(self):
        # Propose new values

        if self._dist == 'Prior':
            self.stoch.random()
        else:
            # Convert _asf to a jump probability
            p_jump = 1.-.5**self._asf
        
            rand_array = random(size=shape(self.stoch.value))
            new_value = copy(self.stoch.value)
            switch_locs = where(rand_array<p_jump)
            new_value[switch_locs] = True - new_value[switch_locs]
            # print switch_locs, rand_array, new_value, self.stoch.value
            self.stoch.value = new_value




class JointMetropolis(StepMethod):
    """
    JointMetropolis will be superseded by AdaptiveMetropolis.
    
    S = Joint(variables, epoch=1000, memory=10, delay=1000)
    
    Applies the Metropolis-Hastings algorithm to several variables
    together. Jumping density is a multivariate normal distribution
    with mean zero and covariance equal to the empirical covariance
    of the variables, times _asf ** 2.
    
    :Arguments:
    - variables (optional) : list or array
            A sequence of pymc objects to handle using
            this StepMethod.
    
    - stoch (optional) : Stochastic
            Alternatively to variables, a single variable can be passed.
    
    - epoch (optional) : integer
            After epoch values are stored in the internal
            traces, the covariance is recomputed.
    
    - memory (optional) : integer
            The maximum number of epochs to consider when
            computing the covariance.
    
    - delay (optional) : integer
            Number of one-at-a-time iterations to do before
            starting to record values for computing the joint
            covariance.
    
    - dist (optional) : string
            The proposal distribution. May be 'Normal', 'RoundedNormal', 'Bernoulli',
            'Prior' or None. If None is provided, a proposal distribution is
            chosen by examining P.value's type.
    
    - scale (optional) : float
            Scale proposal distribution.
    
    - oneatatime_scales (optional) : dict
            Dictionary of scales for one-at-a-time iterations.
    
    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
    
    Externally-accessible attributes:
        
        variables:      A sequence of stochastic variables to handle using
                        this StepMethod.
        
        epoch:          After epoch values are stored in the internal
                        traces, the covariance is recomputed.
        
        memory:         The maximum number of epochs to consider when
                        computing the covariance.
        
        delay:          Number of one-at-a-time iterations to do before
                        starting to record values for computing the joint
                        covariance.
        
        _asf:           Adaptive scale factor.
    
    Externally-accessible methods:
        
        step():         Make a Metropolis step. Applies the one-at-a-time
                        Metropolis algorithm until the first time the
                        covariance is computed, then applies the joint
                        Metropolis algorithm.
        
        tune():         sets _asf according to a heuristic.
        
    Also: don't start joint sampling until all variables have mixed at
    least a little. Warn if another epoch of one-at-a-time sampling is
    required.
    
    Also: when the covariance is nonsquare,
    
    """
    
    
    def __init__(self, variables=None, stoch=None, epoch=1000, memory=10, delay = 0, scale=.1, oneatatime_scales=None, verbose=0):
        
        self.verbose = verbose
        
        if stoch is not None:
            variables = [stoch]
        StepMethod.__init__(self, variables, verbose=verbose)
        
        self.epoch = epoch
        self.memory = memory
        self.delay = delay
        self._id = ''.join([p.__name__ for p in self.stochs])
        self.isdiscrete = {}
        self.scale = scale
        
        # Flag indicating whether covariance has been computed
        self._ready = False
        
        # For making sure the covariance isn't recomputed multiple times
        # on the same trace index
        self.last_trace_index = 0
        
        # Use Metropolis instances to handle independent jumps
        # before first epoch is complete
        if self.verbose > 2:
            print self._id + ': Assigning single-stoch handlers.'
        self._single_stoch_handlers = set()
        
        for stoch in self.stochs:
            if oneatatime_scales is not None:
                scale_now = oneatatime_scales[stoch]
            else:
                scale_now = None
            
            new_method = assign_method(stoch, scale_now)
            self._single_stoch_handlers.add(new_method)
        StepMethodRegistry[JointMetropolis] = Jointcompetence

        
        # Allocate memory for internal traces and get stoch slices
        self._slices = {}
        self._len = 0
        for stoch in self.stochs:
            if isinstance(stoch.value, ndarray):
                stoch_len = len(stoch.value.ravel())
            else:
                stoch_len = 1
            self._slices[stoch] = slice(self._len, self._len + stoch_len)
            self._len += stoch_len
        
        self._proposal_deviate = zeros(self._len,dtype=float)
        
        self._trace = zeros((self._len, self.memory * self.epoch),dtype=float)
        
        # __init__ should also check that each stoch's value is an ndarray or
        # a numerical type.
        
        self._state += ['last_trace_index', '_cov', '_sig',
        '_proposal_deviate', '_trace']

    
    @staticmethod
    def competence(stoch):
        """
        The competence function for JointMetropolis. Currently returning 0,
        because JointMetropolis may be buggy.
        """
        
        return 0
        
        if isinstance(stoch, DiscreteStochastic) or isinstance(stoch, BinaryStochastic):
            # If the stoch's binary or discrete, I can't do it.
            return 0
        elif isinstance(stoch.value, ndarray):
            return 2
        else:
            return 0
    
    
    def compute_sig(self):
        """
        This method computes and stores the matrix square root of the empirical
        covariance every epoch.
        """
        
        if self.verbose > 1:
            print 'Joint StepMethod ' + self._id + ' computing covariance.'
        
        # Figure out which slice of the traces to use
        if (self._model._cur_trace_index - self.delay) / self.epoch > self.memory:
            trace_slice = slice(self._model._cur_trace_index-self.epoch * self.memory,\
                                self._model._cur_trace_index)
            trace_len = self.memory * self.epoch
        else:
            trace_slice = slice(self.delay, self._model._cur_trace_index)
            trace_len = (self._model._cur_trace_index - self.delay)

        
        # Store all the stochs' traces in self._trace
        for stoch in self.stochs:
            stoch_trace = stoch.trace(slicing=trace_slice)
            
            # If stoch is an array, ravel each tallied value
            if isinstance(stoch.value, ndarray):
                for i in range(trace_len):
                    self._trace[self._slices[stoch], i] = stoch_trace[i,:].ravel()
            
            # If stoch is a scalar, there's no need.
            else:
                self._trace[self._slices[stoch], :trace_len] = stoch_trace
        
        # Compute matrix square root of covariance of self._trace
        self._cov = cov(self._trace[: , :trace_len])
        
        self._sig = msqrt(self._cov).T * self.scale
        
        self._ready = True

    
    def tune(self, divergence_threshold = 1e10, verbose=0):
        """
        If the empirical covariance hasn't been computed yet (the first
        epoch isn't over), this method passes the tune() call along to the
        Metropolis instances handling self's variables. If the
        empirical covariance has been computed, the Metropolis
        instances aren't in use anymore so this method does nothing.
        
        We may want to make this method do something eventually.
        """
        
        if not self._accepted > 0 or self._rejected > 0:
            for handler in self._single_stoch_handlers:
                handler.tune(divergence_threshold, verbose)
        # This has to return something... please check this. DH
        return False 
    
    def propose(self):
        """
        This method proposes values for self's variables based on the empirical
        covariance.
        """
        fill_stdnormal(self._proposal_deviate)
        
        N = self._sig.shape[1]
        
        proposed_vals = inner(self._proposal_deviate[:N], self._asf * self._sig)
        
        for stoch in self.stochs:
            
            jump = reshape(proposed_vals[self._slices[stoch]],shape(stoch.value))
            
            stoch.value = stoch.value + jump
    
    def reject(self):
        """Reject a jump."""
        for stoch in self.stochs:
            stoch.value = stoch.last_value
    
    def step(self):
        """
        If the empirical covariance hasn't been computed yet, the step() call
        is passed along to the Metropolis instances that handle self's
        variables before the end of the first epoch.
        
        If the empirical covariance has been computed, values for self's variables
        are proposed and tested simultaneously.
        """
        if not self._ready:
            for handler in self._single_stoch_handlers:
                handler.step()
        else:
            # Probability and likelihood for stoch's current value:
            logp = sum([stoch.logp for stoch in self.stochs])
            loglike = self.loglike
            
            # Sample a candidate value
            self.propose()
            
            # Probability and likelihood for stoch's proposed value:
            try:
                logp_p = sum([stoch.logp for stoch in self.stochs])
            except ZeroProbability:
                self._rejected += 1
                self.reject()
                return
            
            loglike_p = self.loglike
            
            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                # Revert stoch if fail
                self._rejected += 1
                self.reject()
            else:
                self._accepted += 1
        
        # If an epoch has passed, recompute covariance.
        if  (float(self._model._cur_trace_index - self.delay)) % self.epoch == 0 \
            and self._model._cur_trace_index > self.delay \
            and not self._model._cur_trace_index == self.last_trace_index:
            
            # Make sure all the one-at-a-time handlers mixed.
            if not self._ready:
                for handler in self._single_stoch_handlers:
                    if handler._accepted == 0:
                        print self._id+ ": Warnining, stoch " + handler.stoch.__name__ + " did not mix, continuing one-at-a-time sampling"
                
                self.compute_sig()
                self.last_trace_index = self._model._cur_trace_index

# JointMetropolis.competence = StepMethod.competence
