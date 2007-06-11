__docformat__='reStructuredText'
from utils import msqrt, extend_children, check_type, round_array, extend_parents
from numpy import ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any, array, all, abs 
from numpy.linalg.linalg import LinAlgError
from numpy.random import randint, random
from numpy.random import normal as rnormal
from flib import fill_stdnormal
from PyMCObjects import Parameter, Node, PyMCBase, DiscreteParameter, BinaryParameter
from PyMCBase import ZeroProbability

# Changeset history
# 22/03/2007 -DH- Added a _state attribute containing the name of the attributes that make up the state of the sampling method, and a method to return that state in a dict. Added an id.

# TODO: Actually use _asf

class DictWithDoc(dict):
    """
    The sampling method registry is a dictionary mapping each
    sampling method to its competence function. Competence
    functions must be of the form
    
    c = competence(parameter),
    
    where parameter is a Parameter object. c should be a competence 
    score from 0 to 3, assigned as follows:

    0:  I can't handle that parameter.
    1:  I can handle that parameter, but I'm a generalist and
        probably shouldn't be your top choice (Metropolis
        and friends fall into this category).
    2:  I'm designed for this type of situation, but I could be 
        more specialized.
    3:  I was made for this situation, let me handle the parameter.
    
    In order to be eligible for inclusion in the registry, a sampling
    method's init method must work with just a single argument, a
    Parameter object.
    
    :SeeAlso: blacklist, pick_best_methods, assign_method
    """
    pass
    
SamplingMethodRegistry = DictWithDoc()

def blacklist(parameter):
    """
    If you want to exclude a particular sampling method from 
    consideration for handling a parameter, do this:
    
    from PyMC2 import SamplingMethodRegistry
    SamplingMethodRegistry[bad_sampling_method] = blacklist
    """
    return 0
    
def pick_best_methods(parameter):
    """
    Picks the SamplingMethods best suited to handle
    a parameter.
    """
    
    # Keep track of most competent methohd
    max_competence = 0
    # Empty set of appropriate SamplingMethods
    best_candidates = set([])

    # Loop over SamplingMethodRegistry
    for item in SamplingMethodRegistry.iteritems():
        
        # Parse method and its associated competence
        method = item[0]
        competence = item[1](parameter)
        
        # If better than current best method, promote it
        if competence > max_competence:
            best_candidates = set([method])
            max_competence = competence

        # If same competence, add it to the set of best methods
        elif competence == max_competence:
            best_candidates.add(method)
    
    # print parameter.__name__ + ': ', best_candidates, ' ', max_competence
    return best_candidates
    
def assign_method(parameter, scale=None):
    """
    Returns a sampling method instance to handle a 
    parameter. If several methods have the same competence, 
    it picks one arbitrarily (using set.pop()).
    """
    
    # Retrieve set of best candidates
    best_candidates = pick_best_methods(parameter)
    
    # Randomly grab and appropriate method
    method = best_candidates.pop()
    
    if scale:
        return method(parameter = parameter, scale = scale)
        
    return method(parameter = parameter)

class SamplingMethod(object):
    """
    This object knows how to make Parameters take single MCMC steps.
    It's sample() method will be called by Model at every MCMC iteration.
    
    :Parameters:
          -pymc_objects : list, array or set
            Collection of PyMCObjects 
              
          - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    Externally-accessible attributes:
      nodes: The Nodes over which self has jurisdiction.
      parameters: The Parameters over which self has jurisdiction which have isdata = False.
      data: The Parameters over which self has jurisdiction which have isdata = True.
      pymc_objects: The Nodes and Parameters over which self has jurisdiction.
      children: The combined children of all PyMCBases over which self has jurisdiction.
      parents: The combined parents of all PyMCBases over which self has jurisdiction, as a set.
      loglike: The summed log-probability of self's children conditional on all of self's PyMCBases' current values. These will be recomputed only as necessary. This descriptor should eventually be written in C.

    Externally accesible methods:
      sample(): A single MCMC step for all the Parameters over which self has jurisdiction. Must be overridden in subclasses.
      tune(): Tunes proposal distribution widths for all self's Parameters.


    To instantiate a SamplingMethod called S with jurisdiction over a
    sequence/set N of PyMCBases:

      >>> S = SamplingMethod(N)

    :SeeAlso: Metropolis, Sampler.
    """

    def __init__(self, pymc_objects, verbose=0):
        # SamplingMethod initialization
        
        # Initialize public attributes
        self.pymc_objects = set(pymc_objects)
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.children = set()
        self.parents = set()
        
        # Initialize hidden attributes
        self._asf = 1.
        self._accepted = 0.
        self._rejected = 0.
        self._state = ['_rejected', '_accepted', '_asf']
        self.verbose = verbose

        # File away the pymc_objects
        for pymc_object in self.pymc_objects:

            # Sort.
            if isinstance(pymc_object,Node):
                self.nodes.add(pymc_object)
            elif isinstance(pymc_object,Parameter):
                if pymc_object.isdata:
                    self.data.add(pymc_object)
                else:
                    self.parameters.add(pymc_object)

        # Find children, no need to find parents; each pymc_object takes care of those.
        for pymc_object in self.pymc_objects:
            self.children |= pymc_object.children
            for parent in pymc_object.parents.itervalues():
                if isinstance(parent, PyMCBase):
                    self.parents.add(parent) 

        # Find nearest descendent Parameters
        extend_children(self)
        # Find nearest parent Parameters
        extend_parents(self)
        
        # Remove own PyMCObjects from set of children
        self.children -= self.nodes
        self.children -= self.parameters
        self.children -= self.data

        # ID string for verbose feedback
        self._id = 'To define in subclasses'
    
    def step(self):
        """
        Specifies single step of sampling method.
        Must be overridden in subclasses.
        """
        pass

    def tune(self, divergence_threshold=1e10, verbose=0):
        """
        Tunes the scaling hyperparameter for the proposal distribution
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
            print self._id + ' tuning.'
            
        # Calculate recent acceptance rate
        # if not self._accepted > 0 or self._rejected > 0: return ???
        if self._accepted + self._rejected ==0: return
        acc_rate = self._accepted / (self._accepted + self._rejected)

        # Flag for tuning state
        tuning = True

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

        # If the scaling factor is diverging, abort
        # if self._asf > divergence_threshold:
        #     raise DivergenceError, 'Proposal distribution variance diverged'

        # Compute covariance matrix in the multivariate case and the standard
        # variation in all other cases.
        #self.compute_scale(acc_rate,  int_length)

        # More verbose feedback, if requested
        if verbose > 1 or self.verbose > 1:
            print self._id+' acceptance rate:', acc_rate
            print self._id+' adaptive scale factor:', self._asf

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
        
    def _del_loglike(self):
        # Delete log-likelihood
        
        self.loglike = None

    # Make get property for retrieving log-probability
    loglike = property(fget = _get_loglike)

    def current_state(self):
        """Return a dictionary with the current value of the variables defining
        the state of the sampling method."""
        state = {}
        for s in self._state:
            state[s] = getattr(self, s)
        return state
        
# The default SamplingMethod, which Model uses to handle singleton parameters.
class Metropolis(SamplingMethod):
    """
    The default SamplingMethod, which Model uses to handle singleton, continuous parameters.

    Applies the one-at-a-time Metropolis-Hastings algorithm to the Parameter over which self has jurisdiction.

    To instantiate a Metropolis called M with jurisdiction over a Parameter P:

      >>> M = Metropolis(P, scale=1, dist=None)

    :Arguments:
    - parameter : Parameter      
            The parameter over which self has jurisdiction.

    - scale (optional) : number  
            The proposal jump width is set to scale * parameter.value.
    
    - dist (optional) : string  
            The proposal distribution. May be 'Normal', 'RoundedNormal', 'Bernoulli',
            'Prior' or None. If None is provided, a proposal distribution is 
            chosen by examining P.value's type.
            
    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    :SeeAlso: SamplingMethod, Sampler.
    """
    
    def __init__(self, parameter, scale=1., dist=None, verbose=0):
        # Metropolis class initialization
        
        # Initialize superclass
        SamplingMethod.__init__(self, [parameter], verbose)
        
        # Set public attributes
        self.parameter = parameter
        self.verbose = verbose
        
        # Avoid zeros when setting proposal variance
        if all(self.parameter.value != 0.):
            self.proposal_sig = ones(shape(self.parameter.value)) * abs(self.parameter.value) * scale
        else:
            self.proposal_sig = ones(shape(self.parameter.value)) * scale
            
        # Initialize proposal deviate with array of zeros
        self.proposal_deviate = zeros(shape(self.parameter.value), dtype=float)
        
        # Initialize verbose feedback string
        self._id = 'Metropolis_'+parameter.__name__

        # Determine size of parameter
        if isinstance(self.parameter.value, ndarray):
            self._len = len(self.parameter.value.ravel())
        else:
            self._len = 1
        
        # If no dist argument is provided, assign a proposal distribution automatically.
        if not dist:
            
            # Pick Gaussian, just because
            self._dist = "Normal"

            # If self's extended children is the empty set (eg, if
            # self's parameter is a posterior predictive quantity of
            # interest), proposing from the prior is best.
            if self.children:
                try:
                    self.parameter.random()
                    self._dist = "Prior"
                except:
                    pass
        
        else: 
            self._dist = dist
        
    def step(self):
        """
        The default step method applies if the parameter is floating-point 
        valued, and is not being proposed from its prior.
        """

        # Probability and likelihood for parameter's current value:
        
        if self.verbose > 2:
            print
            print self._id + ' getting initial prior.'
            
        if "Prior" in self._dist:
            # No children
            logp = 0.
        else:
            logp = self.parameter.logp

        if self.verbose > 2:
            print self._id + ' getting initial likelihood.'
        loglike = self.loglike

        if self.verbose > 2:
            print self._id + ' proposing.'

        # Sample a candidate value
        if "Prior" in self._dist:
            self.parameter.random()
        else:
            self.propose()

        # Probability and likelihood for parameter's proposed value:
        try:
            if "Prior" in self._dist:
                logp_p = 0.
            else:
                logp_p = self.parameter.logp
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

        # Evaluate acceptance ratio
        if log(random()) > logp_p + loglike_p - logp - loglike:
            
            # Revert parameter if fail
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
        # Sets current parameter value to the last accepted value
        self.parameter.value = self.parameter.last_value
                    
    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self._dist is "Normal" (i.e. no proposal specified).
        """
        if self._dist == "Normal":
            self.parameter.value = rnormal(self.parameter.value, self.proposal_sig)
        
def MetroCompetence(parameter):  
    """
    The competence function for Metropolis
    """
    
    if isinstance(parameter, DiscreteParameter) or isinstance(parameter,BinaryParameter):
        # If the parameter's binary or discrete, I can't do it.
        return 0

    else:
        # If the parameter's value is an ndarray or a number, I can do it,
        # but not necessarily particularly well.
        _type = check_type(parameter)[0]
        if _type in [float, int]:
            return 1
        else:
            return 0
            
SamplingMethodRegistry[Metropolis] = MetroCompetence

            
class DiscreteMetropolis(Metropolis):
    """
    Just like Metropolis, but rounds the parameter's value.
    Good for DiscreteParameters.
    """
    
    def __init__(self, parameter, scale=1., dist=None):
        # DiscreteMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, parameter, scale=scale, dist=dist)
        
        # Initialize verbose feedback string
        self._id = 'DiscreteMetropolis_'+parameter.__name__
    
    def propose(self):      
        # Propose new parameter values using normal distribution
        
        if self._dist == "Normal":
            new_val = rnormal(self.parameter.value,self.proposal_sig)

            self.parameter.value = round_array(new_val)

def DiscreteMetroCompetence(parameter):
    """
    The competence function for DiscreteMetropolis.
    """
    if isinstance(parameter, DiscreteParameter):
        return 1
    else:
        return 0
    
SamplingMethodRegistry[DiscreteMetropolis] = DiscreteMetroCompetence


class BinaryMetropolis(Metropolis):
    """
    Like Metropolis, but with a modified step() method.
    Good for binary parameters.
    
    NOTE this is not compliant with the Metropolis standard
    yet because it lacks a reject() method.
    (??? But, it is a subclass of Metropolis, which has a reject() method)
    """
    
    def __init__(self, parameter, dist=None):
        # BinaryMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, parameter, dist=dist)
        
        # Initialize verbose feedback string
        self._id = 'BinaryMetropolis_'+parameter.__name__
    
    def set_param_val(self, i, val, to_value):
        """
        Utility method for setting a particular element of a parameter's value.
        """
        
        if self._len>1:
            # Vector-valued parameters
            
            val[i] = to_value
            self.parameter.value = reshape(val, self._type[1])
            
        else:
            # Scalar parameters
            
            self.parameter.value = to_value
    
    def step(self):
        """
        This method is substituted for the default step() method in
        BinaryMetropolis.
        """
        
        if "Prior" in self._dist:
            # If no children, just sample new value
            self.parameter.random()
            
        else:        
            
            # Make local variable for value
            if self._len > 1:
                val = self.parameter.value.ravel()
            else:
                val = self.parameter.value

            for i in xrange(self._len):

                self.set_param_val(i, val, True)

                try:    
                    logp_true = self.parameter.logp
                    loglike_true = self.loglike
                except ZeroProbability:
                    self.set_param_val(i, val, False)
                    continue            
                
                self.set_param_val(i, val, False)
            
                try:    
                    logp_false = self.parameter.logp
                    loglike_false = self.loglike            
                except ZeroProbability:
                    self.set_param_val(i,val,True)
                    continue
                
                p_true = exp(logp_true + loglike_true)
                p_false = exp(logp_false + loglike_false)
            
                # Stochastically set value according to relative
                # probabilities of True and False
                if log(random()) > p_true / (p_true + p_false):
                    self.set_param_val(i,val,True)
                    
            # Increment accepted count
            self._accepted += 1
            
def BinaryMetroCompetence(parameter):
    """
    The competence function for Binary One-At-A-Time Metropolis
    """
    if isinstance(parameter, BinaryParameter):
        return 1
    else:
        return 0
        
SamplingMethodRegistry[BinaryMetropolis] = BinaryMetroCompetence
    

class JointMetropolis(SamplingMethod):
    """
    S = Joint(pymc_objects, epoch=1000, memory=10, delay=1000)

    Applies the Metropolis-Hastings algorithm to several parameters
    together. Jumping density is a multivariate normal distribution
    with mean zero and covariance equal to the empirical covariance
    of the parameters, times _asf ** 2.
    
    :Arguments:
    - pymc_objects (optional) : list or array      
            A sequence of pymc objects to handle using
            this SamplingMethod.
            
    - parameter (optional) : Parameter
            Alternatively to pymc_objects, a single parameter can be passed.

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
            Scale parameter.
    
    - oneatatime_scales (optional) : dict
            Dictionary of scales for one-at-a-time iterations.
            
    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    Externally-accessible attributes:

        pymc_objects:   A sequence of pymc objects to handle using
                        this SamplingMethod.

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
        
    TODO: Make this round DiscreteParameters' values and assign Discrete
    individual sampling methods to them, figure out what to do about binary
    parameters.
    
    Also: don't start joint sampling until all parameters have mixed at 
    least a little. Warn if another epoch of one-at-a-time sampling is
    required.
    
    Also: when the covariance is nonsquare,

    """
    def __init__(self, pymc_objects=None, parameter=None, epoch=1000, memory=10, delay = 0, scale=.1, oneatatime_scales=None, verbose=0):
        
        self.verbose = verbose
                
        if parameter is not None:
            pymc_objects = [parameter]
        SamplingMethod.__init__(self, pymc_objects)

        self.epoch = epoch
        self.memory = memory
        self.delay = delay
        self._id = 'JointMetropolis_'+'_'.join([p.__name__ for p in self.parameters])
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
            print self._id + ': Assigning single-parameter handlers.'
        self._single_param_handlers = set()

        SamplingMethodRegistry[JointMetropolis] = blacklist
        for parameter in self.parameters:
            if oneatatime_scales is not None:
                scale_now = oneatatime_scales[parameter]
            else:
                scale_now = None
            
            new_method = assign_method(parameter, scale_now)
            self._single_param_handlers.add(new_method)
        SamplingMethodRegistry[JointMetropolis] = JointMetroCompetence


        # Allocate memory for internal traces and get parameter slices
        self._slices = {}
        self._len = 0
        for parameter in self.parameters:
            if isinstance(parameter.value, ndarray):
                param_len = len(parameter.value.ravel())
            else:
                param_len = 1
            self._slices[parameter] = slice(self._len, self._len + param_len)
            self._len += param_len

        self._proposal_deviate = zeros(self._len,dtype=float)

        self._trace = zeros((self._len, self.memory * self.epoch),dtype=float)

        # __init__ should also check that each parameter's value is an ndarray or
        # a numerical type.

        self._state += ['last_trace_index', '_cov', '_sig',
        '_proposal_deviate', '_trace']


    def compute_sig(self):
        """
        This method computes and stores the matrix square root of the empirical
        covariance every epoch.
        """

        if self.verbose > 1:
            print 'Joint SamplingMethod ' + self._id + ' computing covariance.'

        # Figure out which slice of the traces to use
        if (self._model._cur_trace_index - self.delay) / self.epoch > self.memory:
            trace_slice = slice(self._model._cur_trace_index-self.epoch * self.memory,\
                                self._model._cur_trace_index)
            trace_len = self.memory * self.epoch
        else:
            trace_slice = slice(self.delay, self._model._cur_trace_index)
            trace_len = (self._model._cur_trace_index - self.delay)


        # Store all the parameters' traces in self._trace
        for parameter in self.parameters:
            param_trace = parameter.trace(slicing=trace_slice)

            # If parameter is an array, ravel each tallied value
            if isinstance(parameter.value, ndarray):
                for i in range(trace_len):
                    self._trace[self._slices[parameter], i] = param_trace[i,:].ravel()

            # If parameter is a scalar, there's no need.
            else:
                self._trace[self._slices[parameter], :trace_len] = param_trace

        # Compute matrix square root of covariance of self._trace
        self._cov = cov(self._trace[: , :trace_len])

        self._sig = msqrt(self._cov).T * self.scale

        self._ready = True


    def tune(self, divergence_threshold = 1e10, verbose=0):
        """
        If the empirical covariance hasn't been computed yet (the first
        epoch isn't over), this method passes the tune() call along to the
        Metropolis instances handling self's parameters. If the
        empirical covariance has been computed, the Metropolis
        instances aren't in use anymore so this method does nothing.
        
        We may want to make this method do something eventually.
        """
        if not self._accepted > 0 or self._rejected > 0:
            for handler in self._single_param_handlers:
                handler.tune(divergence_threshold, verbose)


    def propose(self):
        """
        This method proposes values for self's parameters based on the empirical
        covariance.
        """
        fill_stdnormal(self._proposal_deviate)
        
        N = self._sig.shape[1]
        
        proposed_vals = inner(self._proposal_deviate[:N], self._sig)
        
        for parameter in self.parameters:
            
            jump = reshape(proposed_vals[self._slices[parameter]],shape(parameter.value))
            
            parameter.value = parameter.value + jump

    def reject(self):
        """Reject a jump."""
        for parameter in self.parameters:
            parameter.value = parameter.last_value

    def step(self):
        """
        If the empirical covariance hasn't been computed yet, the step() call
        is passed along to the Metropolis instances that handle self's
        parameters before the end of the first epoch.
        
        If the empirical covariance has been computed, values for self's parameters
        are proposed and tested simultaneously.
        """
        if not self._ready:
            for handler in self._single_param_handlers:
                handler.step()
        else:
            # Probability and likelihood for parameter's current value:
            logp = sum([parameter.logp for parameter in self.parameters])
            loglike = self.loglike

            # Sample a candidate value
            self.propose()

            # Probability and likelihood for parameter's proposed value:
            try:
                logp_p = sum([parameter.logp for parameter in self.parameters])
            except ZeroProbability:
                self._rejected += 1
                self.reject()
                return

            loglike_p = self.loglike

            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                # Revert parameter if fail
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
                for handler in self._single_param_handlers:
                    if handler._accepted == 0:
                        print self._id+ ": Warnining, parameter " + handler.parameter.__name__ + " did not mix, continuing one-at-a-time sampling"
                    
                self.compute_sig()
                self.last_trace_index = self._model._cur_trace_index


def JointMetroCompetence(parameter):
    """
    The competence function for Metropolis
    """

    if isinstance(parameter, DiscreteParameter) or isinstance(parameter,BinaryParameter):
        # If the parameter's binary or discrete, I can't do it.
        return 0

    elif isinstance(parameter.value, ndarray):
        return 2

SamplingMethodRegistry[JointMetropolis] = JointMetroCompetence
