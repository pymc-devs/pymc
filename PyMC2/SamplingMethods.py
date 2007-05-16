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
    max_competence = 0
    best_candidates = set([])

    for item in SamplingMethodRegistry.iteritems():
        method = item[0]
        competence = item[1](parameter)
        
        if competence > max_competence:
            best_candidates = set([method])
            max_competence = competence

        elif competence == max_competence:
            best_candidates.add(method)
    
    # print parameter.__name__ + ': ', best_candidates, ' ', max_competence
    return best_candidates
    
def assign_method(parameter):
    """
    Returns a sampling method instance to handle a 
    parameter. If several methods have the same competence, 
    it picks one arbitrarily (using set.pop()).
    """
    best_candidates = pick_best_methods(parameter)
    return best_candidates.pop()(parameter=parameter)

class SamplingMethod(object):
    """
    This object knows how to make Parameters take single MCMC steps.
    It's sample() method will be called by Model at every MCMC iteration.

    Externally-accessible attributes:
      - nodes:  The Nodes over which self has jurisdiction.
      - parameters: The Parameters over which self has jurisdiction which have isdata = False.
      - data:       The Parameters over which self has jurisdiction which have isdata = True.
      - pymc_objects:       The Nodes and Parameters over which self has jurisdiction.
      - children:   The combined children of all PyMCBases over which self has jurisdiction.
      - parents:    The combined parents of all PyMCBases over which self has jurisdiction, as a set.
      - loglike:    The summed log-probability of self's children conditional on all of
                    self's PyMCBases' current values. These will be recomputed only as necessary.
                    This descriptor should eventually be written in C.

    Externally accesible methods:
      - sample():   A single MCMC step for all the Parameters over which self has jurisdiction.
        Must be overridden in subclasses.
      - tune():     Tunes proposal distribution widths for all self's Parameters.


    To instantiate a SamplingMethod called S with jurisdiction over a
    sequence/set N of PyMCBases:

      >>> S = SamplingMethod(N)

    :SeeAlso: Metropolis, Sampler.
    """

    def __init__(self, pymc_objects):

        self.pymc_objects = set(pymc_objects)
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.children = set()
        self.parents = set()
        self._asf = .1
        self._accepted = 0.
        self._rejected = 0.
        self._state = ['_rejected', '_accepted', '_asf']

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

        extend_children(self)
        extend_parents(self)
        self.children -= self.nodes
        self.children -= self.parameters
        self.children -= self.data

        self._id = 'To define in subclasses'
    #
    # Must be overridden in subclasses
    #
    def step(self):
        pass

    #
    # May be overridden in subclasses
    #
    def tune(self, divergence_threshold=1e10, verbose=False):
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
        """

        if verbose:
            print 'Tuning'
            
        # Calculate recent acceptance rate
        #if not self._accepted > 0 or self._rejected > 0: return ???
        if self._accepted + self._rejected ==0:return
        acc_rate = self._accepted / (self._accepted + self._rejected)

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
        
        self.proposal_sig *= self._asf
        self._asf = 1.
        
        # Re-initialize rejection count
        self._rejected = 0.
        self._accepted = 0.

        # If the scaling factor is diverging, abort
        if self._asf > divergence_threshold:
            raise DivergenceError, 'Proposal distribution variance diverged'

        # Compute covariance matrix in the multivariate case and the standard
        # variation in all other cases.
        #self.compute_scale(acc_rate,  int_length)

        if verbose:
            print '\tacceptance rate:', acc_rate
            print '\tnew proposal hyperparameter:', self.proposal_sig

    #
    # Define attribute loglike.
    #
    def _get_loglike(self):
        sum = 0.
        for child in self.children: sum += child.logp
        return sum

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
    The default SamplingMethod, which Model uses to handle singleton continuous parameters.

    Applies the one-at-a-time Metropolis-Hastings algorithm to the Parameter over which
    self has jurisdiction.



    To instantiate a Metropolis called M with jurisdiction over a Parameter P:

      >>> M = Metropolis(P, scale=1, dist=None)



    :Arguments:
    P:      The parameter over which self has jurisdiction.

    scale:  The proposal jump width is set to scale * parameter.value.
    
    dist:   The proposal distribution. May be 'Normal', 'RoundedNormal', 'Bernoulli',
            'Prior' or None. If None is provided, a proposal distribution is chosen by
            examining P.value's type.

    :SeeAlso: SamplingMethod, Sampler.
    """
    def __init__(self, parameter, scale=1., dist=None, verbose=False):
        SamplingMethod.__init__(self,[parameter])
        self.parameter = parameter
        self.verbose = verbose
        if all(self.parameter.value != 0.):
            self.proposal_sig = ones(shape(self.parameter.value)) * abs(self.parameter.value) * scale
        else:
            self.proposal_sig = ones(shape(self.parameter.value)) * scale
            
        self.proposal_deviate = zeros(shape(self.parameter.value),dtype=float)
        self._id = 'Metropolis_'+parameter.__name__

        if isinstance(self.parameter.value, ndarray):
            self._len = len(self.parameter.value.ravel())
        else:
            self._len = 1
        
        # If no dist argument is provided, assign a proposal distribution automatically.
        if dist is None:
            
            self._dist = "Normal"

            # If self's extended children is the empty set (eg, if
            # self's parameter is a posterior predictive quantity of
            # interest), proposing from the prior is best.
            if len(self.children) == 0:
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
        
        if self._dist == "Prior":
            logp = 0.
        else:
            logp = self.parameter.logp
        loglike = self.loglike

        # Sample a candidate value
        if self._dist == "Prior":
            self.parameter.random()
        else:
            self.propose()

        # Probability and likelihood for parameter's proposed value:
        try:
            if self._dist == "Prior":
                logp_p = 0.
            else:
                logp_p = self.parameter.logp
            loglike_p = self.loglike

        except ZeroProbability:
            # print self.parameter.__name__ + ' rejecting value ', self.parameter.value, ' back to ', self.parameter.last_value
            self.parameter.value = self.parameter.last_value
            self._rejected += 1
            return
            
        if self.verbose:
            print 'logp_p - logp: ', logp_p - logp
            print 'loglike_p - loglike: ', loglike_p - loglike

        # Test
        if log(random()) > logp_p + loglike_p - logp - loglike:
            # Revert parameter if fail
            self.parameter.value = self.parameter.last_value
            self._rejected += 1
            if self.verbose:
                print 'Rejecting'
        else:
            self._accepted += 1
            if self.verbose:
                print 'Accepting'
            
            
    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self._dist is "Normal"
        """
        if self._dist == "Normal":
            self.parameter.value = rnormal(self.parameter.value,self.proposal_sig)
        
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
        Metropolis.__init__(self, parameter, scale=scale, dist=dist)
        self._id = 'DiscreteMetropolis_'+parameter.__name__
    
    def propose(self):      
        if self._dist == "Normal":
            new_val = rnormal(self.parameter.value,self.proposal_sig)
            # print new_val, ' ', round_array(new_val)

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
    """
    def __init__(self, parameter, dist=None):
        Metropolis.__init__(self, parameter, dist=dist)
        self._id = 'BinaryMetropolis_'+parameter.__name__
    
    def set_param_val(self, i, val, to_value):
        """
        Utility method for setting a particular element of a parameter's value.
        """
        if self._len>1:
            val[i] = value
            self.parameter.value = reshape(val, self._type[1])
        else:
            self.parameter.value = value
    
    def step(self):
        """
        This method is substituted for the default step() method in
        BinaryMetropolis.
        """
        if self._dist=="Prior":
            self.parameter.random()
            
        else:        
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
            
                if log(random()) > p_true / (p_true + p_false):
                    self.set_param_val(i,val,True)
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
    def __init__(self, pymc_objects=None, parameter=None, epoch=1000, memory=10, delay = 0, oneatatime_scales=None):
        
        if parameter is not None:
            pymc_objects = [parameter]
        SamplingMethod.__init__(self,pymc_objects)

        self.epoch = epoch
        self.memory = memory
        self.delay = delay
        self._id = 'JointMetropolis_'+'_'.join([p.__name__ for p in self.parameters])
        self.isdiscrete = {}
        
##        for parameter in self.parameters:
##            type_now = check_type(parameter)[0]
##            if not type_now is float and not type_now is int:
##                raise TypeError,    'Parameter ' + parameter.__name__ + "'s value must be numeric"+\
##                                    'or ndarray with numeric dtype for JointMetropolis to be applied.'
##            elif type_now is int:
##                self.isdiscrete[parameter] = True
##            else:
##                self.isdiscrete[parameter] = False

        # Flag indicating whether covariance has been computed
        self._ready = False

        # For making sure the covariance isn't recomputed multiple times
        # on the same trace index
        self.last_trace_index = 0

        # Use Metropolis instances to handle independent jumps
        # before first epoch is complete
        self._single_param_handlers = set()
        for parameter in self.parameters:
            if oneatatime_scales is not None:
                self._single_param_handlers.add(Metropolis(parameter,
                                                scale=oneatatime_scales[parameter]))
            else:
                self._single_param_handlers.add(Metropolis(parameter))

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

        try:
            print 'Joint SamplingMethod ' + self.__name__ + ' computing covariance.'
        except AttributeError:
            print 'Joint SamplingMethod ' + ' computing covariance.'

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

        self._sig = msqrt(self._cov).T

        self._ready = True


    def tune(self, divergence_threshold = 1e10, verbose=False):
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
        
        proposed_vals = self._asf * inner(self._proposal_deviate[:N], self._sig)
        
        for parameter in self.parameters:
            
            jump = reshape(proposed_vals[self._slices[parameter]],shape(parameter.value))
            
            parameter.value = parameter.value + jump

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
                for parameter in self.parameters:
                    parameter.value = parameter.last_value
                return

            loglike_p = self.loglike

            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                # Revert parameter if fail
                self._rejected += 1
                for parameter in self.parameters:
                    parameter.value = parameter.last_value
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
