

__docformat__='reStructuredText'
from utils import LikelihoodError, msqrt, extend_children, check_type, round_array
from numpy import ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any
from numpy.linalg.linalg import LinAlgError
from numpy.random import randint, random
from numpy.random import normal as rnormal
from flib import fill_stdnormal
from PyMCObjects import Parameter, Node, PyMCBase

# Changeset history
# 22/03/2007 -DH- Added a _state attribute containing the name of the attributes that make up the state of the sampling method, and a method to return that state in a dict. Added an id.

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

    :SeeAlso: OneAtATimeMetropolis, Sampler.
    """

    def __init__(self, pymc_objects):

        self.pymc_objects = set(pymc_objects)
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.children = set()
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

        extend_children(self)

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
            print
            print 'Tuning', self.name
            print '\tcurrent value:', self.get_value()
            print '\tcurrent proposal hyperparameter:', self._hyp*self._asf

        # Calculate recent acceptance rate
        if not self._accepted > 0 or self._rejected > 0: return
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
            print '\tadaptive scaling factor:', self._asf
            print '\tnew proposal hyperparameter:', self._hyp*self._asf

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
class OneAtATimeMetropolis(SamplingMethod):
    """
    The default SamplingMethod, which Model uses to handle singleton parameters.

    Applies the one-at-a-time Metropolis-Hastings algorithm to the Parameter over which
    self has jurisdiction.



    To instantiate a OneAtATimeMetropolis called M with jurisdiction over a Parameter P:

      >>> M = OneAtATimeMetropolis(P, scale=1, dist='Normal')



    :Arguments:
    P:      The parameter over which self has jurisdiction.

    scale:  The proposal jump width is set to scale * parameter.value.

    :SeeAlso: SamplingMethod, Sampler.
    """
    def __init__(self, parameter, scale=1.):
        SamplingMethod.__init__(self,[parameter])
        self.parameter = parameter
        self.proposal_sig = ones(shape(self.parameter.value)) * abs(self.parameter.value) * scale
        self.proposal_deviate = zeros(shape(self.parameter.value),dtype=float)
        self._id = 'OneAtATimeMetropolis_'+parameter.__name__
        
        self._type = check_type(parameter)
        if self._type[0] is bool:
            self._dist = "Bernoulli"
            self.step = self.bernoulli_proposal_step
        elif self._type[0] is int:
            self._dist = "RoundedNormal"
        elif self._type[0] is float:
            self._dist = "Normal"
        else:
            raise TypeError,    'Parameter ' + parameter.__name__ + "'s value must be numeric or boolean"+\
                                'or ndarray with numeric or boolean dtype for OneAtATimeMetropolis to be applied.'

        # If self's extended children is the empty set (eg, if
        # self's parameter is a posterior predictive quantity of
        # interest), proposing from the prior is best.
        if len(self.children) == 0:
            try:
                self.parameter.random()
                self._dist = "Prior"
                self.step = self.prior_proposal_step
            except:
                pass

    #
    # Do a one-at-a-time Metropolis-Hastings step self's Parameter.
    #
    def step(self):

        # Probability and likelihood for parameter's current value:

        logp = self.parameter.logp

        loglike = self.loglike

        # Sample a candidate value
        self.propose()

        # Probability and likelihood for parameter's proposed value:
        try:
            logp_p = self.parameter.logp
        except LikelihoodError:
            self.parameter.value = self.parameter.last_value
            self._rejected += 1
            return

        loglike_p = self.loglike

        # Test
        if log(random()) > logp_p + loglike_p - logp - loglike:
            # Revert parameter if fail
            self.parameter.value = self.parameter.last_value

            self._rejected += 1
        else:
            self._accepted += 1

    def prior_proposal_step():
        # Probability and likelihood for parameter's current value:

        loglike = self.loglike

        # Sample a candidate value
        self.parameter.random()

        loglike_p = self.loglike

        # Test
        if log(random()) > loglike_p - loglike:
            # Revert parameter if fail
            self.parameter.value = self.parameter.last_value

            self._rejected += 1
        else:
            self._accepted += 1
            
    def bernoulli_proposal_step():

        self.parameter.value = True
        logp_true = self.parameter.logp
        
        self.parameter.value = False
        logp_false = self.parameter.logp
        
        p_true = exp(logp_true)
        p_false = exp(logp_false)
        
        if log(random()) > p_true / (p_true + p_false):
            self.parameter.value = True
        else:
            self.parameter.value = False

    def propose(self):

        # Use for continuous parameters
        if self._dist == 'Normal':
            self.parameter.value = rnormal(self.parameter.value,self.proposal_sig)

        # Use for discrete parameters
        elif self._dist == 'RoundedNormal':
            self.parameter.value = round_array(rnormal(self.parameter.value,self.proposal_sig))

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

    """
    def __init__(self, pymc_objects, epoch=1000, memory=10, delay = 0, oneatatime_scales=None):

        SamplingMethod.__init__(self,pymc_objects)

        self.epoch = epoch
        self.memory = memory
        self.delay = delay
        self._id = 'JointMetropolis_'+'_'.join([p.__name__ for p in self.parameters])
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

        # Flag indicating whether covariance has been computed
        self._ready = False

        # For making sure the covariance isn't recomputed multiple times
        # on the same trace index
        self.last_trace_index = 0

        # Use OneAtATimeMetropolis instances to handle independent jumps
        # before first epoch is complete
        self._single_param_handlers = set()
        for parameter in self.parameters:
            if oneatatime_scales is not None:
                self._single_param_handlers.add(OneAtATimeMetropolis(parameter,
                                                scale=oneatatime_scales[parameter]))
            else:
                self._single_param_handlers.add(OneAtATimeMetropolis(parameter))

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

        self._state.append(['last_trace_index', '_cov', '_sig',
        '_proposal_deviate', '_trace'])

    #
    # Compute and store matrix square root of covariance every epoch
    #
    def compute_sig(self):

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
        if not self._accepted > 0 or self._rejected > 0:
            for handler in self._single_param_handlers:
                handler.tune(divergence_threshold, verbose)


    def propose(self):
        # Eventually, round the proposed values for discrete parameters.
        fill_stdnormal(self._proposal_deviate)
        
        proposed_vals = self._asf * inner(self._proposal_deviate, self._sig)
        
        for parameter in self.parameters:
            
            jump = reshape(proposed_vals[self._slices[parameter]],shape(parameter.value))
            
            if self.isdiscrete[parameter]:
                parameter.value = parameter.value + round_array(jump)
            else:
                parameter.value = parameter.value + jump

    #
    # Make a step
    #
    def step(self):
        # Step
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
            except LikelihoodError:
                for parameter in self.parameters:
                    parameter.value = parameter.last_value
                    self._rejected += 1
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

            self.compute_sig()
            self.last_trace_index = self._model._cur_trace_index
