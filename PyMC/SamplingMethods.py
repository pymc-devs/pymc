__docformat__='reStructuredText'
from PyMCObjects import PyMCBase, Parameter, Node
from utils import LikelihoodError
from numpy import *
from numpy.linalg import cholesky, eigh
from numpy.random import randint, random
from numpy.random import normal as rnormal



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

    :SeeAlso: OneAtATimeMetropolis, Model.
    """

    def __init__(self, pymc_objects):

        self.pymc_objects = set(pymc_objects)
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.children = set()
        self._asf = .1
        self._accepted = 0
        self._rejected = 0

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

        self._extend_children()

        self.children -= self.nodes
        self.children -= self.parameters
        self.children -= self.data

    #
    # Must be overridden in subclasses
    #
    def step(self):
        pass

    #
    # Must be overridden in subclasses
    #
    def tune(self):
        pass

    #
    # Find nearest random descendants
    #
    def _extend_children(self):
        need_recursion = False
        node_children = set()
        for child in self.children:
            if isinstance(child,Node):
                self.children |= child.children
                node_children.add(child)
                need_recursion = True
        self.children -= node_children
        if need_recursion:
            self._extend_children()
        return

    #
    # Define attribute loglike.
    #
    def _get_loglike(self):
        sum = 0.
        for child in self.children: sum += child.logp
        return sum

    loglike = property(fget = _get_loglike)

# The default SamplingMethod, which Model uses to handle singleton parameters.
class OneAtATimeMetropolis(SamplingMethod):
    """
    The default SamplingMethod, which Model uses to handle singleton parameters.

    Applies the one-at-a-time Metropolis-Hastings algorithm to the Parameter over which
    self has jurisdiction.

    To instantiate a OneAtATimeMetropolis called M with jurisdiction over a Parameter P:

      >>> M = OneAtATimeMetropolis(P)

    But you never really need to instantiate OneAtATimeMetropolis, Model does it
    automatically.

    :SeeAlso: SamplingMethod, Model.
    """
    def __init__(self, parameter, scale=1, dist='Normal'):
        SamplingMethod.__init__(self,[parameter])
        self.parameter = parameter
        self.proposal_sig = ones(shape(self.parameter.value)) * abs(self.parameter.value) * scale
        self._dist = dist

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
            self.parameter.revert()
            self._rejected += 1
            return

        loglike_p = self.loglike

        # Test
        if log(random()) > logp_p + loglike_p - logp - loglike:
            # Revert parameter if fail
            self.parameter.revert()
            
            self._rejected+=1
        else:
            self._accepted += 1


    def propose(self):
        if self._dist == 'RoundedNormal':
            self.parameter.value = round(rnormal(self.parameter.value, self.proposal_sig))
        # Default to normal random-walk proposal
        else:
            self.parameter.value = rnormal(self.parameter.value, self.proposal_sig)

    #
    # Tune the proposal width.
    #
    def tune(self):
        #
        # Adjust _asf according to some heuristic
        #
        pass

class JointMetropolis(SamplingMethod):
    """
    S = Joint(pymc_objects, epoch=1000, memory=10, interval=1, delay=1000)

    Applies the Metropolis-Hastings algorithm to several parameters
    together. Jumping density is a multivariate normal distribution
    with mean zero and covariance equal to the empirical covariance
    of the parameters, times _asf ** 2.

    Externally-accessible attributes:

        pymc_objects:   A sequence of pymc objects to handle using
                        this SamplingMethod.

        interval:       The interval at which S's parameters' values
                        should be written to S's internal traces
                        (NOTE: If the traces are moved back into the
                        PyMC objects, it should be possible to avoid this
                        double-tallying. As it stands, though, the traces
                        are stored in Model, and SamplingMethods have no
                        way to know which Model they're going to be a
                        member of.)

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
    def __init__(self, pymc_objects, epoch=1000, memory=10, interval = 1, delay = 0):

        SamplingMethod.__init__(self,pymc_objects)

        self.epoch = epoch
        self.memory = memory
        self.interval = interval
        self.delay = delay

        # Flag indicating whether covariance has been computed
        self._ready = False

        # Use OneAtATimeMetropolis instances to handle independent jumps
        # before first epoch is complete
        self._single_param_handlers = set()
        for parameter in self.parameters:
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
            
        self._trace = zeros((self._len, self.memory * self.epoch),dtype='float')               

        # __init__ should also check that each parameter's value is an ndarray or
        # a numerical type.

    #
    # Compute and store matrix square root of covariance every epoch
    #
    def compute_sig(self):
        
        print 'Joint SamplingMethod ' + self.__name__ + ' computing covariance.'
        
        # Figure out which slice of the traces to use
        if (self._model._cur_trace_index - self.delay) / self.epoch / self.interval > self.memory:
            trace_slice = slice(self._model._cur_trace_index-self.epoch * self.memory,\
                                self._model._cur_trace_index, \
                                self.interval)
            trace_len = self.memory * self.epoch
        else:
            trace_slice = slice(self.delay, self._model._cur_trace_index, \
                                self.interval)
            trace_len = (self._model._cur_trace_index - self.delay) / self.interval
            
        
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
        
        # Try Cholesky factorization
        try:
            self._sig = cholesky(self._cov)
        
        # If there's a small eigenvalue, diagonalize
        except linalg.linalg.LinAlgError:
            val, vec = eigh(self._cov)
            self._sig = vec * sqrt(val)

        self._ready = True



    def tune(self):
        if self._ready:
            for handler in self._single_param_handlers:
                handler.tune()
        else:
            #
            # Adjust _asf according to some heuristic
            #
            pass

    def propose(self):
        # Eventually, round the proposed values for discrete parameters.
        proposed_vals = self._asf * inner(rnormal(size=self._len) , self._sig)
        for parameter in self.parameters:
            parameter.value = parameter.value + reshape(proposed_vals[self._slices[parameter]],shape(parameter.value))

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
                    parameter.revert()
                    self._rejected += 1
                return

            loglike_p = self.loglike

            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                # Revert parameter if fail
                self._rejected += 1
                for parameter in self.parameters:
                    parameter.revert()
            else:
                self._accepted += 1

        # If an epoch has passed, recompute covariance.
        if  (float(self._model._cur_trace_index - self.delay) / float(self.interval)) % self.epoch == 0 \
            and self._model._cur_trace_index > self.delay:
            self.compute_sig()
