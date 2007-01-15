"""
proposition5.py

Classes:

    PyMCBase:                   Abstract base class from which Parameter and PyMCBase inherit.

    Parameter:                  Variable whose value is unknown given the value of its parents
                                (a random variable under the Bayesian interpretation of probability).

    Node:                       Variable whose value is known given the value of its parents.

    SamplingMethod:             Object that knows how to make member variables take one MCMC step.

    OneAtATimeMetropolis:       Default SamplingMethod, instantiated by Model to handle Parameters
                                that aren't handled by any existing SamplingMethod.

    Model:                  Object that manages an MCMC loop.


Functions:

    weight:         Get posterior probabilities for a list of models.

    parameter:      Decorator used to instantiate a Parameter.

    data:           Decorator used to instantiate a Node.

Basically all the underscored functions should eventually be written in C.

"""
from copy import deepcopy
from numpy import *
from numpy.linalg import cholesky, eigh
import sys, inspect
from numpy.random import randint, random
from numpy.random import normal as rnormal
import database
from decorators import magic_set
def _push(seq,new_value):
    """
    Usage:
    _push(seq,new_value)

    Put a deep copy of new_value at the beginning of seq, and kick out the last value.
    """
    length = len(seq)
    for i in range(length-1):
        seq[i+1] = seq[i]
    if isinstance(seq,ndarray):
        # ndarrays will automatically make a copy
        seq[0] = new_value
    else:
        seq[0] = deepcopy(new_value)


class parameter:
    """
    Decorator instantiating the Parameter class. Usage:

    @parameter( init_val, traceable = True, caching = True):
    def P(parent_1_name = some object, parent_2_name = some object, ...):

        def logp_fun(value, parent_1_name, parent_2_name):
            # Function computing log-probability of value given parent values.

        def random(parent_1_name, parent_2_name):
            # Function returning a sample of value given parent values.

    will create a Parameter named P whose log-probability given its parents
    is computed by foo. If random is defined, it will be used to sample
    P's value conditional on its parents when P.draw() is called.

        init_val:       The initial value of the parameter. Required.

        traceable:      Whether Model should make a trace for this Parameter.

        caching:        Whether this Parameter should avoid recomputing its probability
                        by caching previous results.

        parent_i_name:  The label of parent i. See example.

    Example:

    @parameter(init_val = 12.3)
    def P(mu = A, tau = 6.0):

        def logp_fun(value, mu, tau):
            return normal_like(value, mu[10,63], tau)

        # Optional:
        def random(mu, tau):
            return rnormal(mu[10,63], tau)

    creates a parameter called P with two parents, A and 6.0. P.value will be set to
    12.3. When P.prob is computed, it will be set to

    normal_like(P.value, A[10,63], 6.0)             if A is a numpy ndarray

    OR

    normal_like(P.value, A.value[10,63], 6.0)       if A is a PyMC object.

    See also data() and node(),
    as well as Parameter, Node and PyMCBase.
    """
    def __init__(self, **kwds):
        self.kwds = kwds
        self.keys = ['logp_fun', 'random']

    def instantiate(self, *pargs, **kwds):
        """Instantiate the appropriate class."""
        return Parameter(*pargs, **kwds)

    def __call__(self, func):
        self.kwds.update({'doc':func.__doc__, 'name':func.__name__})

        def probeFunc(frame, event, arg):
            if event == 'return':
                locals = frame.f_locals
                self.kwds.update(dict((k,locals.get(k)) for k in self.keys))
                sys.settrace(None)
            return probeFunc

        # Get the functions logp and random (complete interface).
        try:
            sys.settrace(probeFunc)
            func()
        except TypeError:
            print 'Assign func to logp directly (medium interface).'
            self.kwds['logp_fun']=func

        # Build parents dictionary by parsing the function's arguments.
        (args, varargs, varkw, defaults) = inspect.getargspec(func)
        self.kwds.update(dict(zip(args[-len(defaults):], defaults)))

        # Instantiate Class
        C = self.instantiate(**self.kwds)
        return C

class node(parameter):
    """
    Decorator function instantiating the Node class. Usage:

    @Node(  traceable=True, caching=True)
    def N(parent_1_name = some object, parent_2_name = some object, ...):
        # Return the value of the node given the parents' values

    will create a Node named N whose value is computed by bar based on the
    parent objects that were passed into the decorator.

        traceable:      Whether Model should make a trace for this Node.

        caching:        Whether this Node should avoid recomputing its value
                        by caching previous results.

        parent_i_name:  The label of parent i. See example.


    Example:

    @node
    def N(p = .176, q = B):
        return p * q[132]

    creates a Node called N with two parents, .176 and B.
    When N.value is computed, it will be set to

    B[132] * .176           if B is a numpy ndarray

    OR

    B.value[132] * .176     if B is a PyMC object.

    See also parameter() and data(),
    as well as Parameter, Node and PyMCBase.
    """

    def __init__(self, **kwds):
        self.kwds = kwds
        self.keys = ['eval_fun']

    def instantiate(self, *pargs, **kwds):
        """Instantiate the appropriate class."""
        return Node(*pargs, **kwds)

class data(parameter):
    """
    Decorator function instantiating the Parameter class with the 'isdata' flag set to True.
    That means that the attribute value cannot be changed after instantiation. Usage:

    @data(init_val = some value):
    def D(parent_1_name = some object, parent_2_name = some object, ...):

        def logp_fun(value, parent_1_name, parent_2_name, ...):
            # Return the log-probability of value given parent values

        #Optional:
        observed_mask = an ndarray of booleans (not implemented yet)

    will create a Parameter named D whose log-probability is computed by foo, with the property
    that P.value cannot be changed from init_val. Example:

    @data(init_val = 12.3):
    def D(mu = A, tau = 6.0):

        def logp(value, mu, tau):
            return normal_like(value, mu[10,63], tau)

    creates a parameter called D with two parents, A and 6.0. D.value will be set to
    12.3 forever. When D.prob is computed, it will be set to

    normal_like(D.value, A[10,63], 6.0)             if A is a numpy ndarray

    OR

    normal_like(D.value, A.value[10,63], 6.0)       if A is a Node.

    The optional observed_mask is used for partially-observed ndarray-valued parameters.
    It should be set to True at indices where the value has been observed.

    See also parameter() and node(),
    as well as Parameter, Node and PyMCBase.
    """

    def instantiate(self, *pargs, **kwds):
        """Instantiate the appropriate class."""
        kwds['isdata']=True
        kwds['traceable']=False
        return Parameter(*pargs, **kwds)

class PyMCBase(object):
    """
    The base PyMC object. Parameter and Node inherit from this class.

    Externally-accessible attributes:

        parents :       A dictionary containing parents of self with parameter names.
                        Parents can be any type.

        parent_values:  A dictionary containing the values of self's parents.
                        This descriptor should eventually be written in C.

        children :      A set containing children of self.
                        Children must be PyMC objects.

        timestamp :     A counter indicating how many times self's value has been updated.

    Externally-accessible methods:

        init_trace(length): Initializes trace of given length.

        tally():            Writes current value of self to trace.

    PyMCBase should not usually be instantiated directly.

    See also Parameter and Node,
    as well as parameter(), node(), and data().
    """
    def __init__(self, doc, name, cache_depth = 2, **parents):

        self.parents = parents
        self.children = set()
        self.__doc__ = doc
        self.__name__ = name
        self.timestamp = 0

        self._cache_depth = cache_depth

        # Find self's parents that are nodes, to speed up cache checking,
        # and add self to node parents' children sets

        self._parent_timestamp_caches = {}
        self._pymc_object_parents = {}
        self._parent_values = {}

        # Make sure no parents are None.
        for key in self.parents.iterkeys():
            assert self.parents[key] is not None, self.__name__ + ': Error, parent ' + key + ' is None.'

        # Sync up parents and children, figure out which parents are PyMC
        # objects and which are just objects.
        for key in self.parents.iterkeys():

            if isinstance(self.parents[key],PyMCBase):

                # Add self to this parent's children set
                self.parents[key].children.add(self)

                # Remember that this parent is a PyMCBase
                self._pymc_object_parents[key] = self.parents[key]

                # Initialize a timestamp cache for this parent
                self._parent_timestamp_caches[key] = -1 * ones(self._cache_depth,dtype='int')

                # Record a reference to this parent's value
                self._parent_values[key] = self.parents[key].value

            else:

                # Make a private copy of this parent's value
                self._parent_values[key] = deepcopy(self.parents[key])

        self._recompute = True
        self._value = None
        self._match_indices = zeros(self._cache_depth,dtype='int')
        self._cache_index = 1

    #
    # Define the attribute parent_values.
    #
    # Extract the values of parents that are PyMCBases
    def _get_parent_values(self):
        for item in self._pymc_object_parents.iteritems():
            self._parent_values[item[0]] = item[1].value
        return self._parent_values

    parent_values = property(fget=_get_parent_values)

    #
    # Overrideable, if anything needs to be done after Model is initialized.
    #
    def _prepare(self):
        pass

    #
    # Find self's random children. Will be called by Model.__init__().
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

class Node(PyMCBase):
    """
    A PyMCBase that is deterministic conditional on its parents.

    Externally-accessible attributes:

    value : Value conditional on parents. Retrieved from cache when possible,
            recomputed only when necessary. This descriptor should eventually
            be written in C.

    Externally-accessible attributes inherited from PyMCBase:

        parents
        children
        timestamp
        parent_values

    Externally-accessible methods inherited from PyMCBase:

        init_trace(length)
        tally()

    To instantiate: see node()

    See also Parameter and PyMCBase,
    as well as parameter(), and data().
    """

    def __init__(self, eval_fun,  doc, name, traceable=True, caching=False, **parents):

        PyMCBase.__init__(self, doc=doc, name=name, **parents)

        self._eval_fun = eval_fun
        self._value = None
        self._traceable = traceable
        self._caching = caching


        # Caches, if necessary
        if self._caching:
            self._cached_value = []
            for i in range(self._cache_depth): self._cached_value.append(None)

    #
    # Define the attribute value. This should eventually be written in C.
    #
    # See if a recompute is necessary.
    def _check_for_recompute(self):

        # Loop over cache positions
        for index in range(self._cache_depth):
            match = True

            # Loop over parents and try to catch mismatches
            for item in self._pymc_object_parents.iteritems():
                if not self._parent_timestamp_caches[item[0]][index] == item[1].timestamp:
                    match = False
                    break

            # If no mismatches, load value from current cache position
            if match == True:
                self._recompute = False
                self._cache_index = index
                return

        # If there are mismatches at every cache position, recompute
        self._recompute = True

    def _get_value(self):

        if self._caching:
            self._check_for_recompute()

            if self._recompute:

                #Recompute
                self._value = self._eval_fun(**self.parent_values)
                self.timestamp += 1

                # Cache
                _push(self._cached_value, self._value)
                for item in self._pymc_object_parents.iteritems():
                    _push(self._parent_timestamp_caches[item[0]], item[1].timestamp)

            else: self._value = self._cached_value[self._cache_index]

        else:
            self._value = self._eval_fun(**self.parent_values)
            return self._value

        return self._value

    value = property(fget = _get_value)

class Parameter(PyMCBase):
    """
    A PyMCBase that is random conditional on its parents.

    Externally-accessible attributes:

        value :     Current value. When changed, timestamp is incremented. Cannot be
                    changed if isdata = True. This descriptor should eventually be
                    written in C.

        logp :      Current probability of self conditional on parents. Retrieved
                    from cache when possible, recomputed only when necessary.
                    This descriptor should eventually be written in C.

        isdata :    A flag indicating whether self is data.

    Externally-accessible attributes inherited from PyMCBase:

        parents
        children
        timestamp
        parent_values

    Externally-accessible methods:

        revert():   Return value to last value, decrement timestamp.

        random():   If random_fun is defined, this draws a value for self.value from
                    self's distribution conditional on self.parents. Used for
                    model averaging.

    Externally-accessible methods inherited from PyMCBase:

        init_trace(length)
        tally()

    To instantiate with isdata = False: see parameter().
    To instantiate with isdata = True: see data().

    See also PyMCBase and Node,
    as well as node().
    """

    def __init__(self, logp_fun, doc, name, random = None, traceable=True, caching=False, init_val=None, rseed=False, isdata=False, **parents):

        PyMCBase.__init__(self, doc=doc, name=name, **parents)

        self.isdata = isdata
        self._logp_fun = logp_fun
        self._logp = None
        self.last_value = None
        self._traceable = traceable
        self._caching = caching
        self._random = random
        self._rseed = rseed
        if init_val is None:
            self._rseed = True

        # Caches, if necessary
        if self._caching:
            self._cached_logp = zeros(self._cache_depth,dtype='float')
            self._self_timestamp_caches = -1 * ones(self._cache_depth,dtype='int')

        if rseed is True:
            self._value = self.random()
        else:
            self._value = init_val

    #
    # Define the attribute value.
    #
    def _get_value(self, *args, **kwargs):
        return self._value

    # Record new value and increment timestamp
    def _set_value(self, value):
        if self.isdata: print 'Warning, data value updated'
        self.timestamp += 1
        # Save a deep copy of current value
        self.last_value = deepcopy(self._value)
        self._value = value

    value = property(fget=_get_value, fset=_set_value)

    #
    # Define attribute logp.
    #

    # _check_for_recompute should eventually be written in Weave, it's pretty
    # time-consuming.
    def _check_for_recompute(self):

        # Loop over indices
        for index in range(self._cache_depth):
            match = True

            # Look for mismatch of self's timestamp
            if not self._self_timestamp_caches[index] == self.timestamp:
                match = False

            if match == True:
                # Loop over parents and try to catch mismatches
                for item in self._pymc_object_parents.iteritems():
                    if not self._parent_timestamp_caches[item[0]][index] == item[1].timestamp:
                        match = False
                        break

            # If no mismatches, load value from current cache position
            if match == True:
                self._recompute = False
                self._cache_index = index
                return

        # If there are mismatches at any cache position, recompute
        self._recompute = True

    def _get_logp(self):
        if self._caching:
            self._check_for_recompute()
            if self._recompute:

                #Recompute
                self.last_logp = self._logp
                self._logp = self._logp_fun(self._value, **self.parent_values)

                #Cache
                _push(self._self_timestamp_caches, self.timestamp)
                _push(self._cached_logp, self._logp)
                for item in self._pymc_object_parents.iteritems():
                    _push(self._parent_timestamp_caches[item[0]], item[1].timestamp)

            else: self._logp = self._cached_logp[self._cache_index]

        else:
            self.last_logp = self._logp
            self._logp = self._logp_fun(self._value, **self.parent_values)
            return self._logp

        return self._logp

    logp = property(fget = _get_logp)



    #
    # Call this when rejecting a jump.
    #
    def revert(self):
        """
        Call this when rejecting a jump.
        """
        self._logp = self.last_logp
        self._value = self.last_value
        self.timestamp -= 1

    #
    # Sample self's value conditional on parents.
    #
    def random(self):
        """
        Sample self conditional on parents.
        """
        if self._random:
            self.value = self._random(**self.parent_values)
        else:
            raise AttributeError, self.__name__+' does not know how to draw its value, see documentation'

# Was SubModel:
class SamplingMethod(object):
    """
    This object knows how to make Parameters take single MCMC steps.
    It's sample() method will be called by Model at every MCMC iteration.

    Externally-accessible attributes:

        nodes:  The Nodes over which self has jurisdiction.

        parameters: The Parameters over which self has jurisdiction which have isdata = False.

        data:       The Parameters over which self has jurisdiction which have isdata = True.

        pymc_objects:       The Nodes and Parameters over which self has jurisdiction.

        children:   The combined children of all PyMCBases over which self has jurisdiction.

        loglike:    The summed log-probability of self's children conditional on all of
                    self's PyMCBases' current values. These will be recomputed only as necessary.
                    This descriptor should eventually be written in C.

    Externally accesible methods:

        sample():   A single MCMC step for all the Parameters over which self has jurisdiction.
                    Must be overridden in subclasses.

        tune():     Tunes proposal distribution widths for all self's Parameters.

    To instantiate a SamplingMethod called S with jurisdiction over a sequence/set N of PyMCBases:

        S = SamplingMethod(N)

    See also OneAtATimeMetropolis and Model.
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
        #return sum([child.logp for child in self.children])

    loglike = property(fget = _get_loglike)

# The default SamplingMethod, which Model uses to handle singleton parameters.
class OneAtATimeMetropolis(SamplingMethod):
    """
    The default SamplingMethod, which Model uses to handle singleton parameters.

    Applies the one-at-a-time Metropolis-Hastings algorithm to the Parameter over which
    self has jurisdiction.

    To instantiate a OneAtATimeMetropolis called M with jurisdiction over a Parameter P:

        M = OneAtATimeMetropolis(P)

    But you never really need to instantiate OneAtATimeMetropolis, Model does it
    automatically.

    See also SamplingMethod and Model.
    """
    def __init__(self, parameter, scale=.1, dist='Normal'):
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
        logp_p = self.parameter.logp

        # Skip the rest if a bad value is proposed
        if logp_p == -Inf:
            self.parameter.revert()
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
            self.parameter.value += round(rnormal(0,self.proposal_sig))
        # Default to normal random-walk proposal
        else:
            self.parameter.value += rnormal(0,self.proposal_sig)

    #
    # Tune the proposal width.
    #
    def tune(self):
        #
        # Adjust _asf according to some heuristic
        #
        pass

class Joint(SamplingMethod):
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

        # How many values have been recorded this epoch
        self._counter = 0
        # How many epochs have been completed
        self._epoch_counter = 0
        # Flag indicating whether covariance has been computed
        self._ready = False
        # Flag indicating whether to begin joint mode
        self._delaying = True

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
    # Write current value to trace
    #
    def record(self):
        insertion_index = self._epoch_counter * self.epoch + self._counter / self.interval
        for parameter in self.parameters:
            if isinstance(parameter.value, ndarray):
                self._trace[self._slices[parameter], insertion_index] = parameter.value.ravel()
            else:
                self._trace[self._slices[parameter], insertion_index] = parameter.value

    #
    # Compute and store matrix square root of covariance every epoch
    #
    def compute_sig(self):

        # Compute matrix square root
        self._cov = cov(self._trace[: , :(self._epoch_counter+1)*self.epoch])
        try:
            self._sig = cholesky(self._cov)
        except linalg.linalg.LinAlgError:
            val, vec = eigh(self._cov)
            self._sig = vec * sqrt(val)

        self._ready = True



    def tune(self):
        if self._ready == False:
            for handler in self._single_param_handlers:
                handler.tune()
        else:
            #
            # Adjust _asf according to some heuristic
            #
            pass

    def propose(self):
        # Eventually, round the proposed values for discrete parameters.
        self._proposed_vals = self._asf * inner(rnormal(size=self._len) , transpose(self._sig))
        for parameter in self.parameters:
            parameter.value = reshape(self._proposed_vals[self._slices[parameter]],shape(parameter.value))

    #
    # Make a step
    #
    def step(self):
        # Step
        if self._ready == False:
            for handler in self._single_param_handlers:
                handler.step()
        else:
            # Probability and likelihood for parameter's current value:
            logp = sum([parameter.logp for parameter in self.parameters])
            loglike = self.loglike

            # Sample a candidate value
            self.propose()

            # Probability and likelihood for parameter's proposed value:
            logp_p = sum([parameter.logp for parameter in self.parameters])

            # Skip the rest if a bad value is proposed
            if logp_p == -Inf:
                for parameter in self.parameters:
                    parameter.revert()
                return

            loglike_p = self.loglike

            # Test
            if log(random()) > logp_p + loglike_p - logp - loglike:
                # Revert parameter if fail
                for parameter in self.parameters:
                    parameter.revert()

        # When the delay period expires, reset the counter to 0
        if self._counter == self.delay:
            self._delaying = False
            self._counter = 0

        if self._delaying == False:
            # If an interval has passed, record the parameter values.
            if self._counter % self.interval ==0:

                # If an epoch has passed, recompute covariance.
                if self._counter / self.interval % self.epoch == 0 and self._counter > 0:
                    self.compute_sig()

                    # Increment epoch
                    self._epoch_counter += 1

                    # If the trace is full, shift it back an epoch
                    if self._epoch_counter == self.memory:
                        self._trace[: , :(self.memory-1)*self.epoch] = self._trace[: , self.epoch:self.memory*self.epoch]
                        self._epoch_counter -= 1

                    # Start counter over for new epoch
                    self._counter = 0

                self.record()

        self._counter+=1


class Model(object):
    """
    Model manages MCMC loops. It is initialized with no arguments:

    A = Model(class, module or dictionary containing PyMC objects and SamplingMethods)

    Externally-accessible attributes:

        nodes:          All extant Nodes.

        parameters:         All extant Parameters with isdata = False.

        data:               All extant Parameters with isdata = True.

        pymc_objects:               All extant Parameters and Nodes.

        sampling_methods:   All extant SamplingMethods.

    Externally-accessible methods:

        sample(iter,burn,thin): At each MCMC iteration, calls each sampling_method's step() method.
                                Tallies Parameters and Nodes as appropriate.

        trace(parameter, slice): Return the trace of parameter, sliced according to slice.

        remember(trace_index): Return the entire model to the tallied state indexed by trace_index.

        DAG: Draw the model as a directed acyclic graph.

        All the plotting functions can probably go on the base namespace and take Parameters as
        arguments.

    See also SamplingMethod, OneAtATimeMetropolis, PyMCBase, Parameter, Node, and weight.
    """

    def __init__(self, input, dbase=None):

        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.sampling_methods = set()
        self._generations = []
        self._prepared = False
        self.__name__ = None

        if hasattr(input,'__name__'):
            self.__name__ = input.__name__

        #Change input into a dictionary
        if isinstance(input, dict):
            input_dict = input
        else:
            try:
                # If input is a module, reload it to make a fresh copy.
                reload(input)
            except TypeError:
                pass

            input_dict = input.__dict__

        for item in input_dict.iteritems():
            self._fileitem(item)
        
        self._assign_trace_methods(dbase)

    def _fileitem(self, item):

        # If a dictionary is passed in, open it up.
        if isinstance(item[1],dict):
            for subitem in item[1].iteritems():
                self._fileitem(subitem)

        # If another iterable object is passed in, open it up.
        # Broadcast the same name over all the elements.
        """
        This doesn't work so hot, anyone have a better idea?
        I was trying to allow sets/tuples/lists
        of PyMC objects and SamplingMethods to be passed in.

        elif iterable(item[1]) == 1:
            for subitem in item[1]:
                self._fileitem((item[0],subitem))
        """
        # File away the SamplingMethods
        if isinstance(item[1],SamplingMethod):
            # Teach the SamplingMethod its name
            item[1].__name__ = item[0]
            #File it away
            self.__dict__[item[0]] = item[1]
            self.sampling_methods.add(item[1])

        # File away the PyMC objects
        elif isinstance(item[1],PyMCBase):
            self.__dict__[item[0]] = item[1]
            # Add an attribute to the object referencing the model instance.
            setattr(self.__dict__[item[0]], '_model', self)
            
            if isinstance(item[1],Node):
                self.nodes.add(item[1])

            elif isinstance(item[1],Parameter):
                if item[1].isdata:
                    self.data.add(item[1])
                else:  self.parameters.add(item[1])

    #
    # Override __setattr__ so that PyMC objects are read-only once instantiated
    #
    def __setattr__(self, name, value):

        # Don't allow changes to PyMC object attributes
        if self.__dict__.has_key(name):
            if isinstance(self.__dict__[name],PyMCBase):
                raise AttributeError, 'Attempt to write read-only attribute of Model.'

            # Do allow other objects to be changed
            else:
                self.__dict__[name] = value

        # Allow new attributes to be created.
        else:
            self.__dict__[name] = value

    def _assign_trace_methods(self, dbase):
        """Assign trace method to parameters and to the Model class, based on 
        the choice of database.
        Defined database: 
          - None: Traces stored in memory.
          - Txt: Traces stored in memory and saved in txt files at end of 
                sampling. Not implemented.
          - SQLlite: Traces stored in sqllite database. Not implemented. 
          - HDF5: Traces stored in HDF5 database. Partially implemented.
        """
        if dbase is None:
            dbase = 'memory_trace'
        db = getattr(database, dbase)
        reload(db)
        for parameters in self.parameters:
            for name, method in db.parameter_methods().iteritems():
                magic_set(parameters, method)
        
        for name, method in db.model_methods().iteritems():
            magic_set(self, method)
        
    #
    # Prepare for sampling
    #
    def _prepare(self):

        # Initialize database
        self._init_dbase()
        
        # Seed new initial values for the parameters.
        for parameters in self.parameters:
            if parameters._rseed:
                parameters.value = parameters.random(**parameters.parent_values)

        if self._prepared == True:
            return

        self._prepared = True

        # Tell all pymc_objects to get ready for sampling
        self.pymc_objects = self.nodes | self.parameters | self.data
        for pymc_object in self.pymc_objects:
            pymc_object._extend_children()
            pymc_object._prepare()

        # Take care of singleton parameters
        for parameter in self.parameters:

            # Is it a member of any SamplingMethod?
            homeless = True
            for sampling_method in self.sampling_methods:
                if parameter in sampling_method.parameters:
                    homeless = False
                    break

            # If not, make it a new one-at-a-time Metropolis-Hastings SamplingMethod
            if homeless:
                self.sampling_methods.add(OneAtATimeMetropolis(parameter))

    #
    # Return a trace
    #
    def trace(self, object, slice = None):

        """

        Notation: Model is M, Parameter is P

        To return the trace of an P's entire value:

            M.trace(M.A)

        To the trace of a slice of A's value:

            M.trace(M.A, slice)

        This notation is not very nice, it would be better to do
        """

        if slice:
            return self._traces[object]
        else:
            return self._traces[object][slice]

    #
    # Initialize traces
    #
    def _init_traces(self, length):
        """
        init_traces(length)

        Enumerates the pymc_objects that are to be tallied and initializes traces
        for them.

        To be tallyable, a pymc_object has to pass the following criteria:

            -   It is not included in the argument pymc_objects_not_to_tally.

            -   Its value has a shape.

            -   Its value can be made into a numpy array with a numerical
                dtype.
        """
        self._traces = {}
        self._pymc_objects_to_tally = set()
        self._cur_trace_index = 0
        self.max_trace_length = length

        for pymc_object in self.pymc_objects:
            if pymc_object._traceable:
                pymc_object._init_trace(length)
                self._pymc_objects_to_tally.add(pymc_object)

    #
    # Tally
    #
    def tally(self):
        """
        tally()

        Records the value of all tallyable pymc_objects.
        """
        if self._cur_trace_index < self.max_trace_length:
            for pymc_object in self._pymc_objects_to_tally:
                pymc_object.tally(self._cur_trace_index)

        self._cur_trace_index += 1

    #
    # Return to a sampled state
    #
    def remember(self, trace_index = None):
        """
        remember(trace_index = randint(trace length to date))

        Sets the value of all tallyable pymc_objects to a value recorded in
        their traces.
        """
        if trace_index:
            trace_index = randint(self.cur_trace_index)

        for pymc_object in self._pymc_objects_to_tally:
            pymc_object.value = pymc_object.trace()[trace_index]

    #
    # Run the MCMC loop!
    #
    def sample(self,iter,burn,thin):
        """
        sample(iter,burn,thin)

        Prepare pymc_objects, initialize traces, run MCMC loop.
        """

        # Do various preparations for sampling
        self._prepare()

        # Initialize traces
        self._init_traces((iter-burn)/thin)


        for i in range(iter):

            # Tell all the sampling methods to take a step
            for sampling_method in self.sampling_methods:
                sampling_method.step()

            # Tally as appropriate.
            if i > burn and (i - burn) % thin == 0:
                self.tally()

            if i % 1000 == 0:
                print 'Iteration ', i, ' of ', iter

        # Tuning, etc.

        # Finalize
        self._finalize_dbase()
        
    def tune(self):
        """
        Tell all samplingmethods to tune themselves.
        """
        for sampling_method in self.sampling_methods:
            sampling_method.tune()

    def _parse_generations(self):
        """
        Parse up the _generations for model averaging.
        """
        self._prepare()


        # Find root generation
        self._generations.append(set())
        all_children = set()
        for parameter in self.parameters:
            all_children.update(parameter.children & self.parameters)
        self._generations[0] = self.parameters - all_children

        # Find subsequent _generations
        children_remaining = True
        gen_num = 0
        while children_remaining:
            gen_num += 1

            # Find children of last generation
            self._generations.append(set())
            for parameter in self._generations[gen_num-1]:
                self._generations[gen_num].update(parameter.children & self.parameters)

            # Take away parameters that have parents in the current generation.
            thisgen_children = set()
            for parameter in self._generations[gen_num]:
                thisgen_children.update(parameter.children & self.parameters)
            self._generations[gen_num] -= thisgen_children

            # Stop when no subsequent _generations remain
            if len(thisgen_children) == 0:
                children_remaining = False

    def sample_model_likelihood(self,iter):
        """
        Returns iter samples of (log p(data|this_model_params, this_model) | data, this_model)
        """
        loglikes = zeros(iter)

        if len(self._generations) == 0:
            self._parse_generations()
        for i in range(iter):
            if i % 10000 == 0:
                print 'Sample ',i,' of ',iter

            for generation in self._generations:
                for parameter in generation:
                    parameter.random()

            for datum in self.data:
                loglikes[i] += datum.logp

        return loglikes

    def DAG(self,format='raw',path=None):
        """
        DAG(format='raw', path=None)

        Draw the directed acyclic graph for this model and writes it to path.
        If self.__name__ is defined and path is None, the output file is
        ./'name'.'format'. If self.__name__ is undefined and path is None,
        the output file is ./model.'format'.

        Format is a string. Options are:
        'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg',
        'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp',
        'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

        format='raw' outputs a GraphViz dot file.
        """

        if not self._prepared:
            self._prepare()

        if self.__name__ == None:
            self.__name__ = model

        import pydot

        self.dot_object = pydot.Dot()

        pydot_nodes = {}
        pydot_subgraphs = {}

        # Create the pydot nodes from pymc objects
        for datum in self.data:
            pydot_nodes[datum] = pydot.Node(name=datum.__name__,shape='box')
            self.dot_object.add_node(pydot_nodes[datum])

        for parameter in self.parameters:
            pydot_nodes[parameter] = pydot.Node(name=parameter.__name__)
            self.dot_object.add_node(pydot_nodes[parameter])

        for node in self.nodes:
            pydot_nodes[node] = pydot.Node(name=node.__name__,shape='invtriangle')
            self.dot_object.add_node(pydot_nodes[node])

        # Create subgraphs from pymc sampling methods
        for sampling_method in self.sampling_methods:
            if not isinstance(sampling_method,OneAtATimeMetropolis):
                pydot_subgraphs[sampling_method] = subgraph(graph_name = sampling_method.__class__.__name__)
                for pymc_object in sampling_method.pymc_objects:
                    pydot_subgraphs[sampling_method].add_node(pydot_nodes[pymc_object])
                self.dot_object.add_subgraph(pydot_subgraphs[sampling_method])


        # Create edges from parent-child relationships
        counter = 0
        for pymc_object in self.pymc_objects:
            for key in pymc_object.parents.iterkeys():
                if not isinstance(pymc_object.parents[key],PyMCBase):

                    parent_name = pymc_object.parents[key].__class__.__name__ + ' const ' + str(counter)
                    self.dot_object.add_node(pydot.Node(name = parent_name, shape = 'trapezium'))
                    counter += 1
                else:
                    parent_name = pymc_object.parents[key].__name__

                new_edge = pydot.Edge(  src = parent_name,
                                        dst = pymc_object.__name__,
                                        label = key)


                self.dot_object.add_edge(new_edge)

        # Draw the graph
        if not path == None:
            self.dot_object.write(path=path,format=format)
        else:
            ext=format
            if format=='raw':
                ext='dot'
            self.dot_object.write(path='./' + self.__name__ + '.' + ext,format=format)

#
# Get posterior probabilities for a list of models
#
def weight(models, iter, priors = None):
    """
    weight(models, iter, priors = None)

    models is a list of models, iter is the number of samples to use, and
    priors is a dictionary of prior weights keyed by model.

    Example:

    M1 = Model(model_1)
    M2 = Model(model_2)
    weight(models = [M1,M2], iter = 100000, priors = {M1: .8, M2: .2})

    Returns a dictionary keyed by model of the model posterior probabilities.

    Need to attach an MCSE value to the return values!
    """
    loglikes = {}
    i=0
    for model in models:
        print 'Model ', i
        loglikes[model] = model.sample_model_likelihood(iter)
        i+=1

    # Find max log-likelihood for regularization purposes
    max_loglike = 0
    for model in models:
        max_loglike = max((max_loglike,loglikes[model].max()))

    posteriors = {}
    sumpost = 0
    for model in models:
        # Regularize
        loglikes[model] -= max_loglike
        # Exponentiate and average
        posteriors[model] = mean(exp(loglikes[model]))
        # Multiply in priors
        if priors is not None:
            posteriors[model] *= priors[model]
        # Count up normalizing constant
        sumpost += posteriors[model]

    # Normalize
    for model in models:
        posteriors[model] /= sumpost

    return posteriors
