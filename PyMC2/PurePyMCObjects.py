#TODO: Run parent_values before checking timestamps, so node parents know
#TODO: that they have to update. Also make these work with containers.

from copy import deepcopy,copy
from numpy import array,zeros,ones
from numpy.linalg import cholesky, eigh
import sys, inspect
from numpy.random import randint, random
from numpy.random import normal as rnormal
import database
from decorators import magic_set
from AbstractBase import *
from utils import extend_children, _push, _extract


def pure_parameter(__func__=None, **kwds):
    """
    Decorator function for instantiating parameters. Usages:
    
    Medium:
    
        @parameter
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
        
        @parameter(trace = trace_object)
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
            
    Long:

        @parameter
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    
        @parameter(trace = trace_object)
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    where foo() computes the log-probability of the parameter A
    conditional on its value and its parents' values, and bar()
    generates a random value from A's distribution conditional on
    its parents' values.
    """

    def instantiate_p(__func__):
        _extract(__func__, kwds, keys)
        return PureParameter(**kwds)        
    keys = ['logp','random']

    if __func__ is None:
        return instantiate_p
    else:
        instantiate_p.kwds = kwds
        return instantiate_p(__func__)

    return instantiate_p


def pure_node(__func__ = None, **kwds):
    """
    Decorator function instantiating nodes. Usages:
    
        @node
        def B(parent_name = ., ...)
            return baz(parent_name, ...)
            
        @node(trace = trace_object)
        def B(parent_name = ., ...)
            return baz(parent_name, ...)            
        
    where baz returns the node B's value conditional
    on its parents.
    """
    def instantiate_n(__func__):
        _extract(__func__, kwds, keys=[])
        return PureNode(eval_fun = __func__, **kwds)        

    if __func__ is None:
        return instantiate_n
    else:
        instantiate_n.kwds = kwds
        return instantiate_n(__func__)

    return instantiate_n


def pure_data(__func__=None, **kwds):
    """
    Decorator instantiating data objects. Usage is just like
    parameter, but once instantiated value cannot be changed.
    """
    return parameter(__func__, isdata=True, trace=False, **kwds)

class PurePyMCObject(PurePyMCBase):
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
                self._parent_timestamp_caches[key] = -1 * ones(self._cache_depth,dtype=int)

                # Record a reference to this parent's value
                self._parent_values[key] = self.parents[key].value

            else:

                # Make a private copy of this parent's value
                self._parent_values[key] = deepcopy(self.parents[key])

        self._recompute = True
        self._value = None
        self._match_indices = zeros(self._cache_depth,dtype=int)
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


class PureNode(PurePyMCObject,NodeBase):
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

    To instantiate: see node()

    See also Parameter and PyMCBase,
    as well as parameter(), and data().
    """

    def __init__(self, eval_fun,  doc, name, trace=True, caching=False, **parents):

        PyMCBase.__init__(self, doc=doc, name=name, **parents)

        self._eval_fun = eval_fun
        self._value = None
        self.trace = trace
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
            if match:
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

class PureParameter(PurePyMCObject, ParameterBase):
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

    To instantiate with isdata = False: see parameter().
    To instantiate with isdata = True: see data().

    See also PyMCBase and Node,
    as well as node().
    """

    def __init__(self, logp, doc, name, random = None, trace=True, caching=False, value=None, rseed=False, isdata=False, **parents):

        PyMCBase.__init__(self, doc=doc, name=name, **parents)

        self.isdata = isdata
        self._logp_fun = logp
        self._logp = None
        self.last_value = None
        self.trace = trace
        self._caching = caching
        self.random = random
        self.rseed = rseed
        if value is None:
            self.rseed = True

        # Caches, if necessary
        if self._caching:
            self._cached_logp = zeros(self._cache_depth,dtype=float)
            self._self_timestamp_caches = -1 * ones(self._cache_depth,dtype=int)

        if rseed is True:
            self._value = self.random()
        else:
            self._value = value

    #
    # Define the attribute value.
    #
    # NOTE: relative timings:
    #
    # A.value = .2:     22.1s  (16.7s in deepcopy, but a shallow copy just doesn't seem like enough...)
    # A._set_value(.2): 21.2s
    # A._value = .2:    .9s
    #
    # A.value:          1.9s
    # A._get_value():   1.8s
    # A._value:         .9s
    #
    # There's a lot to be gained by writing these in C, but not so much
    # by using direct getters and setters.

    def _get_value(self):
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
    # NOTE: relative timings (with one-parent, trivial logp function):
    #
    # caching=False:
    # 
    # A.logp:           9.5s
    # A._get_logp():    9.3s
    # A._logp:          .9s
    #
    # caching=True:
    # 
    # A.logp:           16.3s
    # A._get_logp():    15.4s
    # A._logp:          .9s 
    #
    # Again, there's a lot to be gained by writing these in C.

    # _check_for_recompute should eventually be written in Weave, it's pretty
    # time-consuming.
    def _check_for_recompute(self):

        # Loop over indices
        for index in range(self._cache_depth):
            match = True

            # Look for mismatch of self's timestamp
            if not self._self_timestamp_caches[index] == self.timestamp:
                match = False

            if match:
                # Loop over parents and try to catch mismatches
                for item in self._pymc_object_parents.iteritems():
                    if not self._parent_timestamp_caches[item[0]][index] == item[1].timestamp:
                        match = False
                        break

            # If no mismatches, load value from current cache position
            if match:
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
