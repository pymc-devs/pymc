"""
Class LazyFunction is defined here.
"""

__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from PyMCObjects import Node, ContainerBase, Variable
import numpy as np

cdef extern from "stdlib.h":
    void* malloc(int size)
    void* free(void* ptr)

cdef extern from "numpy/ndarrayobject.h":
    void* PyArray_DATA(object obj)

cdef class NumberHolder:
    cdef int number
    def __init__(self):
        self.number = 1000

cdef class PointerHolder:
    cdef int* my_pointer
    def __init__(self, number_holder):
        cdef NumberHolder real_holder
        real_holder = number_holder
        self.my_pointer = &real_holder.number
    def get_number(self):
        return self.my_pointer[0]

cdef class Counter:
    cdef long count
    cdef long totcount
    cdef long lastcount
    def __init__(self):
        self.count = 0
        self.totcount = 0
    def get_count(self):
        return self.count
    def click(self):
        self.lastcount = self.count
        self.count = self.totcount+1
        self.totcount += 1
    def unclick(self):
        self.count = self.lastcount

cdef class LazyFunction:
    """
    A function that is bound to its arguments, and which caches
    its last few evaluations so as to skip computation when possible.



    L = LazyFunction(fun, arguments[, cache_depth])
    L.get()



    :Arguments:
    fun:        The function that the LazyFunction uses to compute its
                values.

    arguments:  A dictionary of arguments that the LazyFunction passes
                to its function. If any of the arguments is a Stochastic,
                Deterministic, or Container, that argument's 'value' attribute
                will be substituted for it when passed to fun.

    cache_depth:    The number of prior computations to 'memoize' in
                    order to skip unnecessary computations.



    Externally-accessible methods:

    refresh_argument_values():  Iterates over LazyFunction's parents that are
                                Stochastics, Deterministics, or Containers and
                                records their current values for passing to self's
                                internal function.

    get():  Refreshes argument values, checks cache, calls internal function if
            necessary, returns value.



    Externally-accessible attributes:

    arguments:  A dictionary containing self's arguments.

    fun:        Self's internal function.

    argument_values:    A dictionary containing self's arguments, in which
                        Stochastics, Deterministics, and Containers have been
                        replaced by their 'value' attributes.



    :SeeAlso: Stochastic, Deterministic, Container
    """

    cdef public object arguments, fun, argument_values
    cdef int cache_depth, n_ultimate_args
    cdef public object ultimate_args
    cdef public object cached_values
    # cdef object ultimate_arg_counter
    cdef int *frame_queue
    cdef long* cached_counts,
    cdef long** ultimate_arg_counters
    # cdef public object frame_queue

    def __init__(self, fun, arguments, ultimate_args, cache_depth):

        cdef object arg, name
        cdef int i
        cdef Counter this_counter

        # arguments will be parents and value for stochastics, just parents for deterministics.
        self.arguments = arguments

        self.cache_depth = cache_depth

        # Populate argument iterables
        for name in arguments.iterkeys():
            # This object is arg.
            arg = arguments[name]

        self.ultimate_args = list(ultimate_args)
        self.n_ultimate_args = len(self.ultimate_args)

        # Initialize caches
        self.cached_values = []
        for i in range(self.cache_depth):
            self.cached_values.append(None)

        # Underlying function
        self.fun = fun

        # C initializations
        self.frame_queue = <int*> malloc(sizeof(int)*self.cache_depth)
        for i from 0<=i<self.cache_depth:
            self.frame_queue[i]=i

        self.ultimate_arg_counters = <long**> malloc(sizeof(long*)*self.n_ultimate_args)
        for i from 0 <= i < self.n_ultimate_args:
            this_counter = self.ultimate_args[i].counter
            self.ultimate_arg_counters[i] = &this_counter.count
        # Cache is an array of counter values.
        self.cached_counts = <long*> malloc(sizeof(long)*self.n_ultimate_args*self.cache_depth)
        for i from 0 <= i < self.n_ultimate_args*self.cache_depth:
            self.cached_counts[i] = -1

    def __dealloc__(self):
        free(self.frame_queue)
        free(self.ultimate_arg_counters)
        free(self.cached_counts)


    cdef int check_argument_caches(self)  except *:
        """
        Compare the value of the ultimate arguments to the values in the cache.
        If there's a mismatch (match is by reference, not value), return -1.
        Otherwise return 0.
        """
        # TODO: When GIL-less Python becomes available, do this in the
        # worker threads.
        cdef int i, j, k, mismatch, this_testval
        # TODO: Py_BEGIN_ALLOW_THREADS

        for i from 0 <= i < self.cache_depth:
            mismatch = 0

            for j from 0 <= j < self.n_ultimate_args:
                this_testval = self.ultimate_arg_counters[j][0]
                if not this_testval == self.cached_counts[i * self.n_ultimate_args + j]:
                    mismatch = 1
                    break

            if mismatch == 0:
                # Find i in the frame queue and move it to the end, so the current
                # value gets overwritten late.
                for j from 0 <= j < self.cache_depth:
                    if self.frame_queue[j] == i:
                        break
                for k from j <= k < self.cache_depth-1:
                    self.frame_queue[k] = self.frame_queue[k+1]
                self.frame_queue[self.cache_depth-1] = i
                return i

        i=-1
        # TODO: Py_END_ALLOW_THREADS

        return i

    cdef void cache(self, value) except *:
        """
        Stick self's value in the cache, and also the values of all self's
        ultimate arguments for future caching.
        """
        cdef int i, j, cur_frame

        cur_frame = self.frame_queue[0]
        for j from 0 <= j < self.cache_depth-1:
            self.frame_queue[j] = self.frame_queue[j+1]
        self.frame_queue[self.cache_depth-1] = cur_frame

        # Store new
        self.cached_values[cur_frame] = value
        for j from 0 <= j < self.n_ultimate_args:
            self.cached_counts[cur_frame * self.n_ultimate_args + j] = self.ultimate_arg_counters[j][0]

    def get_frame_queue(self):
        cdef int i
        cdef object out
        out = np.empty(self.cache_depth,dtype=int)
        for i from 0<=i<self.cache_depth:
            out[i] = self.frame_queue[i]
        return out

    def get_cached_counts(self):
        cdef int i, cf
        cdef object out
        out = np.empty((self.n_ultimate_args, self.cache_depth), dtype=long)
        for cf from 0 <= cf < self.cache_depth:
            for i from 0 <= i < self.n_ultimate_args:
                out[i,cf] = self.cached_counts[cf * self.n_ultimate_args + i]
        return out

    def get_ultimate_arg_counter(self):
        cdef object out
        cdef int i
        out = {}
        for i from 0 <= i < self.n_ultimate_args:
            out[self.ultimate_args[i]] = self.ultimate_arg_counters[i][0]
        return out



    def force_cache(self, value):
        """
        Forces an arbitrary value into the cache, associated with
        self's current input arguments. Use with caution!
        """
        cdef int match_index
        match_index = self.check_argument_caches()
        if match_index<0:
            self.cache(value)
        else:
            self.cached_values[match_index]=value

    def force_compute(self):
        """
        For debugging purposes. Skip cache checking and compute a value.
        """
        # Refresh parents' values
        value = self.fun(**self.arguments.value)

        if self.cache_depth>0:
            self.cache(value)


    def get(self):
        """
        Call this method to cause the LazyFunction to refresh its arguments'
        values, decide whether it needs to recompute, and return its current
        value.
        """
        cdef int match_index
        # print 'get called'

        if self.cache_depth == 0:
            return self.fun(**self.arguments.value)

        match_index = self.check_argument_caches()

        if match_index < 0:

            #Recompute
            value = self.fun(**self.arguments.value)
            self.cache(value)

        else:
            value = self.cached_values[match_index]

        return value
