__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from numpy import array, zeros, ones, arange, resize
from PyMC import Node, ContainerBase, Variable
from Container import ArrayContainer
from Container_values import ACValue

cdef extern from "stdlib.h":
    void* malloc(int size)
    
cdef extern from "numpy/ndarrayobject.h":
    void* PyArray_DATA(object obj)

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
    cdef int cache_depth, N_args, 
    cdef public object ultimate_args
    cdef void **ultimate_arg_p, **ultimate_arg_value_p
    cdef public object cached_args, cached_values
    cdef void **cached_arg_p
    
    cdef public object frame_queue
    
    def __init__(self, fun, arguments, cache_depth):
        
        cdef object arg, name
        cdef int i
        
        # arguments will be parents and value for stochs, just parents for dtrms.
        self.arguments = arguments
        
        self.cache_depth = cache_depth
    
        
        self.frame_queue = []
        for i in xrange(self.cache_depth):
            self.frame_queue.append(i)
                
        # Populate argument iterables
        for name in arguments.iterkeys():
            
            # This object is arg.
            arg = arguments[name]
            
        self.ultimate_args = ArrayContainer(array(list(arguments.variables), dtype=object))
        self.N_args = len(self.ultimate_args)
        
        # Initialize caches
        self.cached_args = zeros(self.cache_depth * self.N_args, dtype=object)
        self.cached_values = []
        for i in range(self.cache_depth):
            self.cached_values.append(None)
        
        # Initialize current values and caches to None
        self.cached_args[:] = None

        # Underlying function
        self.fun = fun
        
        # Get pointers from arrays. This has to be a separate method
        # because __init__ can't be a C method.
        self.get_array_data()
    
    
    cdef void get_array_data(self):
        """
        Get underlying pointers from arrays ultimate_args, ultimate_keys,
        ultimate_arg_values, and cached_args.
        
        Although the underlying pointers are PyObject**, they're cast to 
        void** because Pyrex doesn't let you work with object* pointers.
        """
        self.ultimate_arg_p = <void**> PyArray_DATA(self.ultimate_args)
        self.ultimate_arg_value_p = <void**> PyArray_DATA(self.ultimate_args.value)
        self.cached_arg_p = <void**> PyArray_DATA(self.cached_args)
            
    
    cdef int check_argument_caches(self):
        """
        Compare the value of the ultimate arguments to the values in the cache.
        If there's a mismatch (match is by reference, not value), return -1.
        Otherwise return 0.
        """
        cdef int i, j, mismatch
        
        # Refresh ultimate args' values
        ACValue(self.ultimate_args)
        
        if self.cache_depth > 0:
            
            for i from 0 <= i < self.cache_depth:
                mismatch = 0
            
                for j from 0 <= j < self.N_args:
                    if not self.ultimate_arg_value_p[j] == self.cached_arg_p[i * self.N_args + j]:
                        mismatch = 1
                        break
                    
                if mismatch == 0:
                    # Find i in the frame queue and move it to the end, so the current
                    # value gets overwritten late.
                    for j from 0 <= j < self.cache_depth:
                        if self.frame_queue[j] == i:
                            break
                    self.frame_queue.pop(j)
                    self.frame_queue.append(i)
                    return i        

        return -1;

            
        
    cdef void cache(self, value):
        """
        Stick self's value in the cache, and also the values of all self's
        ultimate arguments for future caching.
        """
        cdef int i, j, cur_frame
        
        # print 'cache called'
        
        if self.cache_depth > 0:
            
            cur_frame = self.frame_queue.pop(0)
            self.frame_queue.append(cur_frame)
        
            # Store new
            self.cached_values[cur_frame] = value
            for j from 0 <= j < self.N_args:
                self.cached_args[cur_frame * self.N_args + j] = <object> self.ultimate_arg_value_p[j]
    
    def force_compute(self):
        """
        For debugging purposes. Skip cache checking and compute a value.
        """
        # Refresh parents' values
        value = self.fun(**self.arguments.value)

        # Refresh ultimate args' values
        ACValue(self.ultimate_args)

        # Cache
        self.cache(value)
        

    def get(self):
        """
        Call this method to cause the LazyFunction to refresh its arguments'
        values, decide whether it needs to recompute, and return its current
        value.
        """
        cdef int match_index
        # print 'get called'
        
        match_index = self.check_argument_caches()

        if match_index < 0:

            #Recompute
            value = self.fun(**self.arguments.value)
            self.cache(value)
        
        else:
            value = <object> self.cached_values[match_index]

        return value