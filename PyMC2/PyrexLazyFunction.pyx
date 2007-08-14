# Note, there was an intermittent problem with caching and cache checking
# when round_array rounded scalars instead of casting them to int. Forbidden
# values were getting cached with positive associated probability, leading to
# uncaught ZeroProbabilitys in OneAtATimeMetropolis. I'm not sure what the 
# deal was. The problem went away when round changed to int.

__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from numpy import array, zeros, ones, arange, resize
from PyMC2 import PyMCBase, ContainerBase, Variable

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
                to its function. If any of the arguments is a Parameter,
                Node, or Container, that argument's 'value' attribute
                will be substituted for it when passed to fun.
                
    cache_depth:    The number of prior computations to 'memoize' in
                    order to skip unnecessary computations.
    
    
                    
    Externally-accessible methods:
    
    refresh_argument_values():  Iterates over LazyFunction's parents that are 
                                Parameters, Nodes, or Containers and records their 
                                current values for passing to self's internal 
                                function.
    
    get():  Refreshes argument values, checks cache, calls internal function if
            necessary, returns value.
            
    
    
    Externally-accessible attributes:
    
    arguments:  A dictionary containing self's arguments.

    fun:        Self's internal function.
    
    argument_values:    A dictionary containing self's arguments, in which
                        Parameters, Nodes, and Containers have been
                        replaced by their 'value' attributes.
    
                
    
    :SeeAlso: Parameter, Node, Container
    """
    
    cdef public object arguments, fun, argument_values
    cdef public object variable_args, other_args
    cdef int cache_depth, N_args, 
    cdef public object ultimate_args, ultimate_keys, ultimate_arg_values
    cdef void **ultimate_arg_p, **ultimate_keys_p, **ultimate_arg_value_p
    cdef public object cached_args, cached_values
    cdef void **cached_arg_p
    
    cdef public object frame_queue
    
    def __init__(self, fun, arguments, cache_depth):
        
        cdef object arg, name
        cdef int i
        
        # arguments will be parents and value for parameters, just parents for nodes.
        self.arguments = arguments
        
        self.cache_depth = cache_depth
        
        # A dictionary containing the arguments that are parameters and nodes, with the same keys
        # as arguments.
        self.variable_args = {}
        # dictionary containing the other arguments.
        self.other_args = {}
        
        self.frame_queue = []
        for i in xrange(self.cache_depth):
            self.frame_queue.append(i)
        
        # The ultimate arguments are the arguments, plus the contents of.
        # any argument that is a container.
        self.ultimate_args=[]
        # The names corresponding to the ultimate arguments.
        self.ultimate_keys=[]

        # A dictionary of values to pass to the underlying function.
        self.argument_values = {}
        
        # Populate argument iterables
        for name in arguments.iterkeys():
            
            # This object is arg.
            arg = arguments[name]

            # If arg is a container, it and all of its contents go into
            # ultimate_args.
            if isinstance(arg, ContainerBase):
                
                self.argument_values[name] = arg.value
                self.ultimate_args.append(arg)
                self.ultimate_keys.append(name)                
                
                for obj in arg.variables:
                    self.ultimate_args.append(obj)
                    self.ultimate_keys.append(None)
            
            # If arg is a parameter or node, it goes into
            # variable_args and ultimate_args.    
            elif isinstance(arg, Variable):
                self.variable_args[name] = arg
                self.ultimate_args.append(arg)
                self.ultimate_keys.append(name)
                self.argument_values[name] = arg.value
            
            # If arg is neither parameter, node, nor container, it goes
            # into other_args.    
            else:
                self.other_args[name] = arg
                self.argument_values[name] = arg
        
        # Change ultimate_args and ultimate_keys from lists to arrays.
        # Arrays of dtype object are wrappers for continuous blocks
        # of PyObject* pointers.        
        self.ultimate_args = array(self.ultimate_args, dtype=object)
        self.ultimate_keys = array(self.ultimate_keys, dtype=object)
                
        self.N_args = len(self.ultimate_args)

        # Initialize container for current values        
        self.ultimate_arg_values = zeros(self.N_args, dtype=object)
        
        # Initialize caches
        self.cached_args = zeros(self.cache_depth * self.N_args, dtype=object)
        self.cached_values = []
        for i in range(self.cache_depth):
            self.cached_values.append(None)
        
        # Initialize current values and caches to None
        self.ultimate_arg_values[:] = None
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
        self.ultimate_keys_p = <void**> PyArray_DATA(self.ultimate_keys)
        self.ultimate_arg_value_p = <void**> PyArray_DATA(self.ultimate_arg_values)
        self.cached_arg_p = <void**> PyArray_DATA(self.cached_args)
        
    
    cdef int check_argument_caches(self):
        """
        Compare the value of the ultimate arguments to the values in the cache.
        If there's a mismatch (match is by reference, not value), return -1.
        Otherwise return 0.
        """
        cdef int i, j, mismatch
        
        
        if self.cache_depth > 0:
            
            for i from 0 <= i < self.cache_depth:
                mismatch = 0
            
                for j from 0 <= j < self.N_args:
                    # self.ultimate_arg_value_p[j] and self.cached_arg_p[i * self.N_args + j] are
                    # both void*, so comparing them _should_ be equivalent to casting them to PyObject*
                    # (object) and comparing them with is. Maybe do that instead? I seem to remember that
                    # not working.
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


    def refresh_argument_values(self):
        """
        Iterates over LazyFunction's parents that are Parameters,
        Nodes, or Containers and records their current values
        for passing to self's internal function.
        """

        cdef object item
        cdef int i
        cdef object name
        
        for i from 0 <= i < self.N_args:
            
            # Query the value of all ultimate arguments, even those not in the
            # argument list, and store their references for cache checking.
            item = self.ultimate_args[i].value
            self.ultimate_arg_values[i] = item

            name = self.ultimate_keys[i]

            # If name is not None, this ultimate argument is in the argument dictionary.
            # Record its value for possible passing to the underlying function.
            if name is not None:
                self.argument_values[name] = item
            
        
    cdef void cache(self, value):
        """
        Stick self's value in the cache, and also the values of all self's
        ultimate arguments for future caching.
        """
        cdef int i, j, cur_frame
        
        # print 'cache called'
        
        if self.cache_depth > 0:
            
            cur_frame = self.frame_queue.pop(0)
            # print cur_frame
            # print self.frame_queue
            self.frame_queue.append(cur_frame)
            # print self.frame_queue
            
            # self.cached_values.pop()
            # self.cached_values.insert(0,value)
            #         
            # # Push back
            # for i from 0 <= i < self.cache_depth - 1:
            #     for j from 0 <= j < self.N_args:
            #         # It SHOULD be safe to pointerize this... try it eventually.
            #         # self.cached_arg_p[(i+1) * self.N_args + j] = self.cached_arg_p[i * self.N_args + j]
            #         self.cached_args[(i+1) * self.N_args + j] = self.cached_args[i * self.N_args + j]
        
            # Store new
            self.cached_values[cur_frame] = value
            for j from 0 <= j < self.N_args:
                self.cached_args[cur_frame * self.N_args + j] = self.ultimate_arg_values[j]
    
    def force_compute(self):
        """
        For debugging purposes. Skip cache checking and compute a value.
        """
        self.refresh_argument_values()
        value = self.fun(**self.argument_values)
        self.cache(value)
        

    def get(self):
        """
        Call this method to cause the LazyFunction to refresh its arguments'
        values, decide whether it needs to recompute, and return its current
        value.
        """
        cdef int match_index
        # print 'get called'
        
        self.refresh_argument_values()
        match_index = self.check_argument_caches()
        # match_index = -1

        if match_index < 0:

            #Recompute
            value = self.fun(**self.argument_values)

            self.cache(value)
        
        else: value = <object> self.cached_values[match_index]

        return value