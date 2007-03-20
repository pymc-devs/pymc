# FIXME: No seg faults, but with caching on test_fast is a touch slower
# FIXME: than with caching off. Overwrite the check_cache and cache methods
# FIXME: in pure C. Put the _data arrays in the pure C constructor, too,
# FIXME: so you can make them PyObject** instead of void**. Easier.
# FIXME: Nah, that sucks. Figure out how to get PyObject* into this script,
# FIXME: do those methods in Pyrex-C but without any Python commands.
# FIXME: Actually just do the check_argument_cache one that way, to avoid
# FIXME: having to deal with too many increfs.

from numpy import array, zeros, ones, arange, resize
from PyMC2 import PyMCBase, ContainerBase

cdef class LazyFunction:
    
    cdef public object arguments, fun, argument_values
    cdef object pymc_object_args, other_args
    cdef int cache_depth, N_pymc, N_other
    cdef object ult_pymc_args, ult_other_args, cached_values
    cdef object ult_pymc_arg_cache, ult_other_arg_cache
    
    cdef void *ult_pymc_arg_cache_data
    cdef void *ult_other_arg_cache_data
    cdef void *ult_pymc_arg_data
    cdef void *ult_other_arg_data    

    def __init__(self, fun, arguments, cache_depth):
        
        cdef object arg, name
        
        self.arguments = arguments
        self.cache_depth = cache_depth
        
        self.ult_pymc_args = []
        self.ult_other_args = []
        
        self.pymc_object_args = {}
        self.other_args = {}
        
        for name in arguments.iterkeys():
            arg = arguments[name]
            if isinstance(arg, ContainerBase):
                self.ult_pymc_args.extend(list(arg.pymc_objects))
                self.ult_other_args.extend(list(arg.other_objects))
            elif isinstance(arg, PyMCBase):
                self.ult_pymc_args.append(arg)
                self.pymc_object_args[name] = arg
            else:
                self.ult_other_args.append(arg)
                self.other_args[name] = arg
        
        self.ult_pymc_args = array(self.ult_pymc_args, dtype=object)
        self.ult_other_args = array(self.ult_other_args, dtype=object)
        
        try:        
            self.N_pymc = len(self.ult_pymc_args)
        except TypeError:
            self.N_pymc = 0
        
        try:
            self.N_other = len(self.ult_other_args)
        except TypeError:
            self.N_other = 0
        
        self.ult_pymc_arg_cache = zeros((self.cache_depth, self.N_pymc), dtype=object)
        self.ult_other_arg_cache = zeros((self.cache_depth, self.N_other), dtype=object)        

        # Caches of recent computations of self's value
        self.cached_values = zeros(self.cache_depth, dtype=object)

        self.fun = fun
        
        self.argument_values = {}
        
        self.ult_other_arg_cache_data = <void*> self.ult_other_arg_cache.data
        self.ult_pymc_arg_data = <void*> self.ult_pymc_args.data
        self.ult_pymc_arg_cache_data = <void*> self.ult_pymc_arg_cache.data
        self.ult_other_arg_data = <void*> self.ult_other_args.data

    # See if a recompute is necessary.
    cdef int check_argument_caches(self):
        cdef int i, mismatch

        for i from 0 <= i < self.cache_depth:
            mismatch = 0
            
            for j from 0 <= j < self.N_pymc:
                if not self.ult_pymc_args[j].value is self.ult_pymc_arg_cache[i,j]:
                    mismatch = 1
                    break
            
            if mismatch == 0:
                for j from 0 <= j < self.N_other:
                    if not self.ult_other_args[j] is self.ult_other_arg_cache[i,j]:
                        mismatch = 1
                        break
                        
            if mismatch == 0:
                return i        

        return -1;

    # Extract the values of arguments that are PyMC objects or containers.
    # Don't worry about unpacking the containers, see their value attribute.
    def refresh_argument_values(self):
        
        cdef object item
        
        for item in self.pymc_object_args.iteritems():
            self.argument_values[item[0]] = item[1].value
        for item in self.other_args.iteritems():
            self.argument_values[item[0]] = item[1]

    cdef void cache(self, value):        
        
        cdef int i, j
        
        for i from 0 <= i < self.cache_depth-1:
            self.cached_values[i+1] = self.cached_values[i]
            for j from 0 <= j < self.N_pymc:            
                self.ult_pymc_arg_cache[i+1,j] = self.ult_pymc_arg_cache[i,j]
            for j from 0 <= j < self.N_other:                
                self.ult_other_arg_cache[i+1,j] = self.ult_other_arg_cache[i,j]

        self.cached_values[0] = value
        for j from 0 <= j < self.N_pymc:
            self.ult_pymc_arg_cache[0,j] = self.ult_pymc_args[j].value
        for j from 0 <= j < self.N_other:
            self.ult_other_arg_cache[0,j] = self.ult_other_args[j]

    def get(self):
        
        cdef int match_index
        
        self.refresh_argument_values()
        match_index = self.check_argument_caches()

        if match_index < 0:

            #Recompute
            value = self.fun(**self.argument_values)

            self.cache(value)

        else: value = self.cached_values[match_index]

        return value