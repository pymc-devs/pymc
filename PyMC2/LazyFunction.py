__docformat__='reStructuredText'

from numpy import array, zeros, ones, arange, resize
from PyMCBase import PyMCBase, ContainerBase

class LazyFunction(object):

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
                
    cache_depth:    The number of prior computations to 'memorize' in
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
    
    def __init__(self, fun, arguments, cache_depth):
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
                
        self.N_pymc = len(self.ult_pymc_args)
        self.N_other = len(self.ult_other_args)
        
        self.ult_pymc_arg_cache = zeros((self.cache_depth, self.N_pymc), dtype=object)
        self.ult_other_arg_cache = zeros((self.cache_depth, self.N_other), dtype=object)        

        # Caches of recent computations of self's value
        self.cached_values = zeros(self.cache_depth, dtype=object)

        self.fun = fun
        
        self.argument_values = {}

    # See if a recompute is necessary.
    def check_argument_caches(self):

        for i in xrange(self.cache_depth):
            mismatch = False
            
            for j in xrange(self.N_pymc):
                if not self.ult_pymc_args[j].value is self.ult_pymc_arg_cache[i,j]:
                    mismatch = True
                    break
            
            if not mismatch:
                for j in xrange(self.N_other):
                    if not self.ult_other_args[j] is self.ult_other_arg_cache[i,j]:
                        mismatch=True
                        break
                        
            if not mismatch:
                return i        

        return -1;

    # Extract the values of arguments that are PyMC objects or containers.
    # Don't worry about unpacking the containers, see their value attribute.
    def refresh_argument_values(self):
        """
        Iterates over LazyFunction's parents that are Parameters,
        Nodes, or Containers and records their current values
        for passing to self's internal function.
        """
        for item in self.pymc_object_args.iteritems():
            self.argument_values[item[0]] = item[1].value
        for item in self.other_args.iteritems():
            self.argument_values[item[0]] = item[1]

    def cache(self, value):        
        for i in xrange(self.cache_depth-1):
            self.cached_values[i+1] = self.cached_values[i]
            self.ult_pymc_arg_cache[i+1,:] = self.ult_pymc_arg_cache[i,:]
            self.ult_other_arg_cache[i+1,:] = self.ult_other_arg_cache[i,:]

        self.cached_values[0] = value
        for j in xrange(self.N_pymc):
            self.ult_pymc_arg_cache[0,j] = self.ult_pymc_args[j].value
        for j in xrange(self.N_other):
            self.ult_other_arg_cache[0,j] = self.ult_other_args[j]

    def get(self):
        """
        Call this method to cause the LazyFunction to refresh its arguments'
        values, decide whether it needs to recompute, and return its current
        value.
        """
        self.refresh_argument_values()
        recomp = self.check_argument_caches()

        if recomp < 0:

            #Recompute
            value = self.fun(**self.argument_values)

            self.cache(value)

        else: value = self.cached_values[recomp]

        return value
