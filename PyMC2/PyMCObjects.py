from copy import deepcopy
from numpy import array

class PyMCBase(object):
    def __init__(self, doc, name, parents, cache_depth, trace):

        self.parents = parents
        self.children = set()
        self.__doc__ = doc
        self.__name__ = name
        self._value = None
        self.trace = trace

        self._cache_depth = cache_depth
        
        # Timestamp is an `array scalar' so it stays put in memory.
        # That means self._node_parent_counters and 
        # self._param_parent_counters can contain references
        # directly to the parents' counter objects and skip the
        # attribute lookup.     
        self.counter = array(0,dtype=int)

        
class Node(PyMCBase):

    def __init__(self, eval,  doc, name, parents, trace=True, cache_depth=2):

        PyMCBase.__init__(self, 
                                doc=doc, 
                                name=name, 
                                parents=parents, 
                                cache_depth = cache_depth, 
                                trace=trace)

        # This function gets used to evaluate self's value.
        self._eval_fun = deepcopy(eval)
        
        from LazyFunction import LazyFunction       
        self._value = LazyFunction(fun = eval, arguments = parents.copy(), cache_depth = cache_depth, owner = self)

    def get_value(self):
        tup = self._value.get()
        if tup[0] < 0:
            # Recompute happened
            self.counter += 1
        
        return tup[1]
        
    def set_value(self,value):
        raise AttributeError, 'Node '+self.__name__+'\'s value cannot be set.'

    value = property(fget = get_value, fset=set_value)
    

class Parameter(PyMCBase):

    def __init__(   self, 
                    logp, 
                    doc, 
                    name, 
                    parents, 
                    random = None, 
                    trace=True, 
                    value=None, 
                    rseed=False, 
                    isdata=False,
                    cache_depth=2):

        PyMCBase.__init__(  self, 
                            doc=doc, 
                            name=name, 
                            parents=parents, 
                            cache_depth=cache_depth, 
                            trace=trace)

        # A flag indicating whether self's value has been observed.
        self.isdata = isdata
        
        # This function will be used to evaluate self's log probability.
        self._logp_fun = deepcopy(logp)
        
        # This function will be used to draw values for self conditional on self's parents.
        self._random = random
            
        arguments = parents.copy()
        arguments['value'] = self

        from LazyFunction import LazyFunction
        self._logp = LazyFunction(fun = logp, arguments = arguments, cache_depth = cache_depth, owner = self)       
        
        # A seed for self's rng. If provided, the initial value will be drawn. Otherwise it's
        # taken from the constructor.
        self.rseed = rseed
        if self.rseed and self._random:
            self._value = self.random()
        else:
            self._value = value     

    #
    # Define value attribute
    #   
    def get_value(self):
        return self._value

    # Record new value and increment counter
    def set_value(self, value):
        
        # Value can't be updated if isdata=True
        if self.isdata:
            raise AttributeError, self.__name__+'\'s Value cannot be updated if isdata flag is set'
            
        # Save current value as last_value
        self.last_value = self._value
        self._value = value
        
        self.counter += 1

    value = property(fget=get_value, fset=set_value)

        
    def get_logp(self):
        return self._logp.get()[1]
    
    def set_logp(self):
        raise AttributeError, 'Parameter '+self.__name__+'\'s logp attribute cannot be set'

    logp = property(fget = get_logp, fset=set_logp)


    #
    # Call this when rejecting a jump.
    #
    def revert(self):
        """
        Call this when rejecting a jump.
        """
        self._value = self.last_value
        self.counter -= 1

    #
    # Sample self's value conditional on parents.
    #
    
    def random(self):
        """
        Sample self conditional on parents.
        """
        if self._random:
            args = self._logp.argument_values.copy()
            del args['value']
            self.value = self._random(**args)
        else:
            raise AttributeError, 'Parameter '+self.__name__+' does not know how to draw its value, see documentation'
        return self._value

