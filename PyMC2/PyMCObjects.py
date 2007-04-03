__docformat__='reStructuredText'

from copy import deepcopy, copy
from numpy import array, ndarray, reshape
from PyMCBase import PyMCBase, ParentDict

def import_LazyFunction():
    try:
        LazyFunction
    except NameError:
        try:
            from PyrexLazyFunction import LazyFunction
        except:
            from LazyFunction import LazyFunction
            print 'Using pure lazy functions'
    return LazyFunction
        
class Node(PyMCBase):
    """
    A variable whose value is determined by the values of its parents.

    
    Decorator instantiation:

    @node(trace=True)
    def A(x = B, y = C):
        return sqrt(x ** 2 + y ** 2)

    
    Direct instantiation:

    :Arguments:

        -eval:  The function that computes the node's value from the values 
                of its parents.

        -doc:    The docstring for this node.

        -name:   The name of this node.

        -parents: A dictionary containing the parents of this node.

        -trace (optional):  A boolean indicating whether this node's value 
                            should be traced (in MCMC).

        -cache_depth (optional):    An integer indicating how many of this node's
                                    value computations should be 'memoized'.

                            
    Externally-accessible attribute:

        -value: Returns the node's value given its parents' values. Skips
                computation if possible.
            
    No methods.
    
    :SeeAlso: Parameter, PyMCBase, LazyFunction, parameter, node, data, Model, Container
    """
    def __init__(self, eval,  doc, name, parents, trace=True, cache_depth=2):
        
        self.LazyFunction = import_LazyFunction()

        # This function gets used to evaluate self's value.
        self._eval_fun = eval
        
        PyMCBase.__init__(self, 
                                doc=doc, 
                                name=name, 
                                parents=parents, 
                                cache_depth = cache_depth, 
                                trace=trace)        
        
    def gen_lazy_function(self):
        self._value = self.LazyFunction(fun = self._eval_fun, arguments = self.parents, cache_depth = self.cache_depth)

    def get_value(self):
        return self._value.get()
        
    def set_value(self,value):
        raise AttributeError, 'Node '+self.__name__+'\'s value cannot be set.'

    value = property(fget = get_value, fset=set_value)
    

class Parameter(PyMCBase):
    
    """
    A variable whose value is not determined by the values of its parents.

    
    Decorator instantiation:

    @parameter(trace=True)
    def X(value = 0., mu = B, tau = C):
        return Normal_like(value, mu, tau)
        
    @parameter(trace=True)
    def X(value=0., mu=B, tau=C):

        def logp(value, mu, tau):
            return Normal_like(value, mu, tau)
        
        def random(mu, tau):
            return Normal_r(mu, tau)
            
        rseed = 1.

    
    Direct instantiation:

    :Arguments:

    logp:   The function that computes the parameter's log-probability from
            its value and the values of its parents.

    doc:    The docstring for this parameter.

    name:   The name of this parameter.

    parents: A dictionary containing the parents of this parameter.
    
    random (optional):  A function that draws a new value for this
                        parameter given its parents' values.

    trace (optional):   A boolean indicating whether this node's value 
                        should be traced (in MCMC).
                        
    value (optional):   An initial value for this parameter
    
    rseed (optional):   A seed for this parameter's rng. Either value or rseed must
                        be given.
                        
    isdata (optional):  A flag indicating whether this parameter is data; whether
                        its value is known.

    cache_depth (optional): An integer indicating how many of this parameter's
                            log-probability computations should be 'memoized'.

                            
    Externally-accessible attribute:

    value:  Returns this parameter's current value.

    logp:   Returns the parameter's log-probability given its value and its 
            parents' values. Skips computation if possible.
            
    last_value: Returns this parameter's last value. Useful for rejecting
                Metropolis-Hastings jumps. See touch() and the warning below.
            
    Externally-accessible methods:
    
    random():   Draws a new value for this parameter from its distribution and
                returns it.
                
    touch():    If a parameter's value is changed in-place, the cache-checker will
                get confused. In addition, in MCMC, there won't be a way to reject
                the jump. If you update a parameter's value in-place, call touch()
                immediately afterward.
                    
    :SeeAlso: Node, PyMCBase, LazyFunction, parameter, node, data, Model, Container
    """
    
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

        self.LazyFunction = import_LazyFunction()    

        # A flag indicating whether self's value has been observed.
        self.isdata = isdata
        
        # This function will be used to evaluate self's log probability.
        self._logp_fun = logp
        
        # This function will be used to draw values for self conditional on self's parents.
        self._random = random
        
        PyMCBase.__init__(  self, 
                            doc=doc, 
                            name=name, 
                            parents=parents, 
                            cache_depth=cache_depth, 
                            trace=trace)        
        
        # A seed for self's rng. If provided, the initial value will be drawn. Otherwise it's
        # taken from the constructor.
        self.rseed = rseed
        if self.rseed and self._random:
            self._value = self.random()
        else:
            self._value = value     
    
    def gen_lazy_function(self):
        
        arguments = self.parents.copy()
        arguments['value'] = self

        self._logp = self.LazyFunction(fun = self._logp_fun, arguments = arguments, cache_depth = self._cache_depth)        


    # Define value attribute
    def get_value(self):
        return self._value

    # Record new value and increment counter
    def set_value(self, value):
        
        # Value can't be updated if isdata=True
        if self.isdata:
            raise AttributeError, 'Parameter '+self.__name__+'\'s value cannot be updated if isdata flag is set'
            
        # Save current value as last_value
        self.last_value = self._value
        self._value = value

    value = property(fget=get_value, fset=set_value)


    def get_logp(self):
        return self._logp.get()

    def set_logp(self):
        raise AttributeError, 'Parameter '+self.__name__+'\'s logp attribute cannot be set'

    logp = property(fget = get_logp, fset=set_logp)


    # If the user wants to update the parameter's value in-place,
    # they can call this to prevent the cache-checker from getting
    # confused. 
    def touch(self):
        """ 
        If you want to update a parameter's value in-place,
        you can call touch() to avoid confusing the cache 
        checker... but last_value will be set to self.value, 
        because otherwise the cache checker WOULD get confused. 
        It's safest to avoid updating parameters' values in-place.
        """
        if isinstance(self._value, ndarray):
            self.value = self.value.view()
        else:
            self.value = copy(self._value)
        self.last_value = self.value

    
    # Sample self's value conditional on parents.
    def random(self):
        """
        Draws a new value for a parameter conditional on its parents
        and returns it.
        
        Raises an error if no 'random' argument was passed to __init__.
        """
        if self._random:
            self._logp.refresh_argument_values()
            args = self._logp.argument_values.copy()
            args.pop('value')
            self.value = self._random(**args)
        else:
            raise AttributeError, 'Parameter '+self.__name__+' does not know how to draw its value, see documentation'
        return self._value
