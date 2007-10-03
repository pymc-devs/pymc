__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from copy import deepcopy, copy
from numpy import array, ndarray, reshape, Inf
from PyMCBase import PyMCBase, ZeroProbability, Variable, PotentialBase, ParameterBase, NodeBase
from Container import DictContainer, ContainerBase

d_neg_inf = float(-1.79E308)

# from PyrexLazyFunction import LazyFunction
from LazyFunction import LazyFunction

class ParentDict(DictContainer):
    """
    A special subclass of DictContainer which makes it safe to change parameters'
    and nodes' parents. When __setitem__ is called, a ParentDict instance
    removes its owner from the old parent's children set (if appropriate)
    and adds its owner to the new parent's children set. It then asks
    its owner to generate a new LazyFunction instance using its new
    parents.

    NB: SamplingMethod and Model are expecting parameters' and nodes'
    children to be static. If you want to change indedependence structure
    over the course of an MCMC loop, please do so with indicator variables.
    
    :SeeAlso: DictContainer
    """
    def __init__(self, regular_dict, owner):
        DictContainer.__init__(self, regular_dict)
        self.owner = owner

    def __setitem__(self, key, new_parent):

        old_parent = self[key]

        # Possibly remove me from old parent's children set.
        dict.__setitem__(self, key, new_parent)
        
        if isinstance(old_parent, Variable) or isinstance(old_parent, ContainerBase):
            self.val_keys.remove(key)
            self.nonval_keys.append(key)
        
        if isinstance(old_parent, PyMCBase):
            only_reference = True
            if isinstance(new_parent, Variable) or isinstance(new_parent, ContainerBase):
                self.val_keys.append(key)
                self.nonval_keys.remove(key)
            

            # See if I only claim the old parent via this key.
            for item in self.iteritems():
                if item[0] is old_parent and not item[1] == key:
                    only_reference = False
                    break

            # If so, remove me from the old parent's children set.
            if only_reference:
                old_parent.children.remove(self.owner)

        # If the new parent is a PyMC object, add me to its children set.
        if isinstance(new_parent, PyMCBase):
            new_parent.children.add(self.owner)

        # Tell my owner it needs a new lazy function.
        self.owner.gen_lazy_function()

class Potential(PotentialBase):
    """
    Not a variable; just an arbitrary log-probability term to multiply into the 
    joint distribution. Useful for expressing models that aren't directed, such as
    Markov random fields.

    Decorator instantiation:

    @potential(trace = True)
    def A(x = B, y = C):
        return -.5 * (x-y)**2 / 3.

    Direct instantiation:

    :Arguments:

        -logp: function
              The function that computes the potential's value from the values 
              of its parents.

        -doc: string
              The docstring for this potential.

        -name: string
              The name of this potential.

        -parents: dictionary
              A dictionary containing the parents of this node.

        -cache_depth (optional): integer
              An integer indicating how many of this potential's value computations 
              should be 'memoized'.
                                    
        - verbose (optional) : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

                            
    Externally-accessible attribute:

        -logp: float
              Returns the node's value given its parents' values. Skips
              computation if possible.
            
    No methods.
    
    :SeeAlso: Parameter, PyMCBase, LazyFunction, parameter, node, data, Model, Container
    """
    def __init__(self, logp,  doc, name, parents, cache_depth=2, verbose=0):
        
        self._parents = ParentDict(regular_dict = parents, owner = self)

        # This function gets used to evaluate self's value.
        self._logp_fun = logp
        
        PyMCBase.__init__(self, 
                            doc=doc, 
                            name=name, 
                            parents=parents, 
                            cache_depth = cache_depth, 
                            trace=False,
                            verbose=verbose)

        self.zero_logp_error_msg = "Potential " + self.__name__ + "forbids its parents' current values."

        self._logp.force_compute()

        # Check initial value
        if not isinstance(self.logp, float):
            raise ValueError, "Potential " + self.__name__ + "'s initial log-probability is %s, should be a float." %self.logp.__repr__()

    def gen_lazy_function(self):
        self._logp = LazyFunction(fun = self._logp_fun, arguments = self.parents, cache_depth = self._cache_depth)        

    def get_logp(self):
        if self.verbose > 2:
            print '\t' + self.__name__ + ': log-probability accessed.'
        _logp = self._logp.get()
        if self.verbose > 2:
            print '\t' + self.__name__ + ': Returning log-probability ', _logp

        # Check if the value is smaller than a double precision infinity:
        if _logp <= d_neg_inf:
            raise ZeroProbability, self.zero_logp_error_msg

        return _logp
        
    def set_logp(self,value):      
        raise AttributeError, 'Potential '+self.__name__+'\'s log-probability cannot be set.'

    logp = property(fget = get_logp, fset=set_logp)


        
class Node(NodeBase):
    """
    A variable whose value is determined by the values of its parents.

    
    Decorator instantiation:

    @node(trace=True)
    def A(x = B, y = C):
        return sqrt(x ** 2 + y ** 2)
    
    Direct instantiation:

    :Arguments:

        -eval: function
              The function that computes the node's value from the values 
              of its parents.

        -doc: string
              The docstring for this node.

        -name: string
              The name of this node.

        -parents: dictionary
              A dictionary containing the parents of this node.

        -trace (optional): boolean
              A boolean indicating whether this node's value 
              should be traced (in MCMC).

        -cache_depth (optional): integer  
              An integer indicating how many of this node's
              value computations should be 'memorized'.
                                    
        - verbose (optional) : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

                            
    Externally-accessible attribute:

        -value: any class
              Returns the node's value given its parents' values. Skips
              computation if possible.
            
    No methods.
    
    :SeeAlso: Parameter, PyMCBase, LazyFunction, parameter, node, data, Model, Container
    """
    def __init__(self, eval,  doc, name, parents, trace=True, cache_depth=2, verbose=0):

        self._parents = ParentDict(regular_dict = parents, owner = self)

        # This function gets used to evaluate self's value.
        self._eval_fun = eval
        
        PyMCBase.__init__(self, 
                            doc=doc, 
                            name=name, 
                            parents=parents, 
                            cache_depth = cache_depth, 
                            trace=trace,
                            verbose=verbose)
        
        self._value.force_compute()
        if self.value is None:
            print  "WARNING: Node " + self.__name__ + "'s initial value is None"
        
    def gen_lazy_function(self):
        # self._value = self.LazyFunction(fun = self._eval_fun, arguments = self.parents, cache_depth = self._cache_depth)
        self._value = LazyFunction(fun = self._eval_fun, arguments = self.parents, cache_depth = self._cache_depth)        

    def get_value(self):
        if self.verbose > 2:
            print '\t' + self.__name__ + ': value accessed.'
        _value = self._value.get()
        if isinstance(_value, ndarray):
            _value.flags['W'] = False
        if self.verbose > 2:
            print '\t' + self.__name__ + ': Returning value ',_value
        return _value
        
    def set_value(self,value):      
        raise AttributeError, 'Node '+self.__name__+'\'s value cannot be set.'

    value = property(fget = get_value, fset=set_value)
    

class Parameter(ParameterBase):
    
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

    - logp : function   
            The function that computes the parameter's log-probability from
            its value and the values of its parents.

    - doc : string    
            The docstring for this parameter.

    - name : string   
            The name of this parameter.

    - parents: dict
            A dictionary containing the parents of this parameter.
    
    - random (optional) : function 
            A function that draws a new value for this
            parameter given its parents' values.

    - trace (optional) : boolean   
            A boolean indicating whether this node's value 
            should be traced (in MCMC).
                        
    - value (optional) : number or array  
            An initial value for this parameter
    
    - rseed (optional) : integer or rseed
            A seed for this parameter's rng. Either value or rseed must
            be given.
                        
    - isdata (optional) :  boolean
            A flag indicating whether this parameter is data; whether
            its value is known.

    - cache_depth (optional) : integer
            An integer indicating how many of this parameter's
            log-probability computations should be 'memoized'.
                            
    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

                            
    Externally-accessible attribute:

    - value: any class
          Returns this parameter's current value.

    - logp: float
          Returns the parameter's log-probability given its value and its 
          parents' values. Skips computation if possible.
            
    last_value: any class
          Returns this parameter's last value. Useful for rejecting
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
                    cache_depth=2,
                    verbose = 0):                    

        self._parents = ParentDict(regular_dict = parents, owner = self)

        # A flag indicating whether self's value has been observed.
        self.isdata = isdata
        
        # This function will be used to evaluate self's log probability.
        self._logp_fun = logp
        
        # This function will be used to draw values for self conditional on self's parents.
        self._random = random
        
        # A seed for self's rng. If provided, the initial value will be drawn. Otherwise it's
        # taken from the constructor.
        self.rseed = rseed
        
        # Initialize value, either from value provided or from random function.
        self._value = value
        if value is None:
            
            # Use random function if provided
            if random is not None:
                self._value = random(**self._parents.value)
                
            # Otherwise leave initial value at None and warn.
            else:
                raise ValueError, 'Parameter ' + name + "'s value initialized to None; no initial value or random method provided."
        
        PyMCBase.__init__(  self, 
                            doc=doc, 
                            name=name, 
                            parents=parents, 
                            cache_depth=cache_depth, 
                            trace=trace,
                            verbose=verbose)
        self._logp.force_compute()                   
        self.zero_logp_error_msg = "Parameter " + self.__name__ + "'s value is outside its support."

        # Check initial value
        if not isinstance(self.logp, float):
            raise ValueError, "Parameter " + self.__name__ + "'s initial log-probability is %s, should be a float." %self.logp.__repr__()
            
                
    def gen_lazy_function(self):
        """
        Will be called by PyMCBase at instantiation.
        """
        arguments = {}
        arguments.update(self.parents)
        arguments['value'] = self
        arguments = ParentDict(arguments, owner=self)
        
        # self._logp = self.LazyFunction(fun = self._logp_fun, arguments = arguments, cache_depth = self._cache_depth)        
        self._logp = LazyFunction(fun = self._logp_fun, arguments = arguments, cache_depth = self._cache_depth)                
    
    def get_value(self):
        # Define value attribute
        
        if self.verbose > 2:
            print '\t' + self.__name__ + ': value accessed.'
        return self._value

    
    def set_value(self, value):
        # Record new value and increment counter
        
        if self.verbose > 2:
            print '\t' + self.__name__ + ': value set to ', value
        
        # Value can't be updated if isdata=True
        if self.isdata:
            raise AttributeError, 'Parameter '+self.__name__+'\'s value cannot be updated if isdata flag is set'
            
        if isinstance(value, ndarray):
            value.flags['W'] = False            
            
        # Save current value as last_value
        # Don't copy because caching depends on the object's reference. 
        self.last_value = self._value
        self._value = value
        

    value = property(fget=get_value, fset=set_value)


    def get_logp(self):
        if self.verbose > 2:
            print '\t' + self.__name__ + ': logp accessed.'
        logp = self._logp.get()
        if self.verbose > 2:
            print '\t' + self.__name__ + ': Returning log-probability ', logp
        
        # Check if the value is smaller than a double precision infinity:
        if logp <= d_neg_inf:
            raise ZeroProbability, self.zero_logp_error_msg
    
        return logp

    def set_logp(self):
        raise AttributeError, 'Parameter '+self.__name__+'\'s logp attribute cannot be set'

    logp = property(fget = get_logp, fset=set_logp)        

    
    # Sample self's value conditional on parents.
    def random(self):
        """
        Draws a new value for a parameter conditional on its parents
        and returns it.
        
        Raises an error if no 'random' argument was passed to __init__.
        """
        
        if self._random:
            # Get current values of parents for use as arguments for _random()
            r = self._random(**self.parents.value)
        else:
            raise AttributeError, 'Parameter '+self.__name__+' does not know how to draw its value, see documentation'
        
        # Set Parameter's value to drawn value
        if self.isdata is False:
            self.value = r
        return r

class DiscreteParameter(Parameter):
    """
    A subclass of Parameter that takes only integer values.
    """
        
    def set_value(self, value):
        # Record new value and increment counter
        
        # Value can't be updated if isdata=True
        if self.isdata:
            raise AttributeError, 'Parameter '+self.__name__+'\'s value cannot be updated if isdata flag is set'
            
        # Save current value as last_value
        self.last_value = self._value
        self._value = int(round(value, 0))
        
    
class BinaryParameter(Parameter):
    """
    A subclass of Parameter that takes only boolean values.
    """
    
    def set_value(self, value):
        # Record new value and increment counter
        
        # Value can't be updated if isdata=True
        if self.isdata:
            raise AttributeError, 'Parameter '+self.__name__+'\'s value cannot be updated if isdata flag is set'
            
        # As a first shot at this, all non-zero values are coded as unity and 
        # zero as zero. Better way? (e.g. positives as 1, otherwise zero)
            
        # Save current value as last_value
        self.last_value = self._value
        self._value = bool(value)
