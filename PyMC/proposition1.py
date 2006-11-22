import copy

# This is a very basic decorator that initialize the Parameter class.
# Another implementation is found in test2.py. 
def parameter(f):
    """Decorator function instantiating the Parameter class."""
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    
    P = Parameter(prior_fun=wrapper)
    P.__doc__ = f.__doc__
    
    return P
    
def data(f):
    pass
    

class Parameter(object):
    def __init__(self, prior_fun, parents={}, init_val=None):
        self.prior_fun = prior_fun
        self.parent = parents
        self.recompute = True
        if init_val:
            self._current_prior = self.get_prior(init_val)
            self._current_value = init_val
        else:      
            self._current_prior = None
            self._current_value = None
        
        #Initialize parent dictionary and caches
    
    def value():
        def fget(self, *args):
            return self._current_value
        def fset(self, value):
            self._cache_value = copy.copy(self._current_value)
            self._current_value = value
            self.recompute = True
        return locals()
        #return timestamp and self.value(args)
    value = property(**value())
    
    def get_prior(self):       
        if self.recompute:
            self._cached_prior = copy.copy(self._current_prior)
            self._current_prior = self.prior_fun(self.value, **self.parent)
        return self._current_prior
    
    def revert(self):
        self._current_prior = self._cached_prior
        self._current_value = self._cache_value
        
    def call(self, *args, **kwargs):
        if len(args) >0 or len(kwargs) > 0 :
            return self.prior_fun(*args, **kwargs)
        else:
            return self.get_prior()
    
    __call__ = call
        #Check caches, if necessary call prior_fun (which is passed into
        #__call__ by the decorator)


class Parameter1(object):
    def __init__(self, like, init_val=None, parents={}):
        self._current_value = init_val
        self._like_func = like
        self.parents = parents
        self.recompute = True
        if init_val:
            self._current_prior = self._like_func(init_val)
            self._current_value = init_val
        else:      
            self._current_prior = None
            self._current_value = None
        
        #Initialize parent dictionary and caches
    
    def value():
        def fget(self, *args):
            return self._current_value
        def fset(self, value):
            self._cache_value = copy.copy(self._current_value)
            self._current_value = value
            self.recompute = True
        return locals()
        #return timestamp and self.value(args)
    value = property(**value())
    
    def like(*args, **kw):
        return self._like_func(*args, **kw)
    __call__ = like


class Data(object):
    def __init__(self, value=0, like=0):
        self._value = value
        self._like = like
    def __call__(self):
        return self._like
    def value():
        def fget(self):
            return self._value
        return locals()
    value = property(**value())
        
def data(f):
    local_dict = f()
    D = Data(**local_dict)
    return D

# Testing zone
from test_decorator import normal_like
@Data
def input(self):
    """Measured input driving toy model."""
    # input value
    self = [1,2,3,4]
    # input likelihood     
    return 0
    
@Data
def exp_output(self):
    """Experimental output."""
    # output value
    self = [45,34,34,65]
    # likelihood a value or a function
    return 0
    
@Parameter
def alpha(self):
    """Parameter alpha of toy model."""
    # The return value is the prior. 
    return uniform_like(self, 0, 10)
    
@Node
def sim_output(self, alpha, input, exp_output):
    """Compute the simulated output and return its likelihood.
    Usage: sim_output(alpha, beta, input, exp_output)
    """
    self = alpha * input
    return normal_like(self, exp_output, 2)
