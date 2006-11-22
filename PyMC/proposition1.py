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



# Testing zone
from test_decorator import normal_like
@parameter
def beta(self):
    """Parameter beta of toy model."""
    # Initial value
    return normal_like(self, 5,2)
