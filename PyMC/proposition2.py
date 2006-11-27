# proposition2
# Here all functionality is taken care of by the decorator. 
# Basically, a Node, Parameter or Data object does two things:
# 1. It has an intrinsic value, 
# You can set this value by 
# >>> object.set(value) 
# You get it by typing 
# >>> object 
# 2. It has a likelihood 
# You can get this likelihood by caling the object
# >>> object() 


class structure:
    value = None
    pass
    
def Parameter(f):
    """Decorator function describing PyMC parameters."""
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    wrapper.__doc__ = f.__doc__
    def set(self, value):
        self._value = value
    def get(self):
        return self._value
    wrapper.set = set
    wrapper.__get__ = get
    return wrapper


def Node(f):
    """Decorator function describing PyMC nodes."""
    C = structure()
    def wrapper(*args, **kwargs):
        answer  = f(C, *args, **kwargs)
        wrapper._value = C.self
        return answer
    
    wrapper.__doc__ = f.__doc__
    def get(self):
        return self._value
    wrapper.__get__ = get
    return wrapper

# Data must return its own value.
# Data must have the docstring of the function object. x
# When called, a Data object returns its likelihood, which is constant.
def Data(f):
    """Decorator function for data objects."""
    C = structure()
    # Get value of self.
    f(C.value)
    def wrapper():
        return C.value
    wrapper.__doc__ = f.__doc__
    return wrapper  

# Testing
from test_decorator import normal_like, uniform_like

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
    
print input
print input()
print alpha
print alpha()
print sim_output
print sim_output()



##class Parameter(N.ndarray):
##    def __new__(subtype, data, func, info=None, dtype=None, copy=True):
##        # When data is an InfoArray
##        if isinstance(data, Parameter):
##            if not copy and dtype==data.dtype:
##                return data.view(subtype)
##            else:
##                return data.astype(dtype).view(subtype)
##        subtype._info = info
##        subtype.info = subtype._info
##        subtype._func = func
##        return N.array(data).view(subtype)
##
##    def __array_finalize__(self,obj):
##        if hasattr(obj, "info"):
##            # The object already has an info tag: just use it
##            self.info = obj.info
##        else:
##            # The object has no info tag: use the default
##            self.info = self._info
##    
##    def like(self, *args, **kw):
##        """Likelihood of parameter, or if you prefer, its prior."""
##        if len(args) > 0 or len(kw)>0:
##            return self._func(*args, **kw)
##        else:
##            return self._func(self.__array__())
##
##    def __repr__(self):
##        desc="""Parameter array( %(data)s,
##      tag=%(tag)s)"""
##        return desc % {'data': str(self), 'tag':self.info }
