# Proposition 3
# This is proposition 2, but working...

import numpy as N
from inspect import getargs

class structure:
    self = None
    pass

def parameter(*args, **kwds):
    """Decorator function for PyMC parameter.
    
    Input
        like: a function returning the likelihood of the parameter, ie its prior. 
        init_val: Initial value to start the sampling.
    
    Example
        @parameter(init_val=56)
        def alpha(self):
            "Parameter alpha of model M."
            return uniform_like(self, 0, 10)
    """
    if len(kwds) == 0:
        kwds['init_val'] = args[0]
    def wrapper(func):
        func.__dict__.update(kwds)
        func.parents = getargs(func.func_code)[0]
        func.parents.remove('self')
        return func
    return wrapper


def data(func):
    """Decorator function for PyMC data.
    
    Input
        A function taking as argument the value of the data, 
       and returning the likelihood of the data.
        
    Example
        @data
        def input(value=[1,2,3,4]):
            "Input data to force model."
            return 0
    """
    def wrapper(*args, **kwds):
        func.__dict__.update(kw)
        if len(kwds) == 0:
            func.value = args[0]
        func.parents = getargs(func.func_code)[0]
        return func
    return wrapper


def Node(func):
    """Decorator function for PyMC Node.

    A Node can set itself and return None, or
    set itself and return its likelihood.
    """
    def wrapper(*args, **kw):
        like = f(func.self, *args, **kw)
    
        
        
        
    return wrapper

# Testing
from test_decorator import normal_like, uniform_like

@parameter(4)
def alpha(self):
    """Parameter alpha of toy model."""
    # The return value is the prior. 
    return uniform_like(self, 0, 10)

@parameter(init_val=5)
def beta(self, alpha):
    """Parameter beta of toy model."""
    return normal_like(self, alpha, 2)

@data
def input(value = [1,2,3,4]):
    """Measured input driving toy model."""
    like = 0
    return like
    
@data
def exp_output(value = [45,34,34,65]):
    """Experimental output."""
    # likelihood a value or a function
    return 0
    

@Node
def sim_output(alpha, beta, input, exp_output):
    """Compute the simulated output and return its likelihood.
    Usage: sim_output(alpha, beta, input, exp_output)
    """
    self = toy_model(alpha, beta, input)
    like = normal_like(self, exp_output, 2)
    return like
##
##
##print input.value
##print input()
##
##sim_output(3,4,input.value, exp_output.value):


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
