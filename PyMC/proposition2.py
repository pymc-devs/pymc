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
# When called, a Data object returns its likelihood. x
def Data1(f):
    """Decorator function for data objects."""
    answer = f()
    if type(answer) is type('function'):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
    else:
        def wrapper = lambda x: answer
    wrapper.__doc__ = f.__doc__
    return wrapper  

@Data1
def input(self):
    """Measured input driving toy model."""
    # input value
    self = [1,2,3,4]
    # input likelihood     
    return 0