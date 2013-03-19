from functools import wraps

def quickclass_explicit(baseclass): 
    def decorator(fn):
        class Clas(baseclass):
            __doc__ = fn.__doc__

            @wraps(fn)
            def __init__(self, *args, **kwargs):  #still need to figure out how to give it the right argument names
                properties = fn(*args, **kwargs) 
                if properties is None: 
                    raise TypeError("function should return a dictionary; probably forgot 'return locals()'")
                self.__dict__.update(properties)



        Clas.__name__ = fn.__name__
        return Clas
    return decorator

def quickclass(arg):
    if isinstance(arg, type):
        return quickclass_explicit(arg)
    else: 
        return quickclass_explicit(object)(arg)



