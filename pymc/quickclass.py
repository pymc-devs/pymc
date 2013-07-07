import functools
import inspect

__all__ = ['wraps_with_spec', 'quickclass', 'withdefaults']

def quickclass(baseclass):
    def decorator(f):

        def __init__(self, *args, **kwargs):
            self.__dict__.update(f(*args, **kwargs))

        clsdict = {
            '__init__': __init__,
            '__doc__': f.__doc__}

        cls = type(f.__name__, (baseclass,), clsdict)
        cls._spec = getargspec_wrapped(f)
        return cls 
    
    return decorator


def remove_keys(d, ks):
    return dict((k, v) for k, v in d.iteritems() if k not in ks)


def allowed_args(f, args, kwargs):
    spec = getargspec_wrapped(f)
    varnames = set(spec.args)
    narg = len(spec.args)

    if spec.varargs:
        args_a = args
    else:
        args_a = args[:narg]

    if spec.keywords:
        kwargs_a = kwargs
    else:
        unused = set(kwargs.keys()) - varnames
        kwargs_a = remove_keys(kwargs, unused)

    return args_a, kwargs_a, args[narg:], remove_keys(kwargs, varnames)

"""
These custom definitions of wrap and getargspec allow access to the
getargspec of the original function
"""

def wraps_with_spec(wrapped):
    def decorator(f):
        g = functools.wraps(wrapped)(f)
        g._spec = getargspec_wrapped(wrapped)
        return g
    return decorator

def getargspec_wrapped(f):
    if hasattr(f, '_spec'):
        return f._spec
    else:
        return inspect.getargspec(f)

def check_dict(d):
    if not isinstance(d, dict):
        raise TypeError(
        "function should return a dict perhaps forgot 'return locals()'")


def withdefaults(g):
    def decorator(f):
        @wraps_with_spec(f)
        def newf(*args, **kwargs):
            args_a, kwargs_a, args_left, kwargs_left = allowed_args(f, args,kwargs)
            r = g(*args_left, **kwargs_left)
            check_dict(r)
            r1 = f(*args_a, **kwargs_a)
            check_dict(r1)
            r.update(r1)
            return r

        return newf
    return decorator
