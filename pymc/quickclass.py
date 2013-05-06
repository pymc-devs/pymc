from functools import wraps


def quickclass(baseclass):
    def decorator(fn):

        def __init__(self, *args, **kwargs):
            self.__dict__.update(fn(*args, **kwargs))

        clsdict = {
            '__init__': __init__,
            '__doc__': fn.__doc__}

        return type(fn.__name__, (baseclass,), clsdict)
    return decorator


def argnames(f):
    return set(f.func_code.co_varnames)


def kfilter(ks, d):
    return dict((k, v) for k, v in d.iteritems() if k in ks)

dicterr = TypeError(
    "function should return a dict perhaps forgot 'return locals()'")


def withdefaults(d):
    def decorator(f):
        @wraps(f)
        def fn(*args, **kwargs):
            dvar = argnames(d)
            fvar = argnames(f)

            if set(kwargs) - dvar - fvar:
                raise ValueError("not all arguments used")

            narg = f.func_code.co_argcount
            largs, rargs = args[:narg], args[narg:]
            u = f(*largs, **kfilter(fvar, kwargs))
            if not u:
                raise dicterr

            r = d(*rargs, **kfilter(dvar, kwargs))
            if not r:
                raise dicterr
            r.update(u)
            return r
        return fn
    return decorator
