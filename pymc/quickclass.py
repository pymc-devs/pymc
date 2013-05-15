from functools import wraps
from inspect import getargspec


def quickclass(baseclass):
    def decorator(fn):

        def __init__(self, *args, **kwargs):
            self.__dict__.update(fn(*args, **kwargs))

        clsdict = {
            '__init__': __init__,
            '__doc__': fn.__doc__}

        return type(fn.__name__, (baseclass,), clsdict)
    return decorator



def ksplit(ks, d):
    in_ks = dict((k, v) for k, v in d.iteritems() if k in ks)
    rest = dict((k, v) for k, v in d.iteritems() if k not in ks)
    return in_ks, rest




dicterr = TypeError(
    "function should return a dict perhaps forgot 'return locals()'")


def withdefaults(d):
    def decorator(f):
        @wraps(f)
        def fn(*args, **kwargs):
            dspec = getargspec(d)
            fspec = getargspec(f)

            dvar = set(dspec.args)
            fvar = set(fspec.args)

            narg = len(fspec.args)
            largs, rargs = args[:narg], args[narg:]

            dkwargs,fkwargs = ksplit(dvar, kwargs)

            u = f(*largs, **fkwargs)
            if not u:
                raise dicterr

            r = d(*rargs, **dkwargs)
            if not r:
                raise dicterr
            r.update(u)
            return r
        return fn
    return decorator
