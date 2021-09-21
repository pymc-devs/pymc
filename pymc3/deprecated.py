import functools
import inspect
import warnings

def deprecated(reason=None, version=None):
    """
    PyMC deprecation helper. This decorator emits deprecated warnings.
    """
    cause = reason
    
    if cause is not None and version is not None:
        reason = ", since version " + version + " " + "("+cause+")"
        
    if cause is not None and version is None:
        reason = "("+cause+")"
        
    if cause is None and version is not None :
        reason = ", since version " + version

    def decorator(func):
        if inspect.isclass(func):
            fmt = "class {name} is deprecated{reason}."
        else:
            fmt = "function {name} is deprecated{reason}."

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt.format(name=func.__name__, reason=reason),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)

        return new_func

    return decorator

class deprecated_param:
    def __init__(self, deprecated_args, version, reason):
        self.deprecated_args = set(deprecated_args.split())
        self.version = version
        self.reason = reason

    def __call__(self, callable):
        def wrapper(*args, **kwargs):
            found = self.deprecated_args.intersection(kwargs)
            if found:
                raise Warning("Parameter(s) %s deprecated since version %s; %s" % (
                    ', '.join(map("'{}'".format, found)), self.version, self.reason))
            return callable(*args, **kwargs)
        return wrapper
