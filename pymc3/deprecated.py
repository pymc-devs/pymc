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