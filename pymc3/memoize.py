import functools
import pickle


def memoize(obj):
    """
    An expensive memoizer that works with unhashables
    """
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = (hashable(args), hashable(kwargs))

        if key not in cache:
            cache[key] = obj(*args, **kwargs)

        return cache[key]
    return memoizer


def hashable(a):
    """
    Turn some unhashable objects into hashable ones.
    """
    if isinstance(a, dict):
        return hashable(tuple((hashable(a1), hashable(a2)) for a1, a2 in a.items()))
    try:
        return hash(a)
    except TypeError:
        pass
    # Not hashable >>>
    try:
        return hash(pickle.dumps(a))
    except Exception:
        if hasattr(a, '__dict__'):
            return hashable(a.__dict__)
        else:
            return id(a)
