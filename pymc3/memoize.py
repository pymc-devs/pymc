import functools
import pickle
import collections
from .util import biwrap
CACHE_REGISTRY = []


@biwrap
def memoize(obj, bound=False):
    """
    An expensive memoizer that works with unhashables
    """
    # this is declared not to be a bound method, so just attach new attr to obj
    if not bound:
        obj.cache = {}
        CACHE_REGISTRY.append(obj.cache)

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if not bound:
            key = (hashable(args), hashable(kwargs))
            cache = obj.cache
        else:
            # bound methods have self as first argument, remove it to compute key
            key = (hashable(args[1:]), hashable(kwargs))
            if not hasattr(args[0], '_cache'):
                setattr(args[0], '_cache', collections.defaultdict(dict))
                # do not add to cache regestry
            cache = getattr(args[0], '_cache')[obj.__name__]
        if key not in cache:
            cache[key] = obj(*args, **kwargs)

        return cache[key]
    return memoizer


def clear_cache(obj=None):
    if obj is None:
        for c in CACHE_REGISTRY:
            c.clear()
    else:
        if isinstance(obj, WithMemoization):
            for v in getattr(obj, '_cache', {}).values():
                v.clear()
        else:
            obj.cache.clear()


class WithMemoization:
    def __hash__(self):
        return hash(id(self))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_cache', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


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
