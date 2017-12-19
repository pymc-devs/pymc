import functools
import pickle

CACHE_REGISTRY = []


def memoize(obj):
    """
    An expensive memoizer that works with unhashables
    """
    cache = obj.cache = {}
    CACHE_REGISTRY.append(cache)

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        # remember first argument as well, used to clear cache for particular instance
        key = (hashable(args[:1]), hashable(args), hashable(kwargs))

        if key not in cache:
            cache[key] = obj(*args, **kwargs)

        return cache[key]
    return memoizer


def clear_cache():
    for c in CACHE_REGISTRY:
        c.clear()


class WithMemoization(object):
    def __hash__(self):
        return hash(id(self))

    def __del__(self):
        # regular property call with args (self, )
        key = hash((self, ))
        to_del = []
        for c in CACHE_REGISTRY:
            for k in c.keys():
                if k[0] == key:
                    to_del.append((c, k))
        for (c, k) in to_del:
            del c[k]


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
