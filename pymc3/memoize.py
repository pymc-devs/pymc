#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import collections
import functools

import dill

from pymc3.util import biwrap

CACHE_REGISTRY = []


@biwrap
def memoize(obj, bound=False):
    """
    Decorator to apply memoization to expensive functions.
    It uses a custom `hashable` helper function to hash typically unhashable Python objects.

    Parameters
    ----------
    obj : callable
        the function to apply the caching to
    bound : bool
        indicates if the [obj] is a bound method (self as first argument)
        For bound methods, the cache is kept in a `_cache` attribute on [self].
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
            if not hasattr(args[0], "_cache"):
                setattr(args[0], "_cache", collections.defaultdict(dict))
                # do not add to cache registry
            cache = getattr(args[0], "_cache")[obj.__name__]
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
            for v in getattr(obj, "_cache", {}).values():
                v.clear()
        else:
            obj.cache.clear()


class WithMemoization:
    def __hash__(self):
        return hash(id(self))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def hashable(a) -> int:
    """
    Hashes many kinds of objects, including some that are unhashable through the builtin `hash` function.
    Lists and tuples are hashed based on their elements.
    """
    if isinstance(a, dict):
        # first hash the keys and values with hashable
        # then hash the tuple of int-tuples with the builtin
        return hash(tuple((hashable(k), hashable(v)) for k, v in a.items()))
    if isinstance(a, (tuple, list)):
        # lists are mutable and not hashable by default
        # for memoization, we need the hash to depend on the items
        return hash(tuple(hashable(i) for i in a))
    try:
        return hash(a)
    except TypeError:
        pass
    # Not hashable >>>
    try:
        return hash(dill.dumps(a))
    except Exception:
        if hasattr(a, "__dict__"):
            return hashable(a.__dict__)
        else:
            return id(a)
