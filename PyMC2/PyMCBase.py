#TODO:
# Need ContainerBase again to break cycle in inheritance tree.
# Need ultimate_parents and ultimate_parent_values arrays, currently Container contents won't be checked.
# It'll be convenient to keep non-PyMC-Object parents constant again (no cache checking), no big deal.
# This is moot now, but don't copy the argument dictionary into the cache, make a new dictionary every time.


class PyMCBase(object):
    def __init__(self, doc, name, parents, cache_depth, trace):

        self.parents = parents
        self.children = set()
        self.__doc__ = doc
        self.__name__ = name
        self._value = None
        self.trace = trace

        self._cache_depth = cache_depth
        
        for object in self.parents.itervalues():
            if isinstance(object, PyMCBase):
                object.children.add(self)


class ContainerBase(object):
    pass