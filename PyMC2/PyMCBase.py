class PyMCBase(object):
    """
    The base class from which Parameter and Node inherit.
    Shouldn't need to be instantiated directly.
    
    See source code in PyMCBase.py if you want to subclass it.
    
    :SeeAlso: Parameter, Node
    """
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
    """
    The abstract base class from which Container inherits.
    
    :SeeAlso: Container
    """
    pass