__docformat__='reStructuredText'

"""Exceptions"""

class ZeroProbability(ValueError):
    "Log-likelihood is invalid or negative informationnite"
    pass

class ParentDict(dict):
    """
    A special subclass of dict which makes it safe to change parameters'
    and nodes' parents. When __setitem__ is called, a ParentDict instance
    removes its owner from the old parent's children set (if appropriate)
    and adds its owner to the new parent's children set. It then asks
    its owner to generate a new LazyFunction instance using its new
    parents.

    XXX

    SamplingMethod and Model are expecting parameters' and nodes'
    children to be static. We should figure out what to do about this.
    """
    def __init__(self, regular_dict, owner):
        self.update(regular_dict)
        self.owner = owner

    def __setitem__(self, key, new_parent):

        old_parent = dict.__getitem__(self, key)

        # Possibly remove me from old parent's children set.
        if isinstance(old_parent, PyMCBase):
            only_reference = True

            # See if I only claim the old parent via this key.
            for item in self.iteritems():
                if item[0] is old_parent and not item[1] == key:
                    only_reference = False
                    break

            # If so, remove me from the old parent's children set.
            if only_reference:
                old_parent.children.remove(self.owner)

        # If the new parent is a PyMC object, add me to its children set.
        if isinstance(new_parent, PyMCBase):
            new_parent.children.add(self.owner)

        dict.__setitem__(self, key, new_parent)

        # Tell my owner it needs a new lazy function.
        self.owner.gen_lazy_function()


class PyMCBase(object):
    """
    The base class from which Parameter and Node inherit.
    Shouldn't need to be instantiated directly.
    
    See source code in PyMCBase.py if you want to subclass it.
    
    :SeeAlso: Parameter, Node
    """
    def __init__(self, doc, name, parents, cache_depth, trace):

        self._parents = ParentDict(regular_dict = parents, owner = self)
        self.children = set()
        self.__doc__ = doc
        # self._value = None
        self.__name__ = name
        self.trace = trace

        self._cache_depth = cache_depth
        
        for object in self._parents.itervalues():
            if isinstance(object, PyMCBase):
                object.children.add(self)
        
        self.gen_lazy_function()
                
    def _get_parents(self):
        return self._parents
        
    def _set_parents(self, new_parents):
        for parent in self._parents.itervalues():
            if isinstance(parent, PyMCBase):
                parent.children.discard(self)
        self._parents = ParentDict(regular_dict = new_parents, owner = self)
        self.gen_lazy_function()
        
    def _del_parents(self):
        for parent in self._parents.itervalues():
            if isinstance(parent, PyMCBase):
                parent.children.discard(self)
        del self._parents
        
    parents = property(fget=_get_parents, fset=_set_parents, fdel=_del_parents)
    
    # def __str__(self):
    #     return self.value.__str__()
                
    def gen_lazy_function(self):
        pass                


class ContainerBase(object):
    """
    The abstract base class from which Container inherits.
    
    :SeeAlso: Container
    """
    pass