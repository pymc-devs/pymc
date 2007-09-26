__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

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
    
    :Parameters:
          -doc : string
              The docstring for this node.

          -name : string
              The name of this node.

          -parents : dictionary
              A dictionary containing the parents of this node.

          -trace : boolean
              A boolean indicating whether this node's value 
              should be traced (in MCMC).

          -cache_depth : integer   
              An integer indicating how many of this node's
              value computations should be 'memorized'.
              
          - verbose (optional) : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
    
    :SeeAlso: Parameter, Node
    """
    def __init__(self, doc, name, parents, cache_depth, trace, verbose=0):

        # Initialize parents and children
        self._parents = ParentDict(regular_dict = parents, owner = self)
        self.children = set()
        
        # Name and docstrings
        self.__doc__ = doc
        self.__name__ = name
        
        # Adopt passed trace
        self.trace = trace
        
        # Flag for feedback verbosity
        self.verbose = verbose

        self._cache_depth = cache_depth
        
        # Add self as child of parents
        for object in self._parents.itervalues():
            if isinstance(object, PyMCBase):
                object.children.add(self)
        
        # New lazy function
        self.gen_lazy_function()
                
    def _get_parents(self):
        # Get parents of this object
        return self._parents
        
    def _set_parents(self, new_parents):
        # Define parents of this object
        
        # Iterate over items in ParentDict
        for parent in self._parents.itervalues():
            if isinstance(parent, PyMCBase):
                # Remove self as child
                parent.children.discard(self)
                
        # Specify new parents
        self._parents = ParentDict(regular_dict = new_parents, owner = self)
        
        # Get new lazy function
        self.gen_lazy_function()
        
    def _del_parents(self):
        # Deletes all parents of current object
        
        # Remove as child from parents
        for parent in self._parents.itervalues():
            if isinstance(parent, PyMCBase):
                parent.children.discard(self)
                
        # Delete parents
        del self._parents
        
    parents = property(fget=_get_parents, fset=_set_parents, fdel=_del_parents)
    
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return object.__repr__(self).replace('object', self.__name__)
                
    def gen_lazy_function(self):
        pass                

class Variable(PyMCBase):
    """
    The base class for Parameters and Nodes;
    represents variables that actually participate in the probability model.
    """
    pass

class ContainerBase(object):

    """
    The abstract base class from which containers inherit.
    
    :SeeAlso: Container, ArrayContainer, ListDictContainer, SetContainer
    """
    def __init__(self):
        self.all_objects = set()
        self.variables = set()
        self.nodes = set()
        self.parameters = set()
        self.potentials = set()
        self.data = set()
        self.sampling_methods = set()
    