__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

"""Exceptions"""

import os

class ZeroProbability(ValueError):
    "Log-likelihood is invalid or negative informationnite"
    pass


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

        # _parents has to be assigned in each subclass.
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
    __name__ = 'container'
    
    def __init__(self, input):
        if hasattr(input, '__file__'):
            _filename = os.path.split(input.__file__)[-1]
            self.__name__ = os.path.splitext(_filename)[0]
        elif hasattr(input, '__name__'):
            self.__name__ = input.__name__
        else:
            try:
                self.__name__ = input['__name__']
            except: 
                self.__name__ = 'container'
    
    def _get_logp(self):
        return sum(obj.logp for obj in self.parameters | self.potentials | self.data)
    logp = property(_get_logp)
        
class ParameterBase(Variable):
    pass
    
class NodeBase(Variable):
    pass
    
class PotentialBase(PyMCBase):
    pass
    
class SamplingMethodBase(object):
    pass    