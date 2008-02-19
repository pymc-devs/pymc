"""
Base classes are defined here.
"""

__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

import os

class ZeroProbability(ValueError):
    "Log-likelihood is invalid or negative informationnite"
    pass


class Node(object):
    """
    The base class from which Stochastic, Deterministic and Potential inherit.
    Shouldn't need to be instantiated directly.
    
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
          
          - plot (optional) : boolean
            A flag indicating whether this variable is to be plotted.
          
          - verbose (optional) : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
    
    :SeeAlso: Stochastic, Deterministic
    """
    def __init__(self, doc, name, parents, cache_depth, verbose=0):
        
        # Name and docstrings
        self.__doc__ = doc
        self.__name__ = name
        
        # Level of feedback verbosity
        self.verbose = verbose
        
        # Number of memorized values
        self._cache_depth = cache_depth
        
        # Initialize sets of children and parents
        self.children = set()
        self.parents = parents
        
        # New lazy function
        self.gen_lazy_function()
    
    def _get_parents(self):
        # Get parents of this object
        return self._parents
    
    def _set_parents(self, new_parents):
        # Define parents of this object
        
        # Remove from current parents
        if hasattr(self,'_parents'):
            # Iterate over items in ParentDict
            for parent in self._parents.itervalues():
                if isinstance(parent, Variable):
                    parent.children.discard(self)
                elif isinstance(parent, ContainerBase):
                    for variable in parent.variables:
                        variable.chidren.discard(self)
        
        # Specify new parents
        self._parents = self.ParentDict(regular_dict = new_parents, owner = self)
        
        # Add self as child of parents
        for parent in self._parents.itervalues():
            if isinstance(parent, Variable):
                parent.children.add(self)
            elif isinstance(parent, ContainerBase):
                for variable in parent.variables:
                    variable.children.add(self)
        
        
        # Get new lazy function
        self.gen_lazy_function()
    
    parents = property(_get_parents, _set_parents, doc="Self's parents: the variables referred to in self's declaration.")
    
    def _get_plot(self):
        # Get plotting flag
        return self._plot
    
    plot = property(_get_plot, doc='A flag indicating whether self should be plotted.')
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return object.__repr__(self).replace('object', "'%s'"%self.__name__)
    
    def gen_lazy_function(self):
        pass
    
    def stats(self, alpha=0.05):
        """
        Generate posterior statistics for node.
        """
        from utils import hpd, quantiles
        from numpy import sqrt
        
        trace = self.trace()
        
        return {
            'n': len(trace),
            'standard deviation': trace.std(0),
            'mean': trace.mean(0),
            '%s%s HPD interval' % (int(100*(1-alpha)),'%'): hpd(trace, alpha),
            'mc error': trace.std(0) / sqrt(len(trace)),
            'quantiles': quantiles(trace)
        }

class Variable(Node):
    """
    The base class for Stochastics and Deterministics.
    """
    def __init__(self, doc, name, parents, cache_depth, trace=False, dtype=None, plot=True, verbose=0):
        # print doc, name, parents, cache_depth, trace, dtype, plot, verbose
        self.dtype=dtype        
        self.trace=trace
        self._plot=plot
        Node.__init__(self, doc, name, parents, cache_depth, verbose=verbose)

    def __str__(self):
        return str(self.value)
        
class ContainerBase(object):
    """
    The base class from which containers inherit.
    """
    __name__ = 'container'
    
    def __init__(self, input):
        # ContainerBase class initialization
        
        # Look for name attributes
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
        # Return total log-probabilities from all elements
        return sum(obj.logp for obj in self.stochastics | self.potentials | self.data_stochastics)
    
    # Define log-probability property
    logp = property(_get_logp, doc='The summed log-probability of all stochastic variables (data\nor otherwise) and factor potentials in self.')

class StochasticBase(Variable):
    """
    Abstract base class.
    """
    pass

class DeterministicBase(Variable):
    """
    Abstract base class.
    """
    pass

class PotentialBase(Node):
    """
    Abstract base class.
    """
    pass