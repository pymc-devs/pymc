"""
Base classes are defined here.
"""

__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

import os, pdb
import numpy as np
import types

def batchsd(trace, batches=5):
    """
    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.
    """
    
    if len(shape(trace)) > 1:
        
        return np.array([batchsd(t, batches) for t in trace])
        
    else:
        if batches == 1: return np.std(trace)/np.sqrt(len(trace))
    
        try:
            batched_traces = np.resize(trace, (batches, len(trace)/batches))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(trace) % batches
            batched_traces = np.resize(trace[:-resid], (batches, len(trace)/batches))
        
        means = np.mean(batched_traces, 1)
    
        return np.std(means)/np.sqrt(batches)

class ZeroProbability(ValueError):
    "Log-probability is undefined or negative infinity"
    pass


class Node(object):
    """
    The base class for Stochastic, Deterministic and Potential.

    :Parameters:
    doc : string
      The docstring for this node.

    name : string
      The name of this node.

    parents : dictionary
      A dictionary containing the parents of this node.

    cache_depth : integer
      An integer indicating how many of this node's
      value computations should be 'memorized'.

    verbose (optional) : integer
      Level of output verbosity: 0=none, 1=low, 2=medium, 3=high


    .. seealso::

       :class:`Stochastic`
         The class defining *random* variables, or unknown parameters.

       :class:`Deterministic`
         The class defining deterministic values, ie the result of a function.

       :class:`Potential`
         An arbitrary log-probability term to multiply into the joint
         distribution.

       :class:`Variable`
         The base class for :class:`Stochastics` and :class:`Deterministics`.

    """
    def __init__(self, doc, name, parents, cache_depth, verbose=0):

        # Name and docstrings
        self.__doc__ = doc
        self.__name__ = name

        # Level of feedback verbosity
        self.verbose = verbose

        # Number of memorized values
        self._cache_depth = cache_depth

        # Initialize
        self.parents = parents

    def _get_parents(self):
        # Get parents of this object
        return self._parents

    def _set_parents(self, new_parents):
        # Define parents of this object

        # Remove from current parents
        if hasattr(self,'_parents'):
            self._parents.detach_children()

        # Specify new parents
        self._parents = self.ParentDict(regular_dict = new_parents, owner = self)

        # Add self as child of parents
        self._parents.attach_parents()

        # Get new lazy function
        self.gen_lazy_function()

    parents = property(_get_parents, _set_parents, doc="Self's parents: the variables referred to in self's declaration.")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return object.__repr__(self).replace(' object ', " '%s' "%self.__name__)

    def gen_lazy_function(self):
        pass


class Variable(Node):
    """
    The base class for Stochastics and Deterministics.

    :Parameters:
    doc : string
      The docstring for this node.

    name : string
      The name of this node.

    parents : dictionary
      A dictionary containing the parents of this node.

    cache_depth : integer
      An integer indicating how many of this node's
      value computations should be 'memorized'.

    trace : boolean
      Indicates whether a trace should be kept for this variable
      if its model is fit using a Monte Carlo method.

    plot : boolean
      Indicates whether summary plots should be prepared for this
      variable if summary plots of its model are requested.

    dtype : numpy dtype
      If the value of this variable's numpy dtype can be known in
      advance, it is advantageous to specify it here.

    verbose (optional) : integer
      Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    :SeeAlso:
      Stochastic, Deterministic, Potential, Node
    """
    def __init__(self, doc, name, parents, cache_depth, trace=False, dtype=None, plot=None, verbose=0):

        self.dtype=dtype
        self.trace=trace
        self._plot=plot
        self.children = set()
        self.extended_children = set()

        Node.__init__(self, doc, name, parents, cache_depth, verbose=verbose)

        if self.dtype is None:
            if hasattr(self._value, 'dtype'):
                self.dtype = self._value.dtype
            else:
                self.dtype = np.dtype(type(self._value))

    def __str__(self):
        return self.__name__

    def _get_plot(self):
        # Get plotting flag
        return self._plot

    def _set_plot(self, true_or_false):
        # Set plotting flag
        self._plot = true_or_false

    plot = property(_get_plot, _set_plot, doc='A flag indicating whether self should be plotted.')

    def stats(self, alpha=0.05, start=0, batches=100):
        """
        Generate posterior statistics for node.
        """
        from utils import hpd, quantiles
        from numpy import sqrt

        try:
            trace = np.array(self.trace(), float)[start:]

            n = len(trace)
            if not n:
                print 'Cannot generate statistics for zero-length trace in', self.__name__
                return


            return {
                'n': n,
                'standard deviation': trace.std(0),
                'mean': trace.mean(0),
                '%s%s HPD interval' % (int(100*(1-alpha)),'%'): hpd(trace, alpha),
                'mc error': batchsd(trace, batches),
                'quantiles': quantiles(trace)
            }
        except:
            print 'Could not generate output statistics for', self.__name__

ContainerRegistry = []

class ContainerMeta(type):
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)

        def change_method(self, *args, **kwargs):
            raise NotImplementedError, name + ' instances cannot be changed.'

        if cls.register:
            ContainerRegistry.append((cls, cls.containing_classes))

            for meth in cls.change_methods:
                setattr(cls, meth, types.UnboundMethodType(change_method, None, cls))
        cls.register=False


class ContainerBase(object):
    """
    Abstract base class.

    :SeeAlso:
      ListContainer, SetContainer, DictContainer, TupleContainer, ArrayContainer
    """
    register = False
    __metaclass__ = ContainerMeta
    change_methods = []
    containing_classes = []

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

    def assimilate(self, new_container):
        self.containers.append(new_container)
        self.variables.update(new_container.variables)
        self.stochastics.update(new_container.stochastics)
        self.potentials.update(new_container.potentials)
        self.deterministics.update(new_container.deterministics)
        self.observed_stochastics.update(new_container.observed_stochastics)

    def _get_logp(self):
        # Return total log-probabilities from all elements
        return sum(obj.logp for obj in self.stochastics | self.potentials | self.observed_stochastics)

    # Define log-probability property
    logp = property(_get_logp, doc='The summed log-probability of all stochastic variables (data\nor otherwise) and factor potentials in self.')

StochasticRegistry = []
class StochasticMeta(type):
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        StochasticRegistry.append(cls)
class StochasticBase(Variable):
    """
    Abstract base class.

    :SeeAlso:
      Stochastic, Variable
    """
    __metaclass__ = StochasticMeta

DeterministicRegistry = []
class DeterministicMeta(type):
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        DeterministicRegistry.append(cls)
class DeterministicBase(Variable):
    """
    Abstract base class.

    :SeeAlso:
      Deterministic, Variable
    """
    __metaclass__ = DeterministicMeta

PotentialRegistry = []
class PotentialMeta(type):
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        PotentialRegistry.append(cls)
class PotentialBase(Node):
    """
    Abstract base class.

    :SeeAlso:
      Potential, Variable
    """
    __metaclass__ = PotentialMeta
