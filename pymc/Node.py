"""
Base classes are defined here.
"""

__docformat__ = 'reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

import os
import sys
import pdb
import numpy as np
import types
from . import six
print_ = six.print_

try:
    from types import UnboundMethodType
except ImportError:
    # On Python 3, unbound methods are just functions.
    def UnboundMethodType(func, inst, cls):
        return func


def logp_of_set(s):
    exc = None
    logp = 0.
    for obj in s:
        try:
            logp += obj.logp
        except ZeroProbability:
            raise
        except:
            if exc is None:
                exc = sys.exc_info()
    if exc is None:
        return logp
    else:
        six.reraise(*exc)


def logp_gradient_of_set(variable_set, calculation_set=None):
    """
    Calculates the gradient of the joint log posterior with respect to all the variables in variable_set.
    Calculation of the log posterior is restricted to the variables in calculation_set.

    Returns a dictionary of the gradients.
    """

    logp_gradients = {}
    for variable in variable_set:
        logp_gradients[variable] = logp_gradient(variable, calculation_set)

    return logp_gradients


def logp_gradient(variable, calculation_set=None):
    """
    Calculates the gradient of the joint log posterior with respect to variable.
    Calculation of the log posterior is restricted to the variables in calculation_set.
    """
    return variable.logp_partial_gradient(variable, calculation_set) + sum(
        [child.logp_partial_gradient(variable, calculation_set) for child in variable.children])


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

    def __init__(self, doc, name, parents, cache_depth, verbose=-1):

        # Name and docstrings
        self.__doc__ = doc
        if not isinstance(name, str):
            raise ValueError(
                'The name argument must be a string, but received %s.' %
                name)
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

        # Specify new parents
        self._parents = self.ParentDict(
            regular_dict=new_parents,
            owner=self)

        # Add self as child of parents
        self._parents.attach_parents()

        # Get new lazy function
        self.gen_lazy_function()

    parents = property(
        _get_parents,
        _set_parents,
        doc="Self's parents: the variables referred to in self's declaration.")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return object.__repr__(self).replace(
            ' object ', " '%s' " % self.__name__)

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

    __array_priority__ = 10

    def __init__(self, doc, name, parents, cache_depth,
                 trace=False, dtype=None, plot=None, verbose=-1):

        self.dtype = dtype
        self.keep_trace = trace
        self._plot = plot
        self.children = set()
        self.extended_children = set()

        Node.__init__(self, doc, name, parents, cache_depth, verbose=verbose)

        if self.dtype is None:
            if hasattr(self.value, 'dtype'):
                self.dtype = self.value.dtype
            else:
                self.dtype = np.dtype(type(self.value))

    def __str__(self):
        return self.__name__

    def _get_plot(self):
        # Get plotting flag
        return self._plot

    def _set_plot(self, true_or_false):
        # Set plotting flag
        self._plot = true_or_false

    plot = property(
        _get_plot,
        _set_plot,
        doc='A flag indicating whether self should be plotted.')

    def stats(self, alpha=0.05, start=0, batches=100,
              chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
        """
        Generate posterior statistics for node.

        :Parameters:
        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.

        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.

        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.

        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).

        quantiles : tuple or list
          The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5).
        """
        return self.trace.stats(alpha=alpha, start=start, batches=batches,
                                chain=chain, quantiles=quantiles)


    def summary(self, alpha=0.05, start=0, batches=100, chain=None, roundto=3):
        """
        Generate a pretty-printed summary of the node.

        :Parameters:
        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.

        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.

        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.

        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).

        roundto : int
          The number of digits to round posterior statistics.
        """

        # Calculate statistics for Node
        statdict = self.stats(
            alpha=alpha,
            start=start,
            batches=batches,
            chain=chain)
        
        size = np.size(statdict['mean'])

        print_('\n%s:' % self.__name__)
        print_(' ')

        # Initialize buffer
        buffer = []

        # Index to interval label
        iindex = [key.split()[-1] for key in statdict.keys()].index('interval')
        interval = list(statdict.keys())[iindex]

        # Print basic stats
        buffer += [
            'Mean             SD               MC Error        %s' %
            interval]
        buffer += ['-' * len(buffer[-1])]

        indices = range(size)
        if len(indices) == 1:
            indices = [None]
        
        _format_str = lambda x, i=None, roundto=2: str(np.round(x.ravel()[i].squeeze(), roundto))
            
        for index in indices:
            # Extract statistics and convert to string
            m = _format_str(statdict['mean'], index, roundto)
            sd = _format_str(statdict['standard deviation'], index, roundto)
            mce = _format_str(statdict['mc error'], index, roundto)
            hpd = str(statdict[interval].reshape(
                    (2, size))[:,index].squeeze().round(roundto))

            # Build up string buffer of values
            valstr = m
            valstr += ' ' * (17 - len(m)) + sd
            valstr += ' ' * (17 - len(sd)) + mce
            valstr += ' ' * (len(buffer[-1]) - len(valstr) - len(hpd)) + hpd

            buffer += [valstr]

        buffer += [''] * 2

        # Print quantiles
        buffer += ['Posterior quantiles:', '']

        buffer += [
            '2.5             25              50              75             97.5']
        buffer += [
            ' |---------------|===============|===============|---------------|']

        for index in indices:
            quantile_str = ''
            for i, q in enumerate((2.5, 25, 50, 75, 97.5)):
                qstr = _format_str(statdict['quantiles'][q], index, roundto)
                quantile_str += qstr + ' ' * (17 - i - len(qstr))
            buffer += [quantile_str.strip()]

        buffer += ['']

        print_('\t' + '\n\t'.join(buffer))


ContainerRegistry = []


class ContainerMeta(type):

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)

        def change_method(self, *args, **kwargs):
            raise NotImplementedError(name + ' instances cannot be changed.')

        if cls.register:
            ContainerRegistry.append((cls, cls.containing_classes))

            for meth in cls.change_methods:
                setattr(cls, meth, UnboundMethodType(change_method, None, cls))
        cls.register = False


class ContainerBase(object):

    """
    Abstract base class.

    :SeeAlso:
      ListContainer, SetContainer, DictContainer, TupleContainer, ArrayContainer
    """
    register = False
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
        return logp_of_set(
            self.stochastics | self.potentials | self.observed_stochastics)

    # Define log-probability property
    logp = property(
        _get_logp,
        doc='The summed log-probability of all stochastic variables (data\nor otherwise) and factor potentials in self.')

ContainerBase = six.with_metaclass(ContainerMeta, ContainerBase)

StochasticRegistry = []


class StochasticMeta(type):

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        StochasticRegistry.append(cls)


class StochasticBase(six.with_metaclass(StochasticMeta, Variable)):

    """
    Abstract base class.

    :SeeAlso:
      Stochastic, Variable
    """

DeterministicRegistry = []


class DeterministicMeta(type):

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        DeterministicRegistry.append(cls)


class DeterministicBase(six.with_metaclass(DeterministicMeta, Variable)):

    """
    Abstract base class.

    :SeeAlso:
      Deterministic, Variable
    """

PotentialRegistry = []


class PotentialMeta(type):

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        PotentialRegistry.append(cls)


class PotentialBase(six.with_metaclass(PotentialMeta, Node)):

    """
    Abstract base class.

    :SeeAlso:
      Potential, Variable
    """
