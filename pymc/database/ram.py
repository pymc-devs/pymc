"""
RAM database module

Store the trace in memory using NumPy arrays.

Implementation Notes
--------------------
This is the only backend using preallocated memory. All others simply
append values to a stack. It might be worthwhile to use a list instead
of a NumPy array to 1. simplify this backend, 2. standardize the
`Trace model` and 3. remove the need for a truncate method.
We would need to catch MemoryError exceptions though.
"""

import pymc
from numpy import zeros, shape, concatenate, ndarray, dtype
from . import base
import warnings
import numpy as np

__all__ = ['Trace', 'Database']


class Trace(base.Trace):

    """RAM Trace

    Store the samples in memory. No data is written to disk.
    """

    def __init__(self, name, getfunc=None, db=None, value=None):
        """Create a Trace instance.

        :Parameters:
        name : string
          The trace object name. This name should uniquely identify
          the pymc variable.
        getfunc : function
          A function returning the value to tally.
        db : Database instance
          The database owning this Trace.
        value : list
          The list of trace arrays. This is used when loading the Trace from
          disk."""
        if value is None:
            self._trace = {}
            self._index = {}
        else:
            self._trace = value
            self._index = dict(zip(value.keys(), list(map(len, value.values()))))

        base.Trace.__init__(self, name=name, getfunc=getfunc, db=db)

    def _initialize(self, chain, length):
        """Create an array of zeros with shape (length, shape(obj)), where
        obj is the internal PyMC Stochastic or Deterministic.
        """
        # If this db was loaded from the disk, it may not have its
        # tallied step methods' getfuncs yet.
        if self._getfunc is None:
            self._getfunc = self.db.model._funs_to_tally[self.name]

        # First, see if the object has an explicit dtype.
        value = np.array(self._getfunc())

        if value.dtype is object:
            self._trace[chain] = zeros(length, dtype=object)

        elif value.dtype is not None:
            self._trace[chain] = zeros((length,) + shape(value), value.dtype)

        # Otherwise, if it's an array, read off its value's dtype.
        elif isinstance(value, ndarray):
            self._trace[chain] = zeros((length,) + shape(value), value.dtype)

        # Otherwise, let numpy type its value. If the value is a scalar, the trace will be of the
        # corresponding type. Otherwise it'll be an object array.
        else:
            self._trace[chain] = zeros(
                (length,
                 ) + shape(value),
                dtype=value.__class__)

        self._index[chain] = 0

    def tally(self, chain):
        """Store the object's current value to a chain.

        :Parameters:
        chain : integer
          Chain index.
        """

        value = self._getfunc()

        try:
            self._trace[chain][self._index[chain]] = value.copy()
        except AttributeError:
            self._trace[chain][self._index[chain]] = value
        self._index[chain] += 1

    def truncate(self, index, chain):
        """
        Truncate the trace array to some index.

        :Parameters:
        index : int
          The index within the chain after which all values will be removed.
        chain : int
          The chain index (>=0).
        """
        self._trace[chain] = self._trace[chain][:index]

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace.

        :Stochastics:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if slicing is None:
            slicing = slice(burn, None, thin)
        if chain is not None:
            if chain < 0:
                chain = range(self.db.chains)[chain]
            return self._trace[chain][slicing]
        else:
            return concatenate(list(self._trace.values()))[slicing]

    def __getitem__(self, index):
        chain = self._chain
        if chain is None:
            return concatenate(list(self._trace.values()))[index]
        else:
            if chain < 0:
                chain = range(self.db.chains)[chain]
            return self._trace[chain][index]

    __call__ = gettrace

    def length(self, chain=-1):
        """Return the length of the trace.

        :Parameters:
        chain : int or None
          The chain index. If None, returns the combined length of all chains.
        """
        if chain is not None:
            if chain < 0:
                chain = range(self.db.chains)[chain]
            return self._trace[chain].shape[0]
        else:
            return sum([t.shape[0] for t in self._trace.values()])


class Database(base.Database):

    """RAM database.

    Store the samples in memory. No data is written to disk.
    """

    def __init__(self, dbname):
        """Create a RAM Database instance."""
        self.__name__ = 'ram'
        self.__Trace__ = Trace
        self.dbname = dbname
        self.trace_names = []
        # A list of sequences of names of the objects to tally.
        self._traces = {}  # A dictionary of the Trace objects.
        self.chains = 0
