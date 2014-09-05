"""HDF5 database module.

Store the traces in an HDF5 array using pytables.


Implementation Notes
--------------------

This version only supports numeric objects, and stores them in
extentable HDF5 arrays.  This allows the implementation to handle very
large data vectors.


Additional Dependencies
-----------------------
 * HDF5 version 1.6.5, required by pytables.
 * pytables version 2 and up.  <http://sourceforge.net/projects/pytables/>

"""

import os
import sys
import traceback
import warnings

import numpy as np
import pymc
import tables

from pymc.database import base, pickle
from pymc import six

__all__ = ['Trace', 'Database', 'load']

warn_tally = """
Error tallying %s, will not try to tally it again this chain.
Did you make all the same variables and step methods tallyable
as were tallyable last time you used the database file?

Error:

%s"""


#

class Trace(base.Trace):

    """HDF5 trace."""

    def tally(self, chain):
        """Adds current value to trace."""

        arr = np.asarray(self._getfunc())
        arr = arr.reshape((1,) + arr.shape)
        self.db._arrays[chain, self.name].append(arr)

    # def __getitem__(self, index):
    #     """Mimic NumPy indexing for arrays."""
    #     chain = self._chain
    #     if chain is not None:
    #         tables = [self.db._gettables()[chain],]
    #     else:
    #         tables = self.db._gettables()
    #     out = []
    #     for table in tables:
    #         out.append(table.col(self.name))
    #     if np.isscalar(chain):
    #         return out[0][index]
    #     else:
    #         return np.hstack(out)[index]
    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).

        :Parameters:
        burn : integer
          The number of transient steps to skip.
        thin : integer
          Keep one in thin.
        chain : integer
          The index of the chain to fetch. If None, return all chains. The
          default is to return the last chain.
        slicing : slice object
          A slice overriding burn and thin assignement.
        """

        # XXX: handle chain == None case properly

        if chain is None:
            chain = -1
        chain = self.db.chains[chain]

        arr = self.db._arrays[chain, self.name]

        if slicing is not None:
            burn, stop, thin = slicing.start, slicing.stop, slicing.step

        if slicing is None or stop is None:
            stop = arr.nrows
            return np.asarray(arr.read(start=burn, stop=stop, step=thin))

    __call__ = gettrace

    # def length(self, chain=-1):
    #     """Return the length of the trace.

    #     :Parameters:
    #     chain : int or None
    #       The chain index. If None, returns the combined length of all chains.
    #     """
    #     if chain is not None:
    #         tables = [self.db._gettables()[chain],]
    #     else:
    #         tables = self.db._gettables()

    #     n = np.asarray([table.nrows for table in tables])
    #     return n.sum()


#

class Database(pickle.Database):

    """HDF5 database.

    Create an HDF5 file <model>.h5.  Each chain is stored in a group,
    and the stochastics and deterministics are stored as extendable
    arrays in each group.
    """

    def __init__(self, dbname, dbmode='a',
                 dbcomplevel=0, dbcomplib='zlib',
                 **kwds):
        """Create an HDF5 database instance, where samples are stored
        in extendable arrays.

        :Parameters:
        dbname : string
          Name of the hdf5 file.
        dbmode : {'a', 'w', 'r'}
          File mode: 'a': append, 'w': overwrite, 'r': read-only.
        dbcomplevel : integer (0-9)
          Compression level, 0: no compression.
        dbcomplib : string
          Compression library (zlib, bzip2, lzo)

        :Notes:
          * zlib has a good compression ratio, although somewhat slow, and
            reasonably fast decompression.
          * lzo is a fast compression library offering however a low compression
            ratio.
          * bzip2 has an excellent compression ratio but requires more CPU.
        """

        self.__name__ = 'hdf5ea'
        self.__Trace__ = Trace

        self.dbname = dbname
        self.mode = dbmode

        db_exists = os.path.exists(self.dbname)
        self._h5file = tables.open_file(self.dbname, self.mode)

        default_filter = tables.Filters(
            complevel=dbcomplevel,
            complib=dbcomplib)
        if self.mode == 'r' or (self.mode == 'a' and db_exists):
            self.filter = getattr(self._h5file, 'filters', default_filter)
        else:
            self.filter = default_filter

        self.trace_names = []
        self._traces = {}
        # self._states = {}
        self._chains = {}
        self._arrays = {}

        # load existing data
        existing_chains = [gr for gr in self._h5file.listNodes("/")
                           if gr._v_name[:5] == 'chain']

        for chain in existing_chains:
            nchain = int(chain._v_name[5:])
            self._chains[nchain] = chain

            names = []
            for array in chain._f_listNodes():
                name = array._v_name
                self._arrays[nchain, name] = array

                if name not in self._traces:
                    self._traces[name] = Trace(name, db=self)

                names.append(name)

            self.trace_names.append(names)

    @property
    def chains(self):
        return range(len(self._chains))

    @property
    def nchains(self):
        return len(self._chains)

    # def connect_model(self, model):
    #     """Link the Database to the Model instance.
    #     In case a new database is created from scratch, ``connect_model``
    #     creates Trace objects for all tallyable pymc objects defined in
    #     `model`.
    #     If the database is being loaded from an existing file, ``connect_model``
    #     restore the objects trace to their stored value.
    #     :Parameters:
    #     model : pymc.Model instance
    #       An instance holding the pymc objects defining a statistical
    #       model (stochastics, deterministics, data, ...)
    #     """
    # Changed this to allow non-Model models. -AP
    #     if isinstance(model, pymc.Model):
    #         self.model = model
    #     else:
    #         raise AttributeError('Not a Model instance.')
    # Restore the state of the Model from an existing Database.
    # The `load` method will have already created the Trace objects.
    #     if hasattr(self, '_state_'):
    #         names = set()
    #         for morenames in self.trace_names:
    #             names.update(morenames)
    #         for name, fun in six.iteritems(model._funs_to_tally):
    #             if name in self._traces:
    #                 self._traces[name]._getfunc = fun
    #                 names.remove(name)
    #         if len(names) > 0:
    #             raise RuntimeError("Some objects from the database"
    #                                + "have not been assigned a getfunc: %s"
    #                                   % ', '.join(names))
    def _initialize(self, funs_to_tally, length):
        """Create a group named ``chain#`` to store all data for this chain."""

        chain = self.nchains
        self._chains[chain] = self._h5file.create_group(
            '/', 'chain%d' % chain, 'chain #%d' % chain)

        for name, fun in six.iteritems(funs_to_tally):

            arr = np.asarray(fun())

            assert arr.dtype != np.dtype('object')

            array = self._h5file.createEArray(
                self._chains[chain], name,
                tables.Atom.from_dtype(arr.dtype), (0,) + arr.shape,
                filters=self.filter)

            self._arrays[chain, name] = array
            self._traces[name] = Trace(name, getfunc=fun, db=self)
            self._traces[name]._initialize(self.chains, length)

        self.trace_names.append(list(funs_to_tally.keys()))

    def tally(self, chain=-1):
        chain = self.chains[chain]
        for name in self.trace_names[chain]:
            try:
                self._traces[name].tally(chain)
                self._arrays[chain, name].flush()
            except:
                cls, inst, tb = sys.exc_info()
                warnings.warn(warn_tally
                              % (name, ''.join(traceback.format_exception(cls, inst, tb))))
                self.trace_names[chain].remove(name)

    # def savestate(self, state, chain=-1):
    #     """Store a dictionnary containing the state of the Model and its
    #     StepMethods."""
    #     chain = self.chains[chain]
    #     if chain in self._states:
    #         self._states[chain] = state
    #     else:
    #         s = self._h5file.create_vlarray(chain,'_state_',tables.ObjectAtom(),title='The saved state of the sampler',filters=self.filter)
    #         s.append(state)
    #     self._h5file.flush()
    # def getstate(self, chain=-1):
    #     if len(self._chains)==0:
    #         return {}
    #     elif hasattr(self._chains[chain],'_state_'):
    #         if len(self._chains[chain]._state_)>0:
    #             return self._chains[chain]._state_[0]
    #         else:
    #             return {}
    #     else:
    #         return {}
    def _finalize(self, chain=-1):
        self._h5file.flush()

    def close(self):
        self._h5file.close()


def load(dbname, dbmode='a'):
    """Load an existing hdf5 database.

    Return a Database instance.

    :Parameters:
      filename : string
        Name of the hdf5 database to open.
      mode : 'a', 'r'
        File mode : 'a': append, 'r': read-only.
    """
    if dbmode == 'w':
        raise AttributeError("dbmode='w' not allowed for load.")
    db = Database(dbname, dbmode=dbmode)

    return db
