"""
HDF5 database module.

Store the traces in an HDF5 array using pytables.

Implementation Notes
--------------------
This version supports arbitrary objects through ObjectAtom and VLArray
constructs. Ordinary numerical objects are stored in a Table. Each chain
is stored in an individual group called ``chain#``.

Additional Dependencies
-----------------------
 * HDF5 version 1.6.5, required by pytables.
 * pytables version 2 and up.  <http://sourceforge.net/projects/pytables/>

"""


import numpy as np
from numpy import zeros,shape, asarray, hstack, size, dtype
import pymc
from pymc.database import base, pickle
from copy import copy
import tables
import os, warnings, sys, traceback
import warnings

from pymc import six


__all__ = ['Trace', 'Database', 'load']


class Trace(base.Trace):
    """HDF5 trace.

    Database backend based on the HDF5 format.
    """


    def tally(self, chain):
        """Adds current value to trace"""
        # print chain, self._getfunc()
        print (chain, self.name)
        self.db._arrays[chain, self.name].append(self._getfunc())


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

        if chain is not None:
            tables = [self.db._gettables()[chain],]
        else:
            tables = self.db._gettables()

        for i,table in enumerate(tables):
            if slicing is not None:
                burn, stop, thin = slicing.start, slicing.stop, slicing.step
            if slicing is None or stop is None:
                stop = table.nrows
            col = table.read(start=burn, stop=stop, step=thin, field=self.name)
            if i == 0:
                data = np.asarray(col)
            else:
                data = hstack((data, col))

        return data


    __call__ = gettrace



    def length(self, chain=-1):
        """Return the length of the trace.

        :Parameters:
        chain : int or None
          The chain index. If None, returns the combined length of all chains.
        """
        if chain is not None:
            tables = [self.db._gettables()[chain],]
        else:
            tables = self.db._gettables()

        n = asarray([table.nrows for table in tables])
        return n.sum()


class Database(pickle.Database):
    """HDF5 database.

    Create an HDF5 file <model>.h5.  Each chain is stored in a group,
    and the stochastics and deterministics are stored as arrays in
    each group.
    """
    

    def __init__(self, dbname, dbmode='a',
                 dbcomplevel=0, dbcomplib='zlib',
                 **kwds):
        """Create an HDF5 database instance, where samples are stored in tables.

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
          * LZO is a fast compression library offering however a low compression
            ratio.
          * bzip2 has an excellent compression ratio but requires more CPU.
        """
        
        self.__name__ = 'hdf52'
        self.dbname = dbname
        self.__Trace__ = Trace
        self.mode = dbmode

        self.trace_names = []
        self._traces = {}
        

        db_exists    = os.path.exists(self.dbname)
        self._h5file = tables.openFile(self.dbname, self.mode)

        default_filter = tables.Filters(complevel=dbcomplevel,
                                        complib=dbcomplib)

        if self.mode =='r' or (self.mode=='a' and db_exists):
            self.filter = getattr(self._h5file, 'filters', default_filter)
        else:
            self.filter = default_filter

        self._tables = self._gettables()

        # self._chains = [ gr for gr in self._h5file.listNodes("/")
        #                  if gr._v_name[:5] == 'chain']

        self._chains = {}
        self._arrays = {}

        # # LOAD LOGIC
        # if self.chains > 0:
        #     # Create traces from objects stored in Table.
        #     db = self
        #     for k in db._tables[-1].colnames:
        #         db._traces[k] = Trace(name=k, db=db)
        #         setattr(db, k, db._traces[k])


        #     # Walk nodes proceed from top to bottom, so we need to invert
        #     # the list to have the chains in chronological order.
        #     objects = {}
        #     for chain in self._chains:
        #         for node in db._h5file.walkNodes(chain, classname='VLArray'):
        #             if node._v_name != '_state_':
        #                 try:
        #                     objects[node._v_name].append(node)
        #                 except:
        #                     objects[node._v_name] = [node,]

        #     # Note that the list vlarrays is in reverse order.
        #     for k, vlarrays in six.iteritems(objects):
        #         db._traces[k] = TraceObject(name=k, db=db, vlarrays=vlarrays)
        #         setattr(db, k, db._traces[k])

        #     # Restore table attributes.
        #     # This restores the sampler's state for the last chain.
        #     table = db._tables[-1]
        #     for k in table.attrs._v_attrnamesuser:
        #         setattr(db, k, getattr(table.attrs, k))

        #     # Restore group attributes.
        #     for k in db._chains[-1]._f_listNodes():
        #         if k.__class__ not in [tables.Table, tables.Group]:
        #             setattr(db, k.name, k)

        #     varnames = db._tables[-1].colnames+ objects.keys()
        #     db.trace_names = db.chains * [varnames,]


    @property
    def chains(self):
        return len(self._chains)


    def connect_model(self, model):
        """Link the Database to the Model instance.

        In case a new database is created from scratch, ``connect_model``
        creates Trace objects for all tallyable pymc objects defined in
        `model`.

        If the database is being loaded from an existing file, ``connect_model``
        restore the objects trace to their stored value.

        :Parameters:
        model : pymc.Model instance
          An instance holding the pymc objects defining a statistical
          model (stochastics, deterministics, data, ...)
        """

        # Changed this to allow non-Model models. -AP
        if isinstance(model, pymc.Model):
            self.model = model
        else:
            raise AttributeError('Not a Model instance.')

        # Restore the state of the Model from an existing Database.
        # The `load` method will have already created the Trace objects.
        if hasattr(self, '_state_'):
            names = set()
            for morenames in self.trace_names:
                names.update(morenames)
            for name, fun in six.iteritems(model._funs_to_tally):
                if name in self._traces:
                    self._traces[name]._getfunc = fun
                    names.remove(name)
            if len(names) > 0:
                raise RuntimeError("Some objects from the database have not been assigned a getfunc: %s"% ', '.join(names))


    def _initialize(self, funs_to_tally, length):
        """
        Create a group named ``Chain#`` to store all data for this chain.
        The group contains one pyTables Table, and at least one subgroup
        called ``group#``. This subgroup holds ObjectAtoms, which can hold
        pymc objects whose value is not a numerical array.

        There is too much stuff in here. ObjectAtoms should get initialized
        """

        chain = self.chains
        self._chains[chain] = self._h5file.createGroup(
            '/', 'chain%d' % chain, 'chain #%d' % chain)

        for name, fun in six.iteritems(funs_to_tally):

            arr = asarray(fun())

            if arr.dtype == np.dtype('object'):
                print 'WARNING, SKIPPING ', name
                #self._traces[name] = TraceObject(name, getfunc=fun, db=self)
            else:
                print 'added', chain, name, arr.dtype
                
                array = self._h5file.createEArray(
                    self._chains[chain], name,
                    tables.Atom.from_dtype(arr.dtype), arr.shape + (0,),
                    filters=self.filter)

                self._arrays[chain, name] = array
                self._traces[name] = Trace(name, getfunc=fun, db=self)

            self._traces[name]._initialize(self.chains, length)

        self.trace_names.append(funs_to_tally.keys())


    def tally(self, chain=-1):
        chain = range(self.chains)[chain]
        for name in self.trace_names[chain]:
            try:
                self._traces[name].tally(chain)
                self._arrays[chain, name].flush()
            except:
                cls, inst, tb = sys.exc_info()
                warnings.warn("""
Error tallying %s, will not try to tally it again this chain.
Did you make all the same variables and step methods tallyable
as were tallyable last time you used the database file?

Error:

%s""" % (name, ''.join(traceback.format_exception(cls, inst, tb))))
                self.trace_names[chain].remove(name)

        # self._rows[chain].append()
        # self._arrays[chain].flush()
        # self._rows[chain] = self._tables[chain].row

    def _finalize(self, chain=-1):
        """Close file."""
        # add attributes. Computation time.
        #self._tables[chain].flush()
        self._h5file.flush()


    def savestate(self, state, chain=-1):
        """Store a dictionnary containing the state of the Model and its
        StepMethods."""
        
        cur_chain = range(self.chains)[chain]
        if hasattr(cur_chain, '_state_'):
            cur_chain._state_[0] = state
        else:
            s = self._h5file.createVLArray(cur_chain,'_state_',tables.ObjectAtom(),title='The saved state of the sampler',filters=self.filter)
            s.append(state)
        self._h5file.flush()

    def getstate(self, chain=-1):
        if len(self._chains)==0:
            return {}
        elif hasattr(self._chains[chain],'_state_'):
            if len(self._chains[chain]._state_)>0:
                return self._chains[chain]._state_[0]
            else:
                return {}
        else:
            return {}

    def _model_trace_description(self):
        """Return a description of the table and the ObjectAtoms to be created.

        :Returns:
        table_description : dict
          A Description of the pyTables table.
        ObjectAtomsn : dict
          A
      in terms of PyTables
        columns, and a"""
        D = {}
        for name, fun in six.iteritems(self.model._funs_to_tally):
            arr = asarray(fun())
            D[name] = tables.Col.from_dtype(dtype((arr.dtype,arr.shape)))
        return D, {}

    def _file_trace_description(self):
        """Return a description of the last trace stored in the database."""
        table = self._gettables()[-1][0]
        return table.description

    def _check_compatibility(self):
        """Make sure the next objects to be tallied are compatible with the
        stored trace."""
        stored_descr = self._file_trace_description()
        try:
            for k,v in self._model_trace_description():
                assert(stored_descr[k][0]==v[0])
        except:
            raise ValueError("The objects to tally are incompatible with the objects stored in the file.")

    def _gettables(self):
        """Return a list of hdf5 tables name PyMCsamples."""

        groups = self._h5file.listNodes("/")
        if len(groups) == 0:
            return []
        else:
            return [gr.PyMCsamples for gr in groups if gr._v_name[:5]=='chain']

    def close(self):
        self._h5file.close()

    def add_attr(self, name, object, description='', chain=-1, array=False):
        """Add an attribute to the chain.

        description may not be supported for every date type.
        if array is true, create an Array object.
        """

        if not np.isscalar(chain):
            raise TypeError("chain must be a scalar integer.")

        table = self._tables[chain]

        if array is False:
            table.setAttr(name, object)
            obj = getattr(table.attrs, name)
        else:
            # Create an array in the group
            if description == '':
                description = name
            group = table._g_getparent()
            self._h5file.createArray(group, name, object, description)
            obj = getattr(group, name)

        setattr(self, name, obj)


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


# TODO: Check this. It seems that pickle is pymc.database.pickle, not the pickle module.

def save_sampler(sampler):
    """
    Dumps a sampler into its hdf5 database.
    """
    db = sampler.db
    fnode = tables.filenode.newnode(db._h5file, where='/', name='__sampler__')
    import pickle
    pickle.dump(sampler, fnode)


def restore_sampler(fname):
    """
    Creates a new sampler from an hdf5 database.
    """
    hf = tables.openFile(fname)
    fnode = hf.root.__sampler__
    import pickle
    sampler = pickle.load(fnode)
    return sampler


