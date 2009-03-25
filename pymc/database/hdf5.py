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

__all__ = ['Trace', 'Database', 'load']

class TraceObject(base.Trace):
    """HDF5 Trace for Objects."""
    def __init__(self, name, getfunc=None, db=None, vlarrays=None):
        """Create a Trace instance.

        :Parameters:
        obj : pymc object
          A Stochastic or Determistic object.
        name : string
          The trace object name. This is used only if no `obj` is given.
        db : Database instance
          The database owning this Trace.
        vlarrays : sequence
          The nodes storing the data for this object.
         """

        base.Trace.__init__(self, name, getfunc=getfunc, db=db)
        if vlarrays is None:
            vlarrays = []
        self._vlarrays = vlarrays  # This should be a dict keyed by chain.


    def tally(self, chain):
        """Adds current value to trace"""
        # try:
        self._vlarrays[chain].append(self._getfunc())
        # except:
        #     print self._vlarrays, chain
        #     raise AttributeError

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
            return np.array(self._vlarrays[chain])
        else:
            return np.concatenate(self._vlarrays[:])

    __call__ = gettrace

    def length(self, chain=-1):
        """Return the length of the trace.

        :Parameters:
        chain : int or None
          The chain index. If None, returns the combined length of all chains.
        """
        if chain is not None:
            return len(self._vlarrays[chain])
        else:
            return sum(map(len, self._vlarrays))


class Trace(base.Trace):
    """HDF5 trace

    Database backend based on the HDF5 format.
    """

    def tally(self, chain):
        """Adds current value to trace"""
        self.db._rows[chain][self.name] = self._getfunc()

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

    def hdf5_col(self, chain=-1):
        """Return a pytables column object.

        :Parameters:
        chain : integer
          The index of the chain.

        .. note::
           This method is specific to the ``hdf5`` backend.
        """
        return self.db._tables[chain].colinstances[self.name]

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
    """HDF5 database

    Create an HDF5 file <model>.h5. Each chain is stored in a group, and the
    stochastics and deterministics are stored as arrays in each group.

    """
    def __init__(self, dbname, dbmode='a', dbcomplevel=0, dbcomplib='zlib', **kwds):
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
        self.__name__ = 'hdf5'
        self.dbname = dbname
        self.__Trace__ = Trace
        self.mode = dbmode

        self.trace_names = []   # A list of sequences of names of the objects to tally.
        self._traces = {} # A dictionary of the Trace objects.

        # Deprecation of complevel and complib
        # Remove in V2.1
        if kwds.has_key('complevel'):
            warnings.warn('complevel has been replaced with dbcomplevel.', DeprecationWarning)
            dbcomplevel = kwds.get('complevel')
        if kwds.has_key('complib'):
            warnings.warn('complib has been replaced with dbcomplib.', DeprecationWarning)
            dbcomplib = kwds.get('complib')


        self._h5file = tables.openFile(self.dbname, self.mode)

        self.filter = getattr(self._h5file, 'filters', \
                              tables.Filters(complevel=dbcomplevel, complib=dbcomplib))


        self._tables = self._gettables()  # This should be a dict keyed by chain.
        self._rows = len(self._tables) * [None,] # This should be a dict keyed by chain.
        self._chains = self._h5file.listNodes("/")  # This should be a dict keyed by chain.
        self.chains = len(self._chains)
        self._default_chain = -1

        # LOAD LOGIC
        if self.chains > 0:
            # Create traces from objects stored in Table.
            db = self
            for k in db._tables[-1].colnames:
                db._traces[k] = Trace(name=k, db=db)
                setattr(db, k, db._traces[k])


            # Walk nodes proceed from top to bottom, so we need to invert
            # the list to have the chains in chronological order.
            objects = {}
            for node in db._h5file.walkNodes("/", classname='VLArray'):
                    try:
                        objects[node._v_name].append(node)
                    except:
                        objects[node._v_name] = [node,]

            # Note that the list vlarrays is in reverse order.
            for k, vlarrays in objects.iteritems():
                db._traces[k] = TraceObject(name=k, db=db, vlarrays=vlarrays[::-1])
                setattr(db, k, db._traces[k])

            # Restore table attributes.
            # This restores the sampler's state for the last chain.
            table = db._tables[-1]
            for k in table.attrs._v_attrnamesuser:
                setattr(db, k, getattr(table.attrs, k))

            # Restore group attributes.
            for k in db._chains[-1]._f_listNodes():
                if k.__class__ not in [tables.Table, tables.Group]:
                    setattr(db, k.name, k)

            varnames = db._tables[-1].colnames+ objects.keys()
            db.trace_names = db.chains * [varnames,]

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
            raise AttributeError, 'Not a Model instance.'

        # Restore the state of the Model from an existing Database.
        # The `load` method will have already created the Trace objects.
        if hasattr(self, '_state_'):
            names = set(reduce(list.__add__, self.trace_names))
            for fun, name in model._funs_to_tally.iteritems():
                if self._traces.has_key(name):
                    self._traces[name]._getfunc = fun
                    names.remove(name)
            if len(names) > 0:
                print "Some objects from the database have not been assigned a getfunc", names

        # Create a fresh new state. This is now taken care of in initialize.
        else:
            for name, fun in model._funs_to_tally.iteritems():
                if np.array(fun()).dtype is np.dtype('object'):
                    self._traces[name] = TraceObject(name, getfunc=fun, db=self)
                else:
                    self._traces[name] = Trace(name, getfunc=fun, db=self)

    def nchains(self):
        """Return the number of existing chains."""
        return len(self._h5file.listNodes('/'))

    def _initialize(self, funs_to_tally, length):
        """
        Create a group named ``Chain#`` to store all data for this chain.
        The group contains one pyTables Table, and at least one subgroup
        called ``group#``. This subgroup holds ObjectAtoms, which can hold
        pymc objects whose value is not a numerical array.

        There is too much stuff in here. ObjectAtoms should get initialized
        """
        i = self.chains
        self._chains.append(self._h5file.createGroup("/", 'chain%d'%i, 'Chain #%d'%i))
        current_object_group = self._h5file.createGroup(self._chains[-1], 'group0', 'Group storing objects.')
        group_counter = 0
        object_counter = 0

        # Create the Table in the chain# group, and ObjectAtoms in chain#/group#.
        table_descr = {}
        for name, fun in funs_to_tally.iteritems():

            arr = asarray(fun())

            if arr.dtype is np.dtype('object'):

                self._traces[name]._vlarrays.append(self._h5file.createVLArray(\
                            current_object_group,
                            name, \
                            tables.ObjectAtom(),  \
                            title=name + ' samples.',
                            filters=self.filter))

                object_counter += 1
                if object_counter % 4096 == 0:
                    group_counter += 1
                    current_object_group = self._h5file.createGroup(self._chains[-1], \
                        'group%d'%group_counter, 'Group storing objects.')


            else:
                table_descr[name] = tables.Col.from_dtype(dtype((arr.dtype,arr.shape)))


        table = self._h5file.createTable(self._chains[-1], \
            'PyMCsamples', \
            table_descr, \
            title='PyMC samples', \
            filters=self.filter,
            expectedrows=length)

        self._tables.append(table)
        self._rows.append(self._tables[-1].row)

        # Store data objects
        for object in self.model.observed_stochastics:
            if object.trace is True:
                setattr(table.attrs, object.__name__, object.value)


       # Make sure the variables have a corresponding Trace instance.
        for name, fun in funs_to_tally.iteritems():
            if not self._traces.has_key(name):
                if np.array(fun()).dtype is np.dtype('object'):
                    self._traces[name] = TraceObject(name, getfunc=fun, db=self)
                else:
                    self._traces[name] = Trace(name, getfunc=fun, db=self)


            self._traces[name]._initialize(self.chains, length)


        self.trace_names.append(funs_to_tally.keys())
        self.chains += 1

    def tally(self, chain=-1):
        chain = range(self.chains)[chain]
        for name in self.trace_names[chain]:
            try:
                self._traces[name].tally(chain)
            except:
                cls, inst, tb = sys.exc_info()
                print """
Error tallying %s, will not try to tally it again this chain. 
Did you make all the samevariables and step methods tallyable 
as were tallyable last time you used the database file?

Error:

%s"""%(name, ''.join(traceback.format_exception(cls, inst, tb)))
                self.trace_names[chain].remove(name)
            
        self._rows[chain].append()
        self._tables[chain].flush()
        self._rows[chain] = self._tables[chain].row

    def _finalize(self, chain=-1):
        """Close file."""
        # add attributes. Computation time.
        #self._tables[chain].flush()
        self._h5file.flush()


    def savestate(self, state):
        """Store a dictionnary containing the state of the Model and its
        StepMethods."""
        self.add_attr('_state_', state, 'Final state of the sampler.')
        self._h5file.flush()

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
        for name, fun in self.model._funs_to_tally.iteritems():
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
            raise "The objects to tally are incompatible with the objects stored in the file."

    def _gettables(self):
        """Return a list of hdf5 tables name PyMCsamples.
        """

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
            raise TypeError, "chain must be a scalar integer."

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
        raise AttributeError, "dbmode='w' not allowed for load."
    db = Database(dbname, dbmode=dbmode)

    return db


# TODO: Check this. It seems that pickle is pymc.database.pickle, not the pickle module.

def save_sampler(sampler):
    """
    Dumps a sampler into its hdf5 database.
    """
    db = sampler.db
    fnode = tables.filenode.newnode(db._h5file, where='/', name='__sampler__')
    pickle.dump(sampler, fnode)


def restore_sampler(fname):
    """
    Creates a new sampler from an hdf5 database.
    """
    hf = tables.openFile(fname)
    fnode = hf.root.__sampler__
    sampler = pickle.load(fnode)
    return sampler


