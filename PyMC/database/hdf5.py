###
# HDF5 database module. Version 2. 
# Store the traces in an HDF5 array using pytables.
# Dependencies
# pytables >=2 <http://sourceforge.net/projects/pytables/>
# HDF5 >= 1.6.5
# Numarray >= 1.5.2 (eventually will rely on numpy)
###

# TODO: add a command to save attributes (chain attributes, group attributes)


import numpy as np
from numpy import zeros,shape, asarray, hstack, size, dtype
import PyMC
from PyMC.database import base, pickle
from copy import copy
import tables

class Trace(base.Trace):
    """HDF5 trace

    Database backend based on the HDF5 format.
    """
    def __init__(self,value=None, obj=None, name=None):
        """Assign an initial value and an internal PyMC object."""
        self._trace = value
        if obj is not None:
            if isinstance(obj, PyMC.Variable):
                self._obj = obj
                self.name = self._obj.__name__
            else:
                raise AttributeError, 'Not PyMC object', obj
        else:
            self.name = name
            
    def _initialize(self, length):
        """Make sure the object name is known"""
        if self.name is None:
            self.name = self._obj.__name__
    
    def tally(self):
        """Adds current value to trace"""
        self.db._row[self.name] = self._obj.value
               
    def truncate(self, index):
        """
        When model receives a keyboard interrupt, it tells the traces
        to truncate their values.
        """
        pass

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).

        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if slicing is not None:
            burn, stop, thin = slicing.start, slicing.stop, slicing.step
            
        tables = self.db._gettable(chain)
        
        for i,table in enumerate(tables):
            if slicing is None:
                stop = table.nrows
            col = table.read(start=burn, stop=stop, step=thin, field=self.name)
            if i == 0:
                data = col
            else:
                data = hstack((data, col))
        
        return data
                            
                      
    def _finalize(self):
        """Nothing done here."""
        pass

    __call__ = gettrace

    def length(self, chain=-1):
        """Return the sample length of given chain. If chain is None,
        return the total length of all chains."""
        tables = self.db._gettable(chain)
        n = asarray([table.nrows for table in tables])
        return n.sum()

class Database(pickle.Database):
    """HDF5 database

    Create an HDF5 file <model>.h5. Each chain is stored in a group, and the
    stochs and dtrms are stored as arrays in each group.

    """
    def __init__(self, filename=None, mode='w', complevel=0, complib='zlib', **kwds):
        """Create an HDF5 database instance, where samples are stored in tables. 
        
        :Parameters:
          filename : string
            Specify the name of the file the results are stored in. 
          mode : 'a', 'w', 'r'
            File mode: 'a': append, 'w': overwrite, 'r': read-only.
          complevel : integer (0-9)
            Compression level, 0: no compression.
          complib : string
            Compression library (zlib, bzip2, lzo)
            
        :Notes:
          - zlib has a good compression ratio, although somewhat slow, and 
            reasonably fast decompression.
          - LZO is a fast compression library offering however a low compression
            ratio. 
          - bzip2 has an excellent compression ratio but requires more CPU. 
        """
        self.filename = filename
        self.Trace = Trace
        self.filter = tables.Filters(complevel=complevel, complib=complib)
        self.mode = mode
        
    def connect(self, sampler):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        base.Database.connect(self, sampler)
        self.choose_name('hdf5')
        if not hasattr(self, '_h5file'):
            self._h5file = tables.openFile(self.filename, self.mode)
        root = self._h5file.root
        
        
    def _initialize(self, length):
        """Create group for the current chain."""
        i = len(self._h5file.listNodes('/'))+1
        self._group = self._h5file.createGroup("/", 'chain%d'%i, 'Chain #%d'%i)
        
        self._table = self._h5file.createTable(self._group, 'PyMCsamples', \
            self._model_trace_description(), 'PyMC samples from chain %d'%i, \
            filters=self.filter, expectedrows=length)
        self._row = self._table.row
        for object in self.model._variables_to_tally:
            object.trace._initialize(length)
        
        # Store data objects
        for object in self.model.data:
            if object.trace is True:
                setattr(self._table.attrs, object.__name__, object.value)
    
    def tally(self, index):
        for o in self.model._variables_to_tally:
            o.trace.tally()
        self._row.append()
        self._table.flush()
        self._row = self._table.row
        
    def _finalize(self):
        """Close file."""
        # add attributes. Computation time.
        self._table.flush()
        
    def savestate(self, state):
        """Store a dictionnary containing the state of the Model and its 
        StepMethods."""
        self.add_attr('_state_', state, 'Final state of the sampler.')
            
    def _model_trace_description(self):
        """Return a description of the table to be created in terms of PyTables columns."""
        D = {}
        for o in self.model._variables_to_tally:
            arr = asarray(o.value)
            #if len(s) == 0: 
            #    D[o.__name__] = tables.Col.from_scdtype(arr.dtype)
            #else:
            D[o.__name__] = tables.Col.from_dtype(dtype((arr.dtype,arr.shape)))
        return D

    def _file_trace_description(self):
        """Return a description of the last trace stored in the database."""
        table = self._gettable(-1)[0]
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
            
    def _gettable(self, chain=-1):
        """Return the list of hdf5 tables corresponding to chain. 
        
        chain : scalar or sequence.
        """
        
        groups = self._h5file.listNodes("/")
        nchains = len(groups)    
        if chain == -1:
            chains = [nchains-1]    # Index of last group
        elif chain is None:
            chains = range(nchains)
        elif size(chain) == 1:
           chains = [chain]
        
        if len(chains) == 0:
            raise LookupError, 'No table found, that is, no data in file.'
            
        table = []
        for i,c in enumerate(chains):
            gr = groups[c]
            table.append(getattr(gr, 'PyMCsamples'))

        return table


    def close(self):
        self._h5file.close()

    def add_attr(self, name, object, description='', chain=-1, array=False):
        """Add an attribute to the chain.
        
        description may not be supported for every date type. 
        if array is true, create an Array object. 
        """
        if not np.isscalar(chain):
            raise TypeError, "chain must be a scalar integer."
        table = self._gettable(chain)[0]
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
        

def load(filename, mode='a'):
    """Load an existing hdf5 database.

    Return a Database instance.
    
    :Parameters:
      filename : string
        Name of the hdf5 database to open.
      mode : 'a', 'r'
        File mode : 'a': append, 'r': read-only.
    """ 
    if mode == 'w':
        raise AttributeError, "mode='w' not allowed for load."
    db = Database(filename)
    db._h5file = tables.openFile(filename, mode)
    db._table = db._gettable(-1)[0]
    db._group = db._table._g_getparent()
    for k in db._table.colnames:
        setattr(db, k, Trace(name=k))
        o = getattr(db,k)
        setattr(o, 'db', db)
    for k in db._table.attrs._v_attrnamesuser:
        setattr(db, k, getattr(db._table.attrs, k))
    for k in db._group._f_listNodes():
        if k.__class__ is not tables.table.Table:
            setattr(db, k.name, k)
    return db
        
##    groups = db._h5file.root._g_listGroup()[0]
##    groups.sort()
##    last_chain = '/'+groups[-1]
##    db._table = db._h5file.getDeterministic(last_chain, 'PyMCsamples')

