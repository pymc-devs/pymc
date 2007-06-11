###
# HDF5 database module. Version 2. 
# Store the traces in an HDF5 array using pytables.
# Dependencies
# pytables >=2 <http://sourceforge.net/projects/pytables/>
# HDF5 >= 1.6.5
# Numarray >= 1.5.2 (eventually will rely on numpy)
###


from numpy import zeros,shape, asarray, hstack, size, dtype
import PyMC2
from PyMC2.database import base, pickle
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
            if isinstance(obj, PyMC2.PyMCBase):
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
        self.db.row[self.name] = self._obj.value
               
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
            
        groups = self.db.h5file.listNodes("/")
        nchains = len(groups)-1     # -1 to remove root group
        if chain == -1:
            chains = [nchains-1]    # Index of last group
        elif chain is None:
            chains = range(nchains)
        elif size(chain) == 1:
           chains = [chain]
        
        for i,c in enumerate(chains):
            gr = groups[c]
            table = getattr(gr, 'PyMCsamples')
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

class Database(pickle.Database):
    """HDF5 database

    Create an HDF5 file <model>.h5. Each chain is stored in a group, and the
    parameters and nodes are stored as arrays in each group.

    """
    def __init__(self, filename=None, complevel=0, complib='zlib'):
        """Create an HDF5 database instance, where samples are stored in tables. 
        
        :Parameters:
          filename : string
            Specify the name of the file the results are stored in. 
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
        
        
    def connect(self, sampler):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the PyMC objects to tally.
        """
        base.Database.connect(self, sampler)
        self.choose_name('hdf5')
        try:
            self.h5file = tables.openFile(self.filename, 'a')
        except IOError:
            print "Database file seems already open. Skipping."
        root = self.h5file.root
        try:
            self.main = self.h5file.createGroup(root, "main")
        except tables.exceptions.NodeError:
            pass
        
    def _initialize(self, length):
        """Create group for the current chain."""
        i = len(self.h5file.listNodes('/'))
        self.group = self.h5file.createGroup("/", 'chain%d'%i, 'Chain #%d'%i)
        
        self.table = self.h5file.createTable(self.group, 'PyMCsamples', self.description(), 'PyMC samples from chain %d'%i, filters=self.filter)
        self.row = self.table.row
        for object in self.model._pymc_objects_to_tally:
            object.trace._initialize(length)
    
    def tally(self, index):
        for o in self.model._pymc_objects_to_tally:
            o.trace.tally()
        self.row.append()
        self.table.flush()
        self.row = self.table.row
        
    def _finalize(self):
        """Close file."""
        # add attributes. Computation time.
        self.table.flush()
        

    def description(self):
        """Return a description of the table to be created in terms of PyTables columns."""
        D = {}
        for o in self.model._pymc_objects_to_tally:
            arr = asarray(o.value)
            #if len(s) == 0: 
            #    D[o.__name__] = tables.Col.from_scdtype(arr.dtype)
            #else:
            D[o.__name__] = tables.Col.from_dtype(dtype((arr.dtype,arr.shape)))
        return D

    def close(self):
        self.h5file.close()


def load(filename, mode='a'):
    """Load an existing hdf5 database.

    Return a Database instance.
    """ 
    db = Database(filename)
    db.h5file = tables.openFile(filename, mode)
    table = db.h5file.root.chain1.PyMCsamples
    for k in table.colnames:
        if k == '_state_':
           db._state_ = v
        else:
            setattr(db, k, Trace(name=k))
            o = getattr(db,k)
            setattr(o, 'db', db)
    return db
        
