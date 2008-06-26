###
# HDF5 database module. Version 2. 
# Store the traces in an HDF5 array using pytables.
# Dependencies
# pytables >=2 <http://sourceforge.net/projects/pytables/>
# HDF5 >= 1.6.5
# Numpy >= xxx
###

# TODO: add a command to save attributes (chain attributes, group attributes)


import numpy as np
from numpy import zeros,shape, asarray, hstack, size, dtype
import pymc
from pymc.database import base, pickle
from copy import copy
import tables

od = np.dtype('object')


class Trace(base.Trace):
    """HDF5 trace

    Database backend based on the HDF5 format.
    """
    def __init__(self, value=None, obj=None, name=None):
        """Assign an initial value and an internal PyMC object."""
        self._trace = value
        if obj is not None:
            if isinstance(obj, pymc.Variable):
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
        self._array = self.db.trace_dict[self._obj]
        if self.db.dtype_dict[self._obj] is od:
            self.isnum = False
        else:
            self.isnum = True
        
    
    def tally(self):
        """Adds current value to trace"""
        if not self.isnum:
            self._array.append(self._obj.value)
        else:
            self._array.append(np.atleast_1d(self._obj.value))
                       
    def truncate(self, index):
        """
        When model receives a keyboard interrupt, it tells the traces
        to truncate their values.
        """
        if self.isnum:
            self._array.truncate(len(self._array))

    def gettrace(self, chain=-1):
        """Return the trace (last by default).

        Input:
          - chain (int): The index of the chain to fetch. If None, return all chains.
        """
        if chain is not None:
            if chain==-1:
                chain = len(self.db._h5file.listNodes('/')) - 1
            chain = getattr(self.db._h5file.root, 'chain%d'%chain)
            ### FIXME How to guarantee that this object is in the same group
            ### in every chain?
            group = getattr(chain, self.db.groupnum_dict[self._obj])
            return getattr(chain, self._obj.__name__)
        else:
            chains = []
            for i in xrange(len(self.db._h5file.listNodes('/'))):
                chain = getattr(self.db._h5file.root, 'chain%d'%i)
                group = getattr(chain, self.db.groupnum_dict[self._obj])
                chains.append(getattr(chain, self._obj.__name__))
            return chains
                      
    def _finalize(self):
        """Nothing done here."""
        pass

    __call__ = gettrace

    def length(self, chain=-1):
        """Return the sample length of given chain. If chain is None,
        return the total length of all chains."""
        if chain==-1:
            chain = len(self.db._h5file.listNodes('/')) - 1
        chain = getattr(self.db._h5file.root, 'chain%d'%chain)
        group = getattr(chain, self.db.groupnum_dict[self])
        return len(getattr(group, self._obj.__name__))
        

class Database(pickle.Database):
    """HDF5 database

    Create an HDF5 file <model>.h5. Each chain is stored in a group, and the
    stochastics and deterministics are stored as arrays in each group.

    """
    def __init__(self, filename=None, mode='w', complevel=0, complib='zlib', **kwds):
        """Create an HDF5 database instance, where samples are stored in tables. 
        
        :Parameters:
          filename : string
            Specify the name of the file the results are stored in. 
          mode : 'a', 'w', 'r', or any integer
            File mode: 'a': append, 'w': overwrite, 'r': read-only.
            If mode is an integer, new samples will be appended to that chain.
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
            if not self.mode.__class__ is str:
                open_mode = 'a'
            else:
                open_mode = self.mode

            self._h5file = tables.openFile(self.filename, open_mode)

        # If mode is an integer i, grab the i'th chain to use it.
        if not self.mode.__class__ is str:
            i = self.mode
            self._group = self._h5file.getNode('/chain%d'%i)

        # If mode is not an integer, initialize a new chain (group)
        else:

            self.dtype_dict = {}
            self.trace_dict = {}
            self.groupnum_dict = {}
    
        
    def _initialize(self, length):
        """Create group for the current chain."""
        i = len(self._h5file.listNodes('/'))
        self._group = self._h5file.createGroup("/", 'chain%d'%i, 'Chain #%d'%i)
        
        group_counter = 0
        group_num = 0
        this_group = self._h5file.createGroup(self._group, 'group%d'%group_num, 'PyTables requires that at most 4096 nodes\
 descend from a single group, so the traces are split up into several groups.')
        
        if self.mode.__class__ is str:
        
            # Make EArrays for the numerical- or array-valued variables
            # and VLArrays for the others.
            for o in self.model._variables_to_tally:
                group_counter += 1
                                
                arr_value = np.asarray(o.value)
                title = o.__name__ + ': samples from %s' % self._group._v_name
                self.dtype_dict[o] = arr_value.dtype
                self.groupnum_dict[o] = group_counter
                
                if arr_value.dtype is od:
                    self.trace_dict[o] = self._h5file.createVLArray(this_group, o.__name__, tables.ObjectAtom(), 
                        title=title, filters=self.filter) 
                
                else:
                    self.trace_dict[o] = self._h5file.createEArray(this_group, o.__name__, 
                        tables.Atom.from_dtype(arr_value.dtype, arr_value.shape), (0,), title=title, filters=self.filter)

                if group_counter % 4096 == 0:
                    group_num += 1
                    this_group = self._h5file.createGroup(self._group, 'group%d'%group_num, 'PyTables requires that at most 4096 nodes\
 descend from a single group, so the traces are split up into several groups.')
                    

        for object in self.model._variables_to_tally:
            object.trace._initialize(length)

            
        # Store data objects
        for o in self.model.data_stochastics:
            if o.trace is True:
                
                group_counter += 1
                
                arr_value = np.asarray(o.value)
                title = o.__name__ + ': observed value'
                
                if arr_value.dtype is dtype('object'):
                    va = self._h5file.createVLArray(this_group, o.__name__, tables.ObjectAtom(), title=title, filters=self.filter)
                    va.append(o.value)
                    
                else:
                    ca = self._h5file.createCArray(this_group, o.__name__, tables.Atom.from_dtype(arr_value.dtype, arr_value.shape), 
                        (1,), title=title, filters=self.filter)
                    ca[0] = o.value
                    
                if group_counter % 4096 == 0:
                    group_num += 1
                    this_group = self._h5file.createGroup(self._group, 'group%d'%group_num, 'PyTables requires that at most 4096 nodes\
descend from a single group, so the traces are split up into several groups.')
                
    
    def tally(self, index):
        for o in self.model._variables_to_tally:
            o.trace.tally()
        
    def _finalize(self):
        """Close file."""
        self._h5file.flush()
        
    def savestate(self, state):
        """Store a dictionnary containing the state of the Model and its 
        StepMethods."""
        if not self._group.__contains__('__state__'):
            va = self._h5file.createVLArray(self._group, '__state__', tables.ObjectAtom(), title='The state of the sampler', filters=self.filter)
        va.append(state)
            
    def _check_compatibility(self):
        """Make sure the next objects to be tallied are compatible with the 
        stored trace."""
        for o in self.model._variables_to_tally:
            if not np.asarray(o.value).dtype == self.dtype_dict[0]:
                raise ValueError, "Variable %s's current dtype, '%s', is different from trace dtype '%s'. Cannot tally it." \
                    % (o.__name__, np.asarray(o.value).dtype.name, self.dtype_dict[0].name)
        
    def close(self):
        self._h5file.close()
        
    def getstate(self):
        if self._group.__contains__('__state__'):
            return self._group.__state__[-1]
        else:
            return None


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
    if not mode.__class__ is str:
        open_mode = 'a'
        i = mode
    else:
        open_mode = mode
        i = -1

    db._h5file = tables.openFile(filename, open_mode)

    if i!=-1:
        db._group = getattr(db.root, 'chain%d'%i)

    return db
        
##    groups = db._h5file.root._g_listGroup()[0]
##    groups.sort()
##    last_chain = '/'+groups[-1]
##    db._table = db._h5file.getDeterministic(last_chain, 'PyMCsamples')

