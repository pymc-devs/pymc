###
# HDF5 database module
# Store the traces in an HDF5 array using pytables.
# Dependencies
# pytables: <http://sourceforge.net/projects/pytables/>
# HDF5 >= 1.6.5 
# Numarray >= 1.5.2 (eventually will rely on numpy)
###

from numpy import zeros,shape
import tables

class Trace(object):
    """HDF5 trace 
    
    Database backend based on the HDF5 format.
    """
    def __init__(self, obj, db):
        """Initialize the instance.
        :Parameters:
          obj : PyMC object
            Node or Parameter instance.
          db : {'h5':<hdf5 file object>, 'group':<current group>}
          update_interval: how often database is updated from trace
        """
        self.obj = obj
        self.h5 = db['h5']
        self.group = db['group']
        
    def _initialize(self, length):
        """Initialize the trace."""
        self.h5.createArray(self.group, 
        self.__name__, zeros((length,)+shape(self.value), type(self.value)))
            
    def tally(self, index):
        """Adds current value to trace"""
        ar = getattr(self.group, self.__name__)
        ar.__setitem__(index, self.value)
        
    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).
        
        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains. 
          - slicing: A slice, overriding burn and thin assignement. 
        """
        gr = self.h5.listNodes("/")[chain]
        ar = getattr(gr, self.__name__)
        if slicing is not None:
            burn, stop, thin = slicing.start, slicing.stop, slicing.step
        else:
            stop = ar.shape[0] 
            
        return ar.read(burn, stop, step=thin)
        
    def _finalize(self):
        """Nothing done here."""
        pass
        
    __call__ = gettrace
    
def Database(object):
    """HDF5 database
    
    Create an HDF5 file <model>.h5. Each chain is stored in a group, and the 
    parameters and nodes are stored as arrays in each group.
    
    self.db is a dictionnary containing the (key,value) pairs
    'h5': the HDF5 file object,
    'group': the current group in the file.
    """
    
    def __init__(self, model):
        """Open file.""" 
        self.model = model
        
        name = self.__name__
        self.db = {}
        try:
            self.db['h5'] = tables.openFile(name+'.h5', 'a')
        except tables.exceptions.IOError:
            print "Database file seems already open. Skipping."    
            
    def _initialize(self):
        """Create group for the current chain."""
        i = len(self.db.listNodes('/'))
        self.db['group'] = self.db['h5'].createGroup("/", 'chain%d'%i, 'Chain #%d'%i)
        # Add attributes. Date. 
      
    def _finalize(self):
        """Close file."""
        # add attributes. Computation time. 
        self.db['h5'].close()
        
