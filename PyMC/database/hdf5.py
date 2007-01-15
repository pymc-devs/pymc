###
# HDF5 database module
# Store the traces in an HDF5 array using pytables.
# Dependencies
# pytables: <http://sourceforge.net/projects/pytables/>
# HDF5 >= 1.6.5 
# Numarray >= 1.5.2 (eventually will rely on numpy)
###

###
# To use another database backend, make a copy of this file in the database 
# folder, modify the functions as desired, and rename it, eg. sqllite.py. 
# When instantiating a Model, pass the dbase = dbase_filename argument,
# eg. dbase = 'sqllite'.
###

from numpy import zeros,shape
import tables

def parameter_methods():
    """ Define the methods that will be assigned to each parameter in the 
    Model instance."""
    
    def _init_trace(self, length):
        """Initialize the trace."""
        self._model._dbfile.createArray(self._model._h5_group, 
        self.__name__, zeros((length,)+shape(self.value), type(self.value)))
            
    def tally(self, index):
        """Adds current value to trace"""
        ar = getattr(self._model._h5_group, self.__name__)
        ar.__setitem__(index, self.value)
        
    def trace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).
        
        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains. 
          - slicing: A slice, overriding burn and thin assignement. 
        """
        gr = self._model._dbfile.listNodes("/")[chain]
        ar = getattr(gr, self.__name__)
        if slicing is not None:
            burn, stop, thin = slicing.start, slicing.stop, slicing.step
        else:
            stop = ar.shape[0] 
            
        return ar.read(burn, stop, step=thin)
        
    def _finalize_trace(self):
        """We could export the trace to a txt file."""
        pass
        
    return locals()
    
def model_methods():
    """Define the methods that will be assigned to the Model class"""
    def _init_dbase(self):
        """Initialize database."""
        name = self.__name__
        try:
            h5 = tables.openFile(name+'.h5', 'a')
        except tables.exceptions.IOError:
            print "Database file seems already open. Skipping."
        self._dbfile = h5
        # For each chain create a new group
        i = len(h5.listNodes('/'))
        self._h5_group = h5.createGroup("/", 'chain%d'%i, 'Chain #%d'%i)
        # Add attributes. Date. 
      
    def _finalize_dbase(self):
        """Close database."""
        # add attributes. Computation time. 
        self._dbfile.close()
        
    return locals()
