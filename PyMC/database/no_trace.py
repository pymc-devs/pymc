###
# Basic trace module
# Simply store the trace in memory
###

###
# To use another database backend, make a copy of this file in the database 
# folder, modify the functions as desired, and rename it, eg. sqllite.py. 
# When instantiating a Model, pass the dbase = dbase_filename argument,
# eg. dbase = 'sqllite'.
###

from numpy import zeros,shape

class trace(object):
    """ Define the methods that will be assigned to each parameter in the 
    Model instance."""
    
    def _init_trace(self, pymc_object, length):
        """Initialize the trace."""
        self.pymc_object = pymc_object
            
    def tally(self, index):
        """Adds current value to trace"""
        pass

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """
        Return the trace (last by default).
        
        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains. 
          - slicing: A slice, overriding burn and thin assignement. 
        """
        raise AttributeError, self.pymc_object.__name__ + " has no trace"

    def _finalize_trace(self):
        pass
    
def model_methods():
    """Define the methods that will be assigned to the Model class"""
    def _init_dbase(self, *args, **kwds):
        """Initialize database."""
        pass
    def _finalize_dbase(self, *args, **kwds):
        """Close database."""
        pass
        
    return locals()
