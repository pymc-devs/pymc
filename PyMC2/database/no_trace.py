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
    def __init__(self, pymc_object):
        """Initialize the instance.
        :Parameters:
          obj : PyMC object
            Node or Parameter instance.
          db : database instance
        """
        self.obj = pymc_object
        self._trace = []
        
    def _initialize(self, length):
        """Initialize the trace."""
        pass
            
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
        raise AttributeError, self.obj.__name__ + " has no trace"

    def _finalize(self):
        pass
    
class database(object):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, model):
        self.model = model
        
    def _initialize(self, *args, **kwds):
        """Initialize database."""
        pass
    def _finalize(self, *args, **kwds):
        """Close database."""
        pass
