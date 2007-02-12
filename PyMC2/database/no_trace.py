###
# no_trace database backend
# No memory whatsoever of the samples.
###

from numpy import zeros,shape

class Trace(object):
    """The no-trace backend provides a minimalistic backend where absolutely no
    trace of the values sampled is kept. This may be useful for testing 
    purposes.
    """ 
    
    def __init__(self, pymc_object, db):
        """Initialize the instance.
        :Parameters:
          obj : PyMC object
            Node or Parameter instance.
          db : database instance
        """
        self.obj = pymc_object
        self.db = db
        self._trace = []
        
    def _initialize(self, length):
        """Initialize the trace."""
        pass
            
    def tally(self, index):
        """Dummy method. This does abolutely nothing."""
        pass

    def truncate(self, index):
        """
        When model receives a keyboard interrupt, it tells the traces
        to truncate their values.
        """
        pass

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """
        This doesn't return anything.
        
        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains. 
          - slicing: A slice, overriding burn and thin assignement. 
        """
        raise AttributeError, self.obj.__name__ + " has no trace"

    __call__ = gettrace
    
    def _finalize(self):
        pass
    
class Database(object):
    """The no-trace database is empty."""
    def __init__(self):
        pass
        
    def _initialize(self, length, model):
        """Initialize database."""
        self.model = model
        for object in self.model._pymc_objects_to_tally:
            object.trace._initialize(length)
        
    def _finalize(self):
        """Close database."""
        for object in self.model._pymc_objects_to_tally:
            object.trace._finalize()
