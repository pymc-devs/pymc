###
# Basic trace module
# Simply store the trace in memory
###


from numpy import zeros,shape

class Trace(object):
    """RAM trace 
    
    Store the samples in memory. 
    """
    def __init__(self, obj, db):
        """Initialize the trace attributes.
        
        :Parameters:
          obj : PyMC object
            Node or Parameter instance.
          db : database instance
            Reference to the database object.
        """
        self.obj = obj
        self.db = db
        self._trace = []

    def _initialize(self, length):
        """Create an array of zeros with shape (length, shape(obj)).
        """
        self._trace.append( zeros ((length,) + shape(self.obj.value), type(self.obj.value)) )

    def tally(self, index):
        """Put current value in trace."""
        try:
            self._trace[-1][index] = self.obj.value.copy()
        except AttributeError:
            self._trace[-1][index] = self.obj.value

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).

        :Parameters:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if slicing is None:
            slicing = slice(burn, None, thin)
        return self._trace[chain][slicing]

    def _finalize(self, *args, **kwds):
        """Nothing done here."""
        pass

    __call__ = gettrace

class Database(object):
    """Memory database. Nothing done here."""
    def __init__(self, model):
        self.model = model
        
    def _initialize(self, *args, **kwds):
        """Initialize database. Nothing to do."""
        pass
    def _finalize(self, *args, **kwds):
        """Close database. Nothing to do."""
        pass
