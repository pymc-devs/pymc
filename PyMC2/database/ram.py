###
# Basic trace module
# Simply store the trace in memory
###


from numpy import zeros,shape
import base

class Trace(base.Trace):
    """RAM trace 
    
    Store the samples in memory. 
    """
    def __init__(self, value=None):
        """Initialize the trace attributes.
        
        :Parameters:
          - `value` : Initial value.
        """
        if value is None:
            self._trace = []
        else:
            self._trace = value

    def _initialize(self, length):
        """Create an array of zeros with shape (length, shape(obj)).
        :Parameters:
          - `obj` : PyMC object
                Node or Parameter instance.
          - `length` : Length of array to initialize.
        """
        try:
            self._trace.append( zeros ((length,) + shape(self.obj.value), self.obj.value.dtype) )
        except AttributeError:
            self._trace.append( zeros ((length,) + shape(self.obj.value), dtype=object) )           
        
    def tally(self, index):
        """Store current value of object."""
        try:
            self._trace[-1][index] = self.obj.value.copy()
        except AttributeError:
            self._trace[-1][index] = self.obj.value

    def truncate(self, index):
        """
        When model receives a keyboard interrupt, it tells the traces
        to truncate their values.
        """
        self._trace[-1] = self._trace[-1][:index]

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

    __call__ = gettrace

class Database(base.Database):
    """RAM database."""
    def __init__(self):
        self.Trace = Trace
    
