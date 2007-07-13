###
# Basic trace module
# Simply store the trace in memory
###

from numpy import zeros,shape,concatenate
import pdb

class Trace(list):
    """
    RAM trace 
    
    Store the samples in memory. 
    """
    def __init__(self, name):
        """Assign an initial value and an internal PyMC object."""
        
        list.__init__(self, [])
        self.__name__ = name

    def _initialize(self, length, value):
        """Create an array of zeros with shape = length
        """
        
        try:
            data_type = value.dtype
        except AttributeError:
            data_type = type(value)
        
        self.append( zeros((length,) + shape(value), dtype=data_type) )          
        
    def tally(self, value, index, chain=-1):
        """Store current value."""
        
        self[chain][index] = value

    def get_trace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).

        :Parameters:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if not slicing:
            slicing = slice(burn, None, thin)
        if chain:
            return self[chain][slicing]
        else:
            return concatenate(self)[slicing]
            
    def finalize(self): pass

    __call__ = get_trace
    
    
class Database(object):
    """Dummy Database backend"""
    
    def __init__(self):
        """Get the Trace from the local scope."""
        self.Trace = Trace
        
    def connect(self, *args, **kwargs):
        pass
        
    def close(self, *args, **kwargs):
        pass