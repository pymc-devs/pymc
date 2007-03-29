"""
Base backend
"""
import PyMC2

class Trace(object):
    """Dummy Trace class.
    """ 
    
    def __init__(self,value=None):
        """Initialize the instance.
        """
        self._trace = value
        
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
    
##    def obj():
##        def fset(self, obj):
##            if isinstance(obj, PyMC2.PyMCBase):
##               self.__obj = obj
##            else:
##                raise AttributeError, 'Not PyMC object'
##        def fget(self):
##            return self.__obj
##        return locals()
##    obj = property(**obj())
    
    def _finalize(self):
        pass
    
class Database(object):
    """Dummy Database backend"""
    def __init__(self):
        self.Trace = Trace
        
    def _initialize(self, length):
        """Initialize database."""
        for obj in self.model._pymc_objects_to_tally:
            obj.trace._initialize(length)
        
    def connect(self, model):
        self.model = model
        
        if hasattr(self, '_state_'): # Restore the state of the Sampler.
            for obj in model._pymc_objects_to_tally:
                obj.trace = getattr(self, obj.__name__)
        else: # Set a fresh new state
            for obj in model._pymc_objects_to_tally:
                obj.trace = self.Trace()
            
        for ob in model._pymc_objects_to_tally:
            ob.trace.obj = ob
    
    def _finalize(self):
        """Close database."""
        for object in self.model._pymc_objects_to_tally:
            object.trace._finalize()
    
    def model():
        def fset(self, m):
            if isinstance(m, PyMC2.Model):
                self.__model = m
            else:
                raise AttributeError, 'Not a Model instance.'
        def fget(self):
            return self.__model
        return locals()
    model = property(**model())
