"""
Base backend

Trace and Database classes from the other modules should Subclass the base 
classes. 
"""
import PyMC2

class Trace(object):
    """Dummy Trace class.
    """ 
    
    def __init__(self,value=None, obj=None):
        """Assign an initial value and an internal PyMC object."""
        self._trace = value
        if obj is not None:
            if isinstance(obj, PyMC2.PyMCBase):
                self._obj = obj
            else:
                raise AttributeError, 'Not PyMC object', obj
        
    def _initialize(self, length):
        """Dummy method. Subclass if necessary."""
        pass
            
    def tally(self, index):
        """Dummy method. Subclass if necessary."""
        pass

    def truncate(self, index):
        """Dummy method. Subclass if necessary."""
        pass

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Dummy method. Subclass if necessary.
        
        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains. 
          - slicing: A slice, overriding burn and thin assignement. 
        """
        raise AttributeError, self._obj.__name__ + " has no trace"

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
        """Get the Trace from the local scope."""
        self.Trace = Trace
        
    def _initialize(self, length):
        """Tell the traces to initialize themselves."""
        for o in self.model._pymc_objects_to_tally:
            o.trace._initialize(length)
        
    def connect(self, sampler):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the PyMC objects to tally.
        """
        if isinstance(sampler, PyMC2.Sampler):
            self.model = sampler
        else:
            raise AttributeError, 'Not a Sampler instance.'
                
        if hasattr(self, '_state_'): 
            # Restore the state of the Sampler.
            for o in sampler._pymc_objects_to_tally:
                o.trace = getattr(self, o.__name__)
                o.trace._obj = o
        else: 
            # Set a fresh new state
            for o in sampler._pymc_objects_to_tally:
                o.trace = self.Trace(obj=o)
    
    def _finalize(self):
        """Tell the traces to finalize themselves."""
        for o in self.model._pymc_objects_to_tally:
            o.trace._finalize()
    
