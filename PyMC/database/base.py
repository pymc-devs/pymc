"""
Base backend

Trace and Database classes from the other modules should Subclass the base 
classes. 
"""
import PyMC

class Trace(object):
    """Dummy Trace class.
    """ 
    
    def __init__(self,value=None, obj=None):
        """Assign an initial value and an internal PyMC object."""
        self._trace = value
        if obj is not None:
            if isinstance(obj, PyMC.Variable):
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
    
   
    def _finalize(self):
        pass
    
    def length(self, chain=-1):
        """Return the sample length of given chain. If chain is None,
        return the total length of all chains."""
        pass

class Database(object):
    """Dummy Database backend"""
    def __init__(self):
        """Get the Trace from the local scope."""
        self.Trace = Trace
        
    def _initialize(self, length):
        """Tell the traces to initialize themselves."""
        for o in self.model._variables_to_tally:
            o.trace._initialize(length)
        
    def tally(self, index):
        """Dummy method. Subclass if necessary."""
        for o in self.model._variables_to_tally:
            o.trace.tally(index)
            
    def connect(self, model):
        """Link the Database to the Model instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        # Changed this to allow non-Model models. -AP
        if isinstance(model, PyMC.Model):
            self.model = model
        else:
            raise AttributeError, 'Not a Model instance.'
                
        if hasattr(self, '_state_'): 
            # Restore the state of the Model.
            for o in model._variables_to_tally:
                o.trace = getattr(self, o.__name__)
                o.trace._obj = o
        else: 
            # Set a fresh new state
            for o in model._variables_to_tally:
                o.trace = self.Trace(obj=o)
        
        for o in model._variables_to_tally:
            o.trace.db = self
    
    def _finalize(self):
        """Tell the traces to finalize themselves."""
        for o in self.model._variables_to_tally:
            o.trace._finalize()
    
    def close(self):
        """Close the database."""
        pass
        
    def savestate(self, state):
        """Store a dictionnary containing the state of the Model and its 
        StepMethods."""
        self._state_ = state
        
    def getstate(self):
        """Return a dictionary containing the state of the Model and its 
        StepMethods."""
        return getattr(self, '_state_', {})
        
