__docformat__='reStructuredText'
###
# Empty template of database backend.
# Use this file as a blanck to write a new backend.  
###

class Trace(object):
    r"""Each Trace class contains five methods that need to be defined as 
    follows:
      - __init__(self, PyMCobject, database_instance)
      - _initialize(self, length)
      - tally(self, index)
      - gettrace(self, burn=0, thin=1, chain=-1, slicing=None)  
      - _finalize(self)
      
    This class is initialized when a Model is instantiated ; a Trace instance 
    is attributed to each traceable Deterministic and Stochastic:
    
    .. python::
        for object in self.stochs | self.dtrms :
            if object.trace:
                object.trace = module.Trace(object, self.db)
    
    At the start of the sampling run, `object.trace._initialize(length)` is 
    called, then for each iteration module thin after burn iterations, the 
    `tally` method is called. At the end of the sampling run, `_finalize` is
    called. The gettrace (or __call__) method is called by the user to get the 
    samples collected during the MCMC run.   
                      
    """
    def __init__(self, object, database):
        """This is called at Model instantiation time by method 
        `_assign_database_backend`.
        
        :Stochastics:
          - `object` : PyMC object (Deterministic or Stochastic) instance to be tallied.
          - `database` : The Model's Database instance. 
        """
        pass

    def _initialize(self, *args, **kwds):
        """Called by `database._initialize` when Model.sample 
        is called. Do whatever is needed before tallying may begin. """
        pass

    def tally(self, index):
        """Store current object's value."""
        pass

    def truncate(self, index):
        """
        When model receives a keyboard interrupt, it tells the traces
        to truncate their values.
        """
        pass

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace. 
        
        :Stochastics:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        pass

    def _finalize(self, *args, **kwds):
        """Do whatever is needed once tallying has ended."""
        pass

    __call__ = gettrace

class Database(object):
    """Each Database class contains three methods:
      - `__init__(self, **kwds)`
      - `_initialize(self, length, model)`
      - `_finalize(self)`
      
    """

    def __init__(self, **kwds):
        """This method is called in two different settings:
          - At Model instantiation, in `_assign_database_backend`
            .. python::
                self.db = module.Database()
            In this setting, no arguments can be passed to __init__, so defaults
            values must be set for each argument. 
            >>> M = Model(DisasterModel, db='txt')
            
          - By the user before a Model is instantiated. This usage allows users 
            to pass specific arguments to the Database.
            >>> D = database.mysql(dbuser='david', dbpass='hocuspocus')
            >>> M = Model(DisasterModel, db=D)
             
        """
        self.__dict__.update(kwds)

    def _initialize(self, length, model):
        """This method does whatever needs to be done to create a database. 
        One step is mandatory however, calling the Traces `_initialize` method. 
        """
        self.model = model
        for object in model._variables_to_tally:
            object.trace._initialize()

    def _finalize(self):
        """Do whatever is needed at the end of sampling run. 
        One step is mandatory: call the Traces `_finalize` method."""
        for object in model._variables_to_tally:
            object.trace._initialize()
