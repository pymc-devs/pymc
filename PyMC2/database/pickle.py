###
# Pickle backend module
# Trace are stored in memory during sampling and saved to a
# pickle file at the end of sampling.
###
"""
The object passed as db to Sampler may be:
1. A string (the database module name),
2. A Database instance,

There could be two ways to initialize a Database instance:
1. Instantiation: db = PyMC2.database.pickle.Database(**kwds)
2. Loading : db = PyMC2.database.pickle.load(file)

Supporting 2 is a bit tricky, since it implies that we must :
a) restore the state of the database, using previously computed values,
b) restore the state of the Sampler.
which means that the database must also store the Sampler's state.
"""


import ram, no_trace
import os, datetime, numpy
import string, cPickle

class Trace(ram.Trace):
    pass

class Database(no_trace.Database):
    """Pickle database backend.
    Saves the trace to a pickle file.
    """
    Trace = Trace
    def __init__(self, filename=None):
        """
        Return a Pickle database instance.

        :Parameters:
            - `filename` : Name of file where the results are stored.
        """
        self.filename = filename
        self.Trace = Trace

    def _initialize(self, length):
        """Define filename to store simulation results."""

        if self.filename is None:
            modname = self.model.__name__.split('.')[-1]
            name = modname+'.pickle'
            i=0
            existing_names = os.listdir(".")
            while True:
                if name+'.pickle' in existing_names:
                    name = modname+'_%d'%i+'.pickle'
                    i += 1
                else:
                    break
            self.filename = name

        for object in self.model._pymc_objects_to_tally:
            object.trace._initialize(length)


    def _finalize(self):
        """Dump traces using cPickle."""
        container={}
        for obj in self.model._pymc_objects_to_tally:
            obj.trace._finalize()
            container[obj.__name__] = obj.trace._trace
        container['_state_'] = self._state_
        file = open(self.filename, 'w')
        cPickle.dump(container, file)
        file.close()

##    def state():
##        def fset(self, s):
##            self.__state = s
##        def fget(self):
##            return self.__state
##        return locals()
##    state = property(**state())    
    
def load(file):
    """Load an existing database.

    Return a Database instance.
    """
    container = cPickle.load(file)
    file.close()
    db = Database(file.name)
    for k,v in container.iteritems():
        if k == '_state_':
           db._state_ = v
        else:
            setattr(db, k, Trace(v))
    return db
        
    
