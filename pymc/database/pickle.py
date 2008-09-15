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
1. Instantiation: db = PyMC.database.pickle.Database(**kwds)
2. Loading : db = PyMC.database.pickle.load(file)

Supporting 2 is a bit tricky, since it implies that we must :
a) restore the state of the database, using previously computed values,
b) restore the state of the Sampler.
which means that the database must also store the Sampler's state.
This is partially achieved.
"""


import ram, no_trace, base
import os, datetime, numpy
import string, cPickle

__all__ = ['Trace', 'Database', 'load']

class Trace(ram.Trace):
    pass

class Database(base.Database):
    """Pickle database backend.
    Saves the trace to a pickle file.
    """
    def __init__(self, filename=None):
        """Assign a name to the file the database will be saved in.
        """
        self.__name__ = 'pickle'
        self.filename = filename
        self.Trace = Trace

    def choose_name(self, extension=None):
        """If a file name has not been assigned, choose one from the 
        name of the input module imported by the Sampler."""
        if extension is not None:
            extension = '.'+extension
        else:
            extension = ''
        if self.filename is None:
            modname = self.model.__name__
            name = modname+extension
            i=0
            existing_names = os.listdir(".")
            while True:
                if name+extension in existing_names:
                    name = modname+'_%d'%i+extension
                    i += 1
                else:
                    break
            self.filename = name


    def connect(self, sampler):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        base.Database.connect(self, sampler)
        self.choose_name(extension='pickle')
        
    def close(self):
        """Dump traces using cPickle."""
        container={}
        try:
            for o in self.model._variables_to_tally:
                container[o.__name__] = o.trace._trace
            container['_state_'] = self._state_
        
            file = open(self.filename, 'w')
            cPickle.dump(container, file)
            file.close()
        except AttributeError:
            pass
        
     
def load(filename):
    """Load an existing database.

    Return a Database instance.
    """
    file = open(filename, 'r')
    container = cPickle.load(file)
    file.close()
    db = Database(file.name)
    for k,v in container.iteritems():
        if k == '_state_':
           db._state_ = v
        else:
            setattr(db, k, Trace(value=v))
    return db
        
    
