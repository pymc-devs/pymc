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
a) restore the state of the database, susing previously computed values,
b) restore the state of the Sampler.
which means that the database must also store the Sampler's state.
"""


import ram, no_trace
import os, datetime, numpy
import string, cPickle

class Trace(ram.Trace):
    pass

class Database(object):
    """Pickle database backend.
    Saves the trace to a pickle file.
    """
    def __init__(self, filename=None):
        """
        Return a Pickle database instance.

        :Parameters:
            - `filename` : Name of file where the results are stored.
        """
        self.filename = filename
        self.state = {}

    def _initialize(self, length, model):
        """Define filename to store simulation results."""
        self.model = model

        if self.filename is None:
            modname = self.model.__name__.split('.')[-1]
            name = modname
            i=0
            existing_names = os.listdir(".")
            while True:
                if name+'.pymc' in existing_names:
                    name = modname+'_%d'%i+'.pymc'
                    i += 1
                else:
                    break
            self.filename = name

        for object in self.model._pymc_objects_to_tally:
            if self.reloading:
                object._trace = self.container[object.__name__]
            else:
                object.trace._initialize(length)


    def _finalize(self):
        """Dump traces using cPickle."""
        container={}
        for object in self.model._pymc_objects_to_tally:
            object.trace._finalize()
            container[object.__name__] = object.trace.gettrace()
        container['state'] = self.state
        file = open(self.filename, 'w')
        cPickle.dump(container, file)
        file.close()

    def save_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

def load(file):
    """Load an existing database.

    Return a Database instance.
    """
    container = cPickle.load(f)
    f.close()
    db = Database(f.name)

    # Now traces must be reset to their previous state.
