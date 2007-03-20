###
# Pickle backend module
# Trace are stored in memory during sampling and saved to a
# pickle file at the end of sampling.  
###
"""What I want to do here is allow the possibility to load an existing pickle file to continue sampling where it was left.
To do so, we first initialize the database with the name of the pickled file. Since we cannot pickle the nodes and parameters, the model definition, we cannot affect the stored traces to the objects right away. We must wait until initialization, when the model instance is passed, to fill the traces. This means that _initialize must be called in model if it hasn't been.
model.state = dict(_current_iter, _iter, _burn, _thin, sampling, ready, tuning parameters.)
sampling is True during sampling...
ready is True if it is possible to sample without initializing the database (GUI stopped computation and wants to restart it).
ready is False if to sample we must initialize the database. (Sampling was interrupted, the database finalized and the session closed.)
At the end of a sampling loop, ready is set to false.

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
        """This is a proof of concept. The idea is to allow the __init__ method
        to load an existing database and make it current.
        """
        if filename in os.listdir('.'):
            file = open(filename, 'r')
            self.container = cPickle.load(file)
            file.close()
            self.reloading = True
        else:
            self.reloading=False
        self.filename = filename
    
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
        self.reloading = False
        
    def _finalize(self):
        """Dump traces using cPickle."""
        container={}
        for object in self.model._pymc_objects_to_tally:
            object.trace._finalize()
            container[object.__name__] = object.trace.gettrace()
        file = open(self.filename, 'w')
        cPickle.dump(container, file)
        file.close()
        
