###
# Txt trace module
# Trace are stored in memory during sampling and saved to a
# txt file at the end of sampling. Each object has its own file. 
###

import memory_trace 
import os, io, datetime

class trace(memory_trace.trace):
    def _finalize(self, burn, thin):
        """Dump trace into txt file in the simulation folder _dbdir."""
        path = os.path.join(self.db.dir, self.name+'.txt')
        arr = self.gettrace(burn, thin)
        f = open(path, 'w')
        print >> f, '# Parameter %s' % self.name
        print >> f, '# Burned: %d, thinned= %d' % (burn, thin)
        print >> f, '# Sample shape: %s' % str(arr.shape)
        print >> f, '# Date: %s' % datetime.datetime.now()
        f.close()
        io.aput(arr, path, 'a')
    

class database(object):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, model):
        self.model = model
    
    def _initialize(self, *args, **kwds):
        """Create folder to store simulation results."""
        name = self.__name__
        i=0;again=True
        while again:
            try:
                os.mkdir(name)
                again = False
            except OSError:
                name = self.__name__+'_%d'%i
                i += 1
        self.dir = name
        
    def _finalize(self, burn, thin):
        """Dump samples to file."""
        for object in self.model._pymc_objects_to_tally:
            object._finalize_trace(burn,thin)
    
