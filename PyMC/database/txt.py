###
# Txt trace module
# Trace are stored in memory during sampling and saved to a
# txt file at the end of sampling. Each object has its own file. 
###

from memory_trace import parameter_methods
import os, io, datetime

parameter_methods = parameter_methods()

def _finalize_trace(self, burn, thin):
    """Dump trace into txt file in the simulation folder _dbdir."""
    path = os.path.join(self._model._dbdir, self.name)
    arr = self.trace(burn, thin)
    f = open(path, 'w')
    print >> f, '# Parameter %s' % self.name
    print >> f, '# Burned: %d, thinned= %d' % (burn, thin)
    print >> f, '# Sample shape: %s' % str(arr.shape)
    print >> f, '# Date: %s' % datetime.datetime.now()
    f.close()
    io.aput(arr, path, 'a')
    

def model_methods():
    """Define the methods that will be assigned to the Model class"""
    def _init_dbase(self, *args, **kwds):
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
        self._dbdir = name
        
    def _finalize_dbase(self, burn, thin):
        """Dump samples to file."""
        for object in self._pymc_objects_to_tally:
            object._finalize_trace(burn,thin)
        
    return locals()
