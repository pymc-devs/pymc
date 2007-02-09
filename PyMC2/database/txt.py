###
# Txt trace module
# Trace are stored in memory during sampling and saved to a
# txt file at the end of sampling. Each object has its own file. 
###

import ram
import os, datetime, numpy
from numpy import atleast_2d
import string 

class Trace(ram.Trace):
    def _finalize(self, burn, thin):
        """Dump trace into txt file in the simulation folder _dbdir."""
        path = os.path.join(self.db.dir, self.obj.__name__+'.txt')
        arr = self.gettrace(burn, thin)
        f = open(path, 'w')
        print >> f, '# Parameter %s' % self.obj.__name__
        print >> f, '# Burned: %d, thinned= %d' % (burn, thin)
        print >> f, '# Sample shape: %s' % str(arr.shape)
        print >> f, '# Date: %s' % datetime.datetime.now()
        f.close()
        print 'store stuff in file:', arr.shape
        aput(arr, path, 'a')
        print 'end'

class Database(object):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, model):
        self.model = model
    
    def _initialize(self, *args, **kwds):
        """Create folder to store simulation results."""
        modname = self.model.__name__.split('.')[-1]
        name = modname
        i=0;again=True
        while again:
            try:
                os.mkdir(name)
                again = False
            except OSError:
                name = modname+'_%d'%i
                i += 1
        self.dir = name
        
    def _finalize(self, burn, thin):
        """Dump samples to file."""
        for object in self.model._pymc_objects_to_tally:
            object.trace._finalize(burn,thin)
    
def aput (outarray,fname,writetype='w',delimit=' '):
    """Sends passed 1D or 2D array to an output file and closes the file.
    """
    outfile = open(fname,writetype)
    if outarray.ndim==1:
        outarray = atleast_2d(outarray).transpose()
    
    if outarray.ndim > 2:
        raise TypeError, "aput() require 1D or 2D arrays.  Otherwise use some kind of pickling."
    else: # must be a 2D array
        for row in outarray:
            outfile.write(string.join(map(str,row),delimit))
            outfile.write('\n')
    outfile.close()
    return None
