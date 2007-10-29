###
# Txt trace module
# Trace are stored in memory during sampling and saved to a
# txt file at the end of sampling. Each object has its own file. 
###
# TODO: Implement load function.

import base, ram, no_trace, pickle
import os, datetime, shutil
import numpy
from numpy import atleast_2d
import string 
CHAIN_NAME = 'Chain_%d'

class Trace(ram.Trace):
    def _finalize(self):
        """Dump trace into txt file in the simulation folder _dbdir."""
        path = os.path.join(self.db._cur_dir, self._obj.__name__+'.txt')
        arr = self.gettrace()
        f = open(path, 'w')
        print >> f, '# Variable: %s' % self._obj.__name__
        print >> f, '# Description: %s' % self._obj.__doc__
        print >> f, '# Burned: %d, thinned= %d' % \
            (self.db.model._burn, self.db.model._thin)
        print >> f, '# Sample shape: %s' % str(arr.shape)
        print >> f, '# Date: %s' % datetime.datetime.now()
        f.close()
        aput(arr, path, 'a')
        

class Database(pickle.Database):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, dirname=None, mode='w'):
        self.filename = dirname
        self.Trace = Trace
        self.mode = mode
    
    def connect(self, sampler):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        base.Database.connect(self, sampler)
        self.choose_name()
        self.makedir()
               
    def makedir(self):
        if self.filename in os.listdir('.'):
            if self.mode == 'w':
                shutil.rmtree(self.filename)
                os.mkdir(self.filename)
        else:
            os.mkdir(self.filename)
            
    def get_chains(self):
        chains = []
        try:
            content = os.listdir(self.filename)
            for c in content:
                if os.path.isdir(os.path.join(self.filename, c)) and c.startswith(CHAIN_NAME[:-2]):
                    chains.append(c)
        except:
            pass
        chains.sort()
        return chains
            
    def chain_name(self, chain=-1):
        """Return the name of the directory corresponding to the chain."""
        return self.get_chains()[chain]
    
    def new_chain_name(self):
        """Return the name of the directory for the new chain."""
        n = len(self.get_chains())+1
        return CHAIN_NAME%n
        
        
    def _initialize(self, length):
        """Create folder to store simulation results."""
        self._cur_dir = os.path.join(self.filename, self.new_chain_name())
        os.mkdir(self._cur_dir)
        
        for object in self.model._variables_to_tally:
            object.trace._initialize(length)
            
    def close(self):
        pass
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
    
def load(dirname):
    """Create a Database instance from the data stored in the directory."""
    pass 
