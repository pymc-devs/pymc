###
# Txt trace module
# Trace are stored in memory during sampling and saved to a
# txt file at the end of sampling. Each object has its own file. 
###
# TODO: Implement state dunp and recovery.

# Changeset
# Nov. 30, 2007: Implemented load function. DH

import base, ram, no_trace, pickle
import os, datetime, shutil, re
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
        print >> f, '# Description: %s' % self._obj.__doc__.replace('\n', '\n# ')
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
    
    def connect(self, model):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        base.Database.connect(self, model)
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
    
    #---------------------------------------
def readArray(filename, skipchar = '#'):
    #---------------------------------------
    # From the NumPy Wiki
    """
    PURPOSE: read an array from a file. Skip empty lines or lines
             starting with the comment delimiter (defaulted to '#').
    
    OUTPUT: a float numpy array
    
    EXAMPLE: >>> data = readArray("/home/albert/einstein.dat")
             >>> x = data[:,0]        # first column
             >>> y = data[:,1]        # second column
    """
    
    myfile = open(filename, "r")
    contents = myfile.readlines()
    myfile.close()
    
    data = []
    for line in contents:
        stripped_line = string.lstrip(line)
        if (len(stripped_line) != 0):
          if (stripped_line[0] != skipchar):
            items = string.split(stripped_line)
            data.append(map(float, items))
    
    a = numpy.array(data)
    (Nrow,Ncol) = a.shape
    if ((Nrow == 1) or (Ncol == 1)): a = numpy.ravel(a)
    return(a)

  
def load(dirname, mode='a'):
    """Create a Database instance from the data stored in the directory."""
    db = Database(dirname, mode)
    chains = [os.path.join(dirname, c) for c in db.get_chains()]
    files = os.listdir(os.path.join(chains[0]))
    varnames = varname(files)
    data = dict([(v, []) for v in varnames])
    for c in chains:
        files = os.listdir(os.path.join(c))
        for f in files:
            file = os.path.join(c, f)
            a = readArray(file)
            data[varname(f)].append(a)
    for k in varnames:
        setattr(db, k, Trace(data[k]))
        o = getattr(db,k)
        setattr(o, 'db', db)
    db._state_ = {}
    return db
    
def varname(file):
    """Return variable names from file names."""
    if type(file) is str:
        files = [file]
    else:
        files = file
    bases = [os.path.basename(f) for f in files]
    names = [os.path.splitext(b)[0] for b in bases]
    if type(file) is str:
        return names[0]
    else:
        return names
    
