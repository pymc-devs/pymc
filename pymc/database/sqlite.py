"""
mysql.py

MySQL trace module

Created by Chris Fonnesbeck on 2007-02-01.
Updated by DH on 2007-04-04.
"""

from numpy import zeros, shape, squeeze, transpose
import sqlite3
import base, pickle, ram, pymc
import pdb

class Trace(object):
    """SQLite Trace class."""
    
    def __init__(self, obj=None):
        """Assign an initial value and an internal PyMC object."""       
        if obj is not None:
            if isinstance(obj, pymc.Variable):
                self._obj = obj
            else:
                raise AttributeError, 'Not PyMC object', obj
                
    def _initialize(self):
        """Initialize the trace. Create a table.
        """
        size = 1
        try:
            size = len(self._obj.value)
        except TypeError:
            pass
        
        self.current_trace = 1
        try:
            query = "create table %s (recid INTEGER NOT NULL PRIMARY KEY, trace int(5), %s FLOAT)" % (self._obj.__name__, ' FLOAT, '.join(['v%s' % (x+1) for x in range(size)]))
            self.db.cur.execute(query)
        except Exception:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self._obj.__name__)
            last_trace = self.db.cur.fetchall()[0][0]
            if last_trace: self.current_trace =  last_trace + 1

    def tally(self,index):
        """Adds current value to trace"""
        
        size = 1
        try:
            size = len(self._obj.value)
        except TypeError:
            pass
            
        try:
            value = self._obj.value.copy()
            valstring = ', '.join([str(x) for x in value])
        except AttributeError:
            value = self._obj.value
            valstring = str(value)  
            
        # Add value to database
        self.db.cur.execute("INSERT INTO %s (recid, trace, %s) values (NULL, %s, %s)" % (self._obj.__name__, ' ,'.join(['v%s' % (x+1) for x in range(size)]), self.current_trace, valstring))

    def gettrace(self, burn=0, thin=1, chain=None, slicing=None):
        """Return the trace (last by default).

        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if not slicing:
            slicing = slice(burn, None, thin)
            
        if not chain:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self._obj.__name__)
            chain = self.db.cur.fetchall()[0][0]
            
        self.db.cur.execute('SELECT * FROM %s WHERE trace=%s' % (self._obj.__name__, chain))
        trace = transpose(transpose(self.db.cur.fetchall())[2:])
            
        return squeeze(trace[slicing])

    def _finalize(self):
        pass

    __call__ = gettrace


    def length(self, chain=-1):
        """Return the sample length of given chain. If chain is None,
        return the total length of all chains."""
        return len(self.gettrace(chain=chain))

class Database(pickle.Database):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, filename=None):
        """Assign a name to the file the database will be saved in.
        """
        self.filename = filename
        self.Trace = Trace
    
    def _initialize(self,length):
        """Tell the traces to initialize themselves."""
        for o in self.model._variables_to_tally:
            o.trace._initialize()
        
    def connect(self, sampler):
        """Link the Database to the Sampler instance. 
        
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        base.Database.connect(self, sampler)
        self.choose_name('sqlite')
        self.DB = sqlite3.connect(self.filename)
        self.cur = self.DB.cursor()
    
    def commit(self):
        """Commit updates to database"""
        
        self.DB.commit()
                   
    def close(self, *args, **kwds):
        """Close database."""
        self.cur.close()
        self.DB.close()

# TODO: Code savestate and getstate to enable storing of the model's state.
# state is a dictionary with two keys: sampler and step_methods.
# state['sampler'] is another dictionary containing information about
# the sampler's state (_current_iter, _iter, _burn, etc.)
# state['step_methods'] is a dictionary with keys refering to ids for
# each samplingmethod defined in sampler. 
# Each id refers to another dictionary containing the state of the 
# step method. 
# To do this efficiently, we would need functions that stores and retrieves 
# a dictionary to and from a sqlite database. Unfortunately, I'm not familiar with 
# SQL enough to do that without having to read too much SQL documentation 
# for my taste. 

    def savestate(self, state):
        """Store a dictionnary containing the state of the Sampler and its 
        StepMethods."""
        pass
                
    def getstate(self):
        """Return a dictionary containing the state of the Sampler and its 
        StepMethods."""
        return {}

def load(filename):
    """Load an existing database.

    Return a Database instance.
    """
    db = Database(filename)
    return db
