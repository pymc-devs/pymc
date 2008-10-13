"""
mysql.py

MySQL trace module

Created by Chris Fonnesbeck on 2007-02-01.
Updated by DH on 2007-04-04.
"""
import numpy as np
from numpy import zeros, shape, squeeze, transpose
import sqlite3
import base, pickle, ram, pymc
import pdb

__all__ = ['Trace', 'Database', 'load']

class Trace(object):
    """SQLite Trace class."""
    
    def __init__(self, obj=None, name=None):
        """Assign an initial value and an internal PyMC object."""       
        if obj is not None:
            if isinstance(obj, pymc.Variable):
                self._obj = obj
                self.name = self._obj.__name__
            else:
                raise AttributeError, 'Not PyMC object', obj
        else:
            self.name = name
            
    def _initialize(self):
        """Initialize the trace. Create a table.
        """
        size = 1
        try:
            size = len(self._obj.value)
        except TypeError:
            pass
        
        # Try to create a new table. If a table of same name exists, simply 
        # update self.last_trace.
        
        try:
            query = "create table %s (recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, trace  int(5), %s FLOAT)" % (self.name, ' FLOAT, '.join(['v%s' % (x+1) for x in range(size)]))
            self.db.cur.execute(query)
            self.current_trace = 1
        except Exception:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.name)
            last_trace = self.db.cur.fetchall()[0][0]
            self.current_trace =  last_trace + 1
        
    def tally(self,index):
        """Adds current value to trace"""
        
        try:
            value = self._obj.value[self._obj._missing]
        except (AttributeError, TypeError):
            value = self._obj.value
        
        size = 1
        try:
            size = len(value)
        except TypeError:
            pass
            
        try:
            # I changed str(x) to '%f'%x to solve a bug appearing due to
            # locale settings. In french for instance, str prints a comma
            # instead of a colon to indicate the decimal, which confuses
            # the database into thinking that there are more values than there
            # is.  A better solution would be to use another delimiter than the 
            # comma. -DH
            valstring = ', '.join(['%f'%x for x in value])
        except:
            valstring = str(value)  
            
        # Add value to database
        self.db.cur.execute("INSERT INTO %s (recid, trace, %s) values (NULL, %s, %s)" % (self.name, ' ,'.join(['v%s' % (x+1) for x in range(size)]), self.current_trace, valstring))
        
    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).

        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all 
            chains. By default, the last chain is returned.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if not slicing:
            slicing = slice(burn, None, thin)
   
        # Find the number of traces.
        self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.name)
        n_traces = self.db.cur.fetchall()[0][0]
        
        # If chain is None, get the data from all chains.
        if chain is None: 
            self.db.cur.execute('SELECT * FROM %s' % self.name)
            trace = self.db.cur.fetchall()
        else:
            # Deal with negative chains (starting from the end)
            if chain < 0:
                chain = n_traces+chain+1
            self.db.cur.execute('SELECT * FROM %s WHERE trace=%s' % (self.name, chain))
            trace = self.db.cur.fetchall()
         
        trace = np.array(trace)[:,2:]
        return squeeze(trace[slicing])

    def _finalize(self):
        pass
        #self.commit()

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
        self.__name__ = 'sqlite'
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
        self.DB = sqlite3.connect(self.filename, check_same_thread=False)
        self.cur = self.DB.cursor()
    
    def commit(self):
        """Commit updates to database"""
        
        self.DB.commit()
                   
    def close(self, *args, **kwds):
        """Close database."""
        self.cur.close()
        self.commit()
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
    db.DB = sqlite3.connect(db.filename)
    db.cur = db.DB.cursor()
    
    # Get the name of the objects
    tables = get_table_list(db.cur)
    
    # Create a Trace instance for each object
    for name in tables:
        setattr(db, name, Trace(name=name))
        o = getattr(db, name)
        setattr(o, 'db', db)
    return db

# Copied form Django.
def get_table_list(cursor): 
    """Returns a list of table names in the current database."""
    # Skip the sqlite_sequence system table used for autoincrement key
    # generation.
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND NOT name='sqlite_sequence'
        ORDER BY name""")
    return [row[0] for row in cursor.fetchall()]
