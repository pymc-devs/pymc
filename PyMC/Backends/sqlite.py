"""
sqlite.py

SQLite trace module

Created by Chris Fonnesbeck on 2007-02-01.
"""

from numpy import zeros, shape, squeeze, transpose
import sqlite3
import pdb

class Trace(list):
    """SQLite Trace class."""
    
    def __init__(self, name):
        """Assign an initial value and an internal PyMC object.""" 
        
        list.__init__(self, [])
        self.__name__ = name
        
    def _initialize(self, length, value):
        """Initialize the trace. Create a table.
        """
        size = 1
        try:
            size = len(value)
        except TypeError:
            pass
           
        try: 
            if value.dtype.name.startswith('int'):
                data_type = 'INTEGER'
            else:
                data_type = 'REAL'
        except AttributeError:
            if type(value) == type(0):
                data_type = 'INTEGER'
            else:
                data_type = 'REAL'
        
        self.current_trace = 1
        
        try:
            query = "create table %s (recid INTEGER NOT NULL PRIMARY KEY, trace INTEGER,%s)" % (self.__name__, ''.join([' v%s %s,' % (x+1, data_type) for x in range(size)])[:-1])
            
            self.db.cur.execute(query)
        except Exception:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.__name__)
            last_trace = self.db.cur.fetchall()[0][0]
            if last_trace: 
                self.current_trace = last_trace + 1

    def tally(self, value, index):
        """Adds current value to trace"""
        
        self.append(value)
        
        if not index % 100:
            
            self._write_to_db()
            self.__init__(self.__name__)
        
    def _write_to_db(self):
        
        size = 1
        try:
            size = len(self[0])
        except TypeError:
            pass
        
        for value in self:    
            try:
                valstring = ', '.join([str(x) for x in value])
            except TypeError:
                valstring = str(value)  
            
            # Add value to database
            self.db.cur.execute("INSERT INTO %s (recid, trace, %s) values (NULL, %s, %s)" % (self.__name__, ' ,'.join(['v%s' % (x+1) for x in range(size)]), self.current_trace, valstring))
            
        self.db.con.commit()

    def get_trace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace (last by default).

        Input:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        if not slicing:
            slicing = slice(burn, None, thin)
            
        if chain == -1:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.__name__)
            chain = self.db.cur.fetchall()[0][0]
            
        self.db.cur.execute('SELECT * FROM %s WHERE trace=%s' % (self.__name__, chain))
        trace = transpose(transpose(self.db.cur.fetchall())[2:])
            
        return squeeze(trace[slicing])

    def finalize(self):
        
        if self.__len__(): self._write_to_db()

    __call__ = get_trace


class Database:
    """Define the methods that will be assigned to the Model class"""
    
    def __init__(self, filename=None):
        """Assign a name to the file the database will be saved in.
        """
        
        self.filename = filename
        self.Trace = Trace
        self.Trace.db = self
        
    def connect(self):
        """
        If database is loaded from a file, restore the objects trace 
        to their stored value, if a new database is created, instantiate
        a Trace for the PyMC objects to tally.
        """
        
        self.con = sqlite3.connect(self.filename)
        self.cur = self.con.cursor()
                   
    def close(self, *args, **kwds):
        """Close database."""
        
        self.cur.close()
        self.con.close()
        

def load(filename):
    """Load an existing database.

    Return a Database instance.
    """
    db = Database(filename)
    return db
