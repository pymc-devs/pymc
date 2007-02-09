"""
mysql.py

MySQL trace module

Created by Chris Fonnesbeck on 2007-02-01.
"""

from numpy import zeros, shape, squeeze, transpose
from pysqlite2 import dbapi2 as sql

class Trace(object):
    """ Define the methods that will be assigned to each parameter in the
    Model instance."""
    def __init__(self, obj, db):
        """Initialize the instance.
        :Parameters:
          obj : PyMC object
            Node or Parameter instance.
          db : database instance
          update_interval: how often database is updated from trace
        """
        self.obj = obj
        self.db = db

    def _initialize(self):
        """Initialize the trace.
        """
        
        size = 1
        try:
            size = len(self.obj.value)
        except TypeError:
            pass
        
        self.current_trace = 1
        try:
            query = "create table %s (recid INTEGER NOT NULL PRIMARY KEY, trace int(5), %s FLOAT)" % (self.obj.__name__, ' FLOAT, '.join(['v%s' % (x+1) for x in range(size)]))
            self.db.cur.execute(query)
        except Exception:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.obj.__name__)
            last_trace = self.db.cur.fetchall()[0][0]
            if last_trace: self.current_trace =  last_trace + 1

    def tally(self,index):
        """Adds current value to trace"""
        try:
            value = self.obj.value.copy()
            valstring = ', '.join(value.astype('c'))
        except AttributeError:
            value = self.obj.value
            valstring = str(value)  
            
        # Add value to database
        self.db.cur.execute("INSERT INTO %s values (NULL, %s, %s)" % (self.obj.__name__, self.current_trace, valstring))

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
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.obj.__name__)
            chain = self.db.cur.fetchall()[0][0]
            
        self.db.cur.execute('SELECT * FROM %s WHERE trace=%s' % (self.obj.__name__, chain))
        trace = transpose(transpose(self.db.cur.fetchall())[2:])
            
        return squeeze(trace[slicing])

    def _finalize(self):
        """Nothing done here."""
        pass

    __call__ = gettrace

class Database(object):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self):
        pass
        
    def _initialize(self, length, model):
        """Initialize database."""
        self.model = model
        # Connect to database
        self.db = sql.connect(self.model.__name__)
        self.cur = self.db.cursor()
        
        for object in self.model._pymc_objects_to_tally:
            object.trace._initialize()
            
    def _finalize(self, *args, **kwds):
        """Close database."""
        self.cur.close()
        self.db.close()
