"""
mysql.py

MySQL trace module

Created by Chris Fonnesbeck on 2007-02-01.
"""

from numpy import zeros, shape, squeeze, transpose
import MySQLdb

class Trace(object):
    """ Define the methods that will be assigned to each stochastic in the
    Model instance."""
    def __init__(self, obj, db):
        """Initialize the instance.
        :Stochastics:
          obj : PyMC object
            Deterministic or Stochastic instance.
          db : database instance
          update_interval: how often database is updated from trace
        """
        self.obj = obj
        self.db = db

    def _initialize(self, update_interval):
        """Initialize the trace.
        """
        self.update_interval = update_interval
        size = 1
        try:
            size = len(self.obj.value)
        except TypeError:
            pass
        
        self._trace = squeeze(zeros((self.update_interval, size), 'f'))
        self.current_trace = 1
        try:
            query = "create table %s (recid int(11) NOT NULL auto_increment PRIMARY KEY, trace int(5), %s FLOAT)" % (self.obj.name, ' FLOAT, '.join(['v%s' % (x+1) for x in range(size)]))
            self.db.cur.execute(query)
        except Exception:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.obj.name)
            last_trace = self.db.cur.fetchall()[0][0]
            if last_trace: self.current_trace =  last_trace + 1

    def tally(self, index):
        """Adds current value to trace"""
        try:
            self._trace[index] = self.obj.value.copy()
        except AttributeError:
            self._trace[index] = self.obj.value
            
        if index + 1 == self.update_interval:
            
            values = ""
            
            for sample in self._trace:
                valstring = ', '.join(sample.astype('c'))
                values += "(NULL, %s, %s)," % (self.current_trace, valstring)
            
            # Add trace to database
            self.db.cur.execute("INSERT INTO %s values %s" % (self.obj.name, values[:-1]))
            
            # Re-initialize trace
            self._trace = zeros(shape(self._trace), self._trace.dtype)

    def truncate(self, index):
        """
        When model receives a keyboard interrupt, it tells the traces
        to truncate their values.
        """
        pass

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
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.obj.name)
            chain = self.db.cur.fetchall()[0][0]
            
        self.db.cur.execute('SELECT * FROM %s WHERE trace=%s' % (self.obj.name, chain))
        trace = transpose(transpose(self.db.cur.fetchall())[2:])
            
        return squeeze(trace[slicing])

    def _finalize(self):
        """Nothing done here."""
        pass

    __call__ = gettrace
    
    def length(self, chain=-1):
        """Return the sample length of given chain. If chain is None,
        return the total length of all chains."""
        return len(self.gettrace(chain=chain))

class Database(object):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, dbuser='', dbpass=''):
        self._user = dbuser
        self._passwd = dbpass
        
    def _initialize(self, length, model):
        """Initialize database."""
        self.model = model
        # Connect to database
        self.db = MySQLdb.connect(user=self._user, passwd=self._passwd)
        self.cur = self.db.cursor()
        
        try:
            # Try and create database with model name
            self.cur.execute('CREATE DATABASE %s' % self.model.__name__)
        except Exception:
            # If already exists, switch to database
            self.cur.execute('USE %s' % self.model.__name__)
    
        for object in self.model._variables_to_tally:
            object.trace._initialize()
            
    def _finalize(self, *args, **kwds):
        """Close database."""
        self.db.close()
