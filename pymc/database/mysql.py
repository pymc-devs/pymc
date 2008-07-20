"""
mysql.py

MySQL trace module

Created by Chris Fonnesbeck on 2007-02-01.
Updated by CF 2008-07-13.
"""

from numpy import zeros, shape, squeeze, transpose
import base, pickle, ram, pymc, sqlite
import MySQLdb, pdb

class Trace(sqlite.Trace):
    """MySQL Trace class."""
    
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
            query = "create table %s (recid INTEGER PRIMARY KEY AUTO_INCREMENT, trace int(5), %s FLOAT)" % (self.name, ' FLOAT, '.join(['v%s' % (x+1) for x in range(size)]))
            self.db.cur.execute(query)
            self.current_trace = 1
        except Exception:
            self.db.cur.execute('SELECT MAX(trace) FROM %s' % self.name)
            last_trace = self.db.cur.fetchall()[0][0]
            try:
                self.current_trace = last_trace + 1
            except TypeError:
                self.current_trace = 1
    
    def tally(self,index):
        """Adds current value to trace"""
        
        size = 1
        try:
            size = len(self._obj.value)
        except TypeError:
            pass
        
        try:
            # I changed str(x) to '%f'%x to solve a bug appearing due to
            # locale settings. In french for instance, str prints a comma
            # instead of a colon to indicate the decimal, which confuses
            # the database into thinking that there are more values than there
            # is.  A better solution would be to use another delimiter than the
            # comma. -DH
            valstring = ', '.join(['%f'%x for x in self._obj.value])
        except:
            valstring = str(self._obj.value)
        
        # Add value to database
        self.db.cur.execute("INSERT INTO %s (trace, %s) values (%s, %s)" % (self.name, ' ,'.join(['v%s' % (x+1) for x in range(size)]), self.current_trace, valstring))


class Database(pickle.Database):
    """Define the methods that will be assigned to the Model class"""
    def __init__(self, dbuser='', dbpass='', dbhost='localhost', dbport=3306):
        
        self._user = dbuser
        self._passwd = dbpass
        self._host = dbhost
        self._port = dbport
        
        self.Trace = Trace
    
    def _initialize(self, length):
        """Initialize database."""
        
        dbname = self.model.__name__
        
        # Connect to database
        self.DB = MySQLdb.connect(user=self._user, passwd=self._passwd, host=self._host, port=self._port)
        self.cur = self.DB.cursor()
        
        try:
            # Try and create database with model name
            self.cur.execute('CREATE DATABASE %s' % dbname)
        except Exception:
            # If already exists, switch to database
            self.cur.execute('USE %s' % dbname)
        
        for object in self.model._variables_to_tally:
            object.trace._initialize()
    
    def connect(self, sampler):
        """Link the Database to the Sampler instance.
        
        If database is loaded from a file, restore the objects trace
        to their stored value, if a new database is created, instantiate
        a Trace for the nodes to tally.
        """
        base.Database.connect(self, sampler)
    
    def commit(self):
        """Commit updates to database"""
        
        self.DB.commit()
        
    def clean(self):
        """Deletes tables from database"""
        
        tables = get_table_list(self.cur)
        
        for t in tables:
            self.cur.execute('DROP TABLE %s' % t)
        
    
    def close(self, *args, **kwds):
        """Close database."""
        self.cur.close()
        self.commit()
        self.DB.close()
    
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
        db.DB = MySQLdb.connect(db.filename)
        db.cur = db.DB.cursor()
        
        # Get the name of the objects
        tables = get_table_list(db.cur)
        
        # Create a Trace instance for each object
        for name in tables:
            setattr(db, name, Trace(name=name))
            o = getattr(db, name)
            setattr(o, 'db', db)
        return db

def load(dbname='', dbuser='', dbpass='', dbhost='localhost', dbport=3306):
    """Load an existing database.
    
    Return a Database instance.
    """
    db = Database()
    db.DB = MySQLdb.connect(db=dbname, user=dbuser, passwd=dbpass, host=dbhost, port=dbport)
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
    cursor.execute("SHOW TABLES")
    return [row[0] for row in cursor.fetchall()]