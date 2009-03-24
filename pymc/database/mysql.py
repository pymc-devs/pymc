"""
MySQL database backend

Store traces from tallyable objects in individual SQL tables.

Implementation Notes
--------------------
For each object, a table is created with the following format:

key (INT), trace (INT),  v1 (FLOAT), v2 (FLOAT), v3 (FLOAT) ...

The key is autoincremented each time a new row is added to the table.
trace denotes the chain index, and starts at 0.

Additional Dependencies
-----------------------
 * MySQL <http://www.mysql.com/downloads/>
 * mysql-python <http://sourceforge.net/projects/mysql-python>

Created by Chris Fonnesbeck on 2007-02-01.
Updated by CF 2008-07-13.
DB API changes, October 2008, DH.
"""

# TODO: Commit multiple tallies with single database call.
# TODO: Add support for integer valued objects.

from numpy import zeros, shape, squeeze, transpose
import base, pickle, ram, pymc, sqlite
import MySQLdb, pdb

__all__ = ['Trace', 'Database', 'load']

class Trace(sqlite.Trace):
    """MySQL Trace class."""

    def _initialize(self, chain, length):
        """Initialize the trace. Create a table.
        """
        # If the table already exists, exit now.
        if chain != 0:
            return

        # Determine size
        try:
            size = len(self._getfunc())
        except TypeError:
            size = 1

        query = "create table %s (recid INTEGER PRIMARY KEY AUTO_INCREMENT, trace int(5), %s FLOAT)" % (self.name, ' FLOAT, '.join(['v%s' % (x+1) for x in range(size)]))
        self.db.cur.execute(query)

    def tally(self, chain):
        """Adds current value to trace."""
        size = 1
        try:
            size = len(self._getfunc())
        except TypeError:
            pass

        try:
            valstring = ', '.join(['%f'%x for x in self._getfunc()])
        except:
            valstring = str(self._getfunc())

        # Add value to database
        query = "INSERT INTO %s (trace, %s) values (%s, %s)" % (self.name, ' ,'.join(['v%s' % (x+1) for x in range(size)]), chain, valstring)

        self.db.cur.execute(query)



class Database(sqlite.Database):
    """MySQL database."""

    def __init__(self, dbname, dbuser='', dbpass='', dbhost='localhost', dbport=3306, dbmode='a'):
        """Open or create a MySQL database.

        :Parameters:
        dbname : string
          The name of the database file.
        dbuser : string
          The database user name.
        dbpass : string
          The database user password.
        dbhost : string
          The location of the database host.
        dbport : int
          The port number to use to reach the database host.
        dbmode : {'a', 'w'}
          File mode.  Use `a` to append values, and `w` to overwrite
          an existing database.
        """
        self.__name__ = 'mysql'
        self.dbname = dbname
        self.__Trace__ = Trace

        self.variables_to_tally = []   # A list of sequences of names of the objects to tally.
        self._traces = {} # A dictionary of the Trace objects.
        self.chains = 0
        self._default_chain = -1

        self._user = dbuser
        self._passwd = dbpass
        self._host = dbhost
        self._port = dbport
        self.mode = dbmode

        # Connect to database
        self.DB = MySQLdb.connect(user=self._user, passwd=self._passwd, host=self._host, port=self._port)
        self.cur = self.DB.cursor()

        # Try and create database with model name
        try:
            self.cur.execute('CREATE DATABASE %s' % self.dbname)
        except Exception:
            # If already exists, switch to database
            self.cur.execute('USE %s' % self.dbname)

            # If in write mode, remove existing tables.
            if self.mode == 'w':
                self.clean()


    def clean(self):
        """Deletes tables from database"""
        tables = get_table_list(self.cur)

        for t in tables:
            self.cur.execute('DROP TABLE %s' % t)

    def savestate(self, state):
        """Store a dictionnary containing the state of the Sampler and its
        StepMethods."""
        pass

    def getstate(self):
        """Return a dictionary containing the state of the Sampler and its
        StepMethods."""
        return {}


def load(dbname='', dbuser='', dbpass='', dbhost='localhost', dbport=3306):
    """Load an existing MySQL database.

    Return a Database instance.
    """
    db = Database(dbname=dbname, dbuser=dbuser, dbpass=dbpass, dbhost=dbhost, dbport=dbport, dbmode='a')
    db.DB = MySQLdb.connect(db=dbname, user=dbuser, passwd=dbpass, host=dbhost, port=dbport)
    db.cur = db.DB.cursor()

    # Get the name of the objects
    tables = get_table_list(db.cur)

    # Create a Trace instance for each object
    chains = 0
    for name in tables:
        db._traces[name] = Trace(name=name, db=db)
        setattr(db, name, db._traces[name])
        db.cur.execute('SELECT MAX(trace) FROM %s'%name)
        chains = max(chains, db.cur.fetchall()[0][0]+1)

    db.chains=chains
    db.variables_to_tally = chains * [tables,]
    db._state_ = {}
    return db

# Copied form Django.
def get_table_list(cursor):
    """Returns a list of table names in the current database."""
    # Skip the sqlite_sequence system table used for autoincrement key
    # generation.
    cursor.execute("SHOW TABLES")
    return [row[0] for row in cursor.fetchall()]
