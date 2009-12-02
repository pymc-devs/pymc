"""
SQLite database backend

Store traces from tallyable objects in individual SQL tables.

Implementation Notes
--------------------
For each object, a table is created with the following format:

key (INT), trace (INT),  v1 (FLOAT), v2 (FLOAT), v3 (FLOAT) ...

For multidimensional objects, ndim>1, eg (2,2) the table has the following format:

key (INT), trace (INT),  v1_1 (FLOAT), v1_2 (FLOAT), v2_1 (FLOAT), v2_2 (FLOAT)

The key is autoincremented each time a new row is added to the table.
trace denotes the chain index, and starts at 0.

Additional Dependencies
-----------------------
sqlite3 <http://www.sqlite.org>

Changeset
---------
Created by Chris Fonnesbeck on 2007-02-01.
Updated by DH on 2007-04-04.
DB API changes, October 2008, DH.
Added support for multidimensional arrays, DH Oct. 2009
"""

# TODO: Add support for integer valued objects.

import numpy as np
from numpy import zeros, shape, squeeze, transpose
import sqlite3
import base, pickle, ram, pymc
import pdb,os
from pymc.database import base

__all__ = ['Trace', 'Database', 'load']

class Trace(base.Trace):
    """SQLite Trace class."""

    def _initialize(self, chain, length):
        """Create an SQL table.
        """

        if self._getfunc is None:
            self._getfunc = self.db.model._funs_to_tally[self.name]


        # Determine size
        try:
            self._shape = np.shape(self._getfunc())
        except TypeError:
            self._shape = None

        self._vstr = ', '.join(var_str(self._shape))

        # If the table already exists, exit now.
        if chain != 0:
            return

        # Create the variable name strings.
        vstr = ', '.join(v + ' FLOAT' for v in var_str(self._shape))

        query = "create table %s (recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, trace  int(5), %s )" % (self.name, vstr)

        self.db.cur.execute(query)



    def tally(self, chain):
        """Adds current value to trace."""

        try:
            # I changed str(x) to '%f'%x to solve a bug appearing due to
            # locale settings. In french for instance, str prints a comma
            # instead of a colon to indicate the decimal, which confuses
            # the database into thinking that there are more values than there
            # is.  A better solution would be to use another delimiter than the
            # comma. -DH
            valstring = ', '.join(['%f'%x for x in np.ravel(self._getfunc())])
        except:
            valstring = str(self._getfunc())


        # Add value to database
        query = "INSERT INTO %s (recid, trace, %s) values (NULL, %s, %s)" % \
            (self.name, self._vstr, chain, valstring)
        self.db.cur.execute(query)



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

        # If chain is None, get the data from all chains.
        if chain is None:
            self.db.cur.execute('SELECT * FROM %s' % self.name)
            trace = self.db.cur.fetchall()
        else:
            # Deal with negative chains (starting from the end)
            if chain < 0:
                chain = range(self.db.chains)[chain]
            self.db.cur.execute('SELECT * FROM %s WHERE trace=%s' % (self.name, chain))
            trace = self.db.cur.fetchall()
        trace = np.array(trace)[:,2:]
        if len(self._shape) > 1:
            trace = trace.reshape(-1, *self._shape)
        return squeeze(trace[slicing])


    __call__ = gettrace

#    def nchains(self):
#        """Return the number of existing chains, completed or not."""
#        try:
#            self.db.cur.execute('SELECT MAX(trace) FROM %s'%self.name)
#            trace = self.db.cur.fetchall()[0][0]
#
#            if trace is None:
#                return 0
#            else:
#                return trace + 1
#        except:
#           return 0

    def length(self, chain=-1):
        """Return the sample length of given chain. If chain is None,
        return the total length of all chains."""
        return len(self.gettrace(chain=chain))

class Database(base.Database):
    """SQLite database.
    """

    def __init__(self, dbname, dbmode='a'):
        """Open or create an SQL database.

        :Parameters:
        dbname : string
          The name of the database file.
        dbmode : {'a', 'w'}
          File mode.  Use `a` to append values, and `w` to overwrite
          an existing file.
        """
        self.__name__ = 'sqlite'
        self.dbname = dbname
        self.__Trace__ = Trace

        self.trace_names = []   # A list of sequences of names of the objects to tally.
        self._traces = {} # A dictionary of the Trace objects.
        self.chains = 0

        if os.path.exists(dbname) and dbmode=='w':
            os.remove(dbname)

        self.DB = sqlite3.connect(dbname, check_same_thread=False)
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

def load(dbname):
    """Load an existing SQLite database.

    Return a Database instance.
    """
    db = Database(dbname)

    # Get the name of the objects
    tables = get_table_list(db.cur)

    # Create a Trace instance for each object
    chains = 0
    for name in tables:
        db._traces[name] = Trace(name=name, db=db)
        db._traces[name]._shape = get_shape(db.cur, name)
        setattr(db, name, db._traces[name])
        db.cur.execute('SELECT MAX(trace) FROM %s'%name)
        chains = max(chains, db.cur.fetchall()[0][0]+1)

    db.chains=chains
    db.trace_names = chains * [tables,]
    db._state_ = {}
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

def get_shape(cursor, name):
    """Return the shape of the table ``name``."""
    cursor.execute('select * from %s'% name)
    inds = cursor.description[-1][0][1:].split('_')
    return tuple([int(i) for i in inds])


def var_str(shape):
    """Return a sequence of strings naming the element of the tallyable object.

    :Examples:
    >>> var_str((5,))
    ['v1', 'v2', 'v3', 'v4', 'v5']

    >>> var_str((2,2))
    ['v1_1', 'v1_2', 'v2_1', 'v2_2']
    """

    if shape in [None, ()]:
        return ['v1',]

    size = np.prod(shape)
    indices = (np.indices(shape) + 1).reshape(-1, size)
    return ['v'+'_'.join(map(str, i)) for i in zip(*indices)]
