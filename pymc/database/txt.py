"""
TXT database module

Store the traces in ASCII files.

For each chain, a directory named `Chain_#` is created. In this directory,
one file per tallyable object is created containing the values of the object.

Implementation Notes
--------------------
The NumPy arrays are saved and loaded using NumPy's `loadtxt` and `savetxt`
functions.

Changeset
---------
Nov. 30, 2007: Implemented load function. DH
Oct. 24, 2008: Implemented savestate. Implemented parallel chain tallying. DH
"""


import base, ram
import os, datetime, shutil, re
import numpy as np
from numpy import array
import string

__all__ = ['Trace', 'Database', 'load']

CHAIN_NAME = 'Chain_%d'

class Trace(ram.Trace):
    """Txt Trace Class.

    Store the trace in a ASCII file located in one directory per chain.

    dbname/
      Chain_0/
        <object name>.txt
        <object name>.txt
        ...
      Chain_1/
        <object name>.txt
        <object name>.txt
        ...
      ...
    """

    def _finalize(self, chain):
        """Write the trace to an ASCII file.

        :Parameter:
        chain : int
          The chain index.
        """

        path = os.path.join(self.db._directory, self.db.get_chains()[chain], self.name+'.txt')
        arr = self.gettrace(chain=chain)
        f = open(path, 'w')
        print >> f, '# Variable: %s' % self.name
        print >> f, '# Sample shape: %s' % str(arr.shape)
        print >> f, '# Date: %s' % datetime.datetime.now()
        np.savetxt(f, arr, delimiter=',')
        f.close()

class Database(base.Database):
    """Txt Database class."""

    def __init__(self, dbname=None, dbmode='a'):
        """Create a Txt Database.

        :Parameters:
        dbname : string
          Name of the directory where the traces are stored.
        dbmode : {a, r, w}
          Opening mode: a:append, w:write, r:read.
        """
        self.__name__ = 'txt'
        self._directory = dbname
        self.__Trace__ = Trace
        self.mode = dbmode

        self.variables_to_tally = []   # A list of sequences of names of the objects to tally.
        self._traces = {} # A dictionary of the Trace objects.
        self.chains = 0
        self._default_chain = -1

        if os.path.exists(self._directory):
            if dbmode=='w':
                shutil.rmtree(self._directory)
                os.mkdir(self._directory)
        else:
            os.mkdir(self._directory)

    def get_chains(self):
        """Return an ordered list of the `Chain_#` directories in the db
        directory."""
        chains = []
        try:
            content = os.listdir(self._directory)
            for c in content:
                if os.path.isdir(os.path.join(self._directory, c)) and c.startswith(CHAIN_NAME[:-2]):
                    chains.append(c)
        except:
            pass
        chains.sort()
        return chains

    def _initialize(self, variables, length):
        """Create folder to store simulation results."""

        dir = os.path.join(self._directory, CHAIN_NAME%self.chains)
        os.mkdir(dir)

        base.Database._initialize(self, variables, length)

    def savestate(self, state):
        """Save the sampler's state in a state.txt file."""
        np.set_printoptions(threshold=1e6)
        file = open(os.path.join(self._directory, 'state.txt'), 'w')
        print >> file, state
        file.close()



def load(dirname):
    """Create a Database instance from the data stored in the directory."""
    if not os.path.exists(dirname):
        raise AttributeError, 'No txt database named %s'%dirname

    db = Database(dirname, dbmode='a')
    chain_folders = [os.path.join(dirname, c) for c in db.get_chains()]
    db.chains = len(chain_folders)

    data = {}
    for chain, folder in enumerate(chain_folders):
        files = os.listdir(folder)
        varnames = varname(files)
        db.variables_to_tally.append(varnames)
        for file in files:
            name = varname(file)
            try:
                data[name][chain] = np.loadtxt(os.path.join(folder, file), delimiter=',')
            except:
                data[name] = {chain:np.loadtxt(os.path.join(folder, file), delimiter=',')}


    # Create the Traces.
    for name, values in data.iteritems():
        db._traces[name] = Trace(name=name, value=values, db=db)
        setattr(db, name, db._traces[name])

    # Load the state.
    statefile = os.path.join(dirname, 'state.txt')
    if os.path.exists(statefile):
        file = open(statefile, 'r')
        db._state_ = eval(file.read())
    else:
        db._state_= {}
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

