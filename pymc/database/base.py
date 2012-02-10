"""
Base backend

Trace and Database classes from the other modules should Subclass the base
classes.

Concepts
--------

Each backend must define a Trace and a Database class that subclass the
base.Trace and base.Database classes.

The Model itself is given a `db` attribute, an instance of the Database
class. The __init__ method of the Database class does the work of preparing the
database. This might mean creating a file, connecting to a remote database
server, etc.

Right after a Database is instantiated, the Sampler tells it to connect itself
the model. This is taken care of by the connect_model method. This method
creates Trace objects for all the tallyable pymc objects. These traces are
stored in a dictionary called _traces, owned by the Database instance of this
Sampler. Previously, the Trace instances were owned by the tallyable variables
themselves. As A.P. pointed out, this is problematic if the same object is used
in different models, because each new Sampler will overwrite the last Trace
instance of the variable.

When the `sample` method is called, the Sampler tells the Database to initialize
itself. Database then tells all the tallyable objects to initialize themselves.
This might mean creating a numpy array, a list, an sqlite table, etc.

When the Sampler calls `tally`, it sends a message to the Database to tally its
objects. Each object then appends its current value to the given chain.

At the end of sampling, the Sampler tells the Database to finalize it's state,
and the database relays the finalize call to each Trace object.

To get the samples from the database, the Sampler class provides a trace method
``trace(self, name, chain=-1)`` which first sets the db attribute
_default_chain to chain and returns the trace instance, so that calling
S.trace('early_mean')[:] will return all samples from the last chain. One potential
problem with this is that user could simply do ``S.trace('early_mean') = 5`` and
erase the Trace object. A read-only dictionary-like class could solve
that problem.

Some backends require being closed before saving the results. This needs to be
done explicitly by the user.
"""
import pymc
import numpy as np
import types
import sys, traceback, warnings
import copy
__all__=['Trace', 'Database']

from pymc import six
from pymc.six import print_

class Trace(object):
    """Base class for Trace objects.

    Each tallyable pymc object is given a trace attribute, which is a Trace
    instance. These trace methods are generally called by the Database instance
    or by the user.
    """

    def __init__(self, name, getfunc=None, db=None):
        """Create a Trace instance.

        :Parameters:
        name : string
          The trace object name. This name should uniquely identify
          the pymc variable.
        getfunc : function
          A function returning the value to tally.
        db : Database instance
          The database owning this Trace.
        """
        self._getfunc = getfunc
        self.name = name
        self.db = db
        self._chain = -1

    def _initialize(self, chain, length):
        """Prepare for tallying. Create a new chain."""
        # If this db was loaded from the disk, it may not have its
        # tallied step methods' getfuncs yet.
        if self._getfunc is None:
            self._getfunc = self.db.model._funs_to_tally[self.name]


    def tally(self, chain):
        """Appends the object's value to a chain.

        :Parameters:
        chain : integer
          Chain index.
        """
        pass

    def truncate(self, index, chain):
        """For backends that preallocate memory, removed the unused memory."""
        pass

    def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
        """Return the trace.

        :Parameters:
          - burn (int): The number of transient steps to skip.
          - thin (int): Keep one in thin.
          - chain (int): The index of the chain to fetch. If None, return all chains.
          - slicing: A slice, overriding burn and thin assignement.
        """
        warnings.warn('Use Sampler.trace method instead.', DeprecationWarning)


    # By convention, the __call__ method is assigned to gettrace.
    __call__ = gettrace

    def __getitem__(self, i):
        """Return the trace corresponding to item (or slice) i for the chain
        defined by self._chain.
        """
        if type(i) == types.SliceType:
            return self.gettrace(slicing=i, chain=self._chain)
        else:
            return self.gettrace(slicing=slice(i,i+1), chain=self._chain)

    def _finalize(self, chain):
        """Execute task necessary when tallying is over for this trace."""
        pass

    def length(self, chain=-1):
        """Return the length of the trace.

        :Parameters:
        chain : int or None
          The chain index. If None, returns the combined length of all chains.
        """
        pass
        
    def stats(self, alpha=0.05, start=0, batches=100, chain=None):
        """
        Generate posterior statistics for node.
        
        :Parameters:
        name : string
          The name of the tallyable object.
          
        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.

        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.
          
        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.
          
        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        """
        from pymc.utils import hpd, quantiles

        try:
            trace = np.squeeze(np.array(self.db.trace(self.name)(chain=chain), float))[start:]
            
            n = len(trace)
            if not n:
                print_('Cannot generate statistics for zero-length trace in', self.__name__)
                return


            return {
                'n': n,
                'standard deviation': trace.std(0),
                'mean': trace.mean(0),
                '%s%s HPD interval' % (int(100*(1-alpha)),'%'): hpd(trace, alpha),
                'mc error': batchsd(trace, batches),
                'quantiles': quantiles(trace)
            }
        except:
            print_('Could not generate output statistics for', self.name)
            return


class Database(object):
    """Base Database class.

    The Database job is to create a database on disk and communicate
    with its Trace objects to tally values.
    """

    def __init__(self, dbname):
        """Create a Database instance.

        This method:
         * Assigns the local Trace class to the __Trace__ attribute.
         * Assings its name to the __name__ attribute.
         * Does whatever is necessary to set up the database.

        :Note:
        All arguments to __init__ should begin by db: dbname, dbmode, etc. This
        is critical to avoid clashes between arguments to Sampler and Database.

        This method should not be subclassed.
        """
        self.__Trace__ = Trace
        self.__name__ = 'base'
        self.dbname = dbname
        self.trace_names = []   # A list of sequences of names of the objects to tally.
        self._traces = {} # A dictionary of the Trace objects.
        self.chains = 0

    def _initialize(self, funs_to_tally, length=None):
        """Initialize the tallyable objects.

        Makes sure a Trace object exists for each variable and then initialize
        the Traces.

        :Parameters:
        funs_to_tally : dict
          Name- function pairs.
        length : int
          The expected length of the chain. Some database may need the argument
          to preallocate memory.
        """

        for name, fun in six.iteritems(funs_to_tally):
            if name not in self._traces:
                self._traces[name] = self.__Trace__(name=name, getfunc=fun, db=self)

            self._traces[name]._initialize(self.chains, length)

        self.trace_names.append(funs_to_tally.keys())

        self.chains += 1

    def tally(self, chain=-1):
        """Append the current value of all tallyable object.

       :Parameters:
       chain : int
         The index of the chain to append the values to. By default, the values
         are appended to the last chain.
        """

        chain = range(self.chains)[chain]
        for name in self.trace_names[chain]:
            try:
                self._traces[name].tally(chain)
            except:
                cls, inst, tb = sys.exc_info()
                warnings.warn("""
Error tallying %s, will not try to tally it again this chain.
Did you make all the samevariables and step methods tallyable
as were tallyable last time you used the database file?

Error:

%s"""%(name, ''.join(traceback.format_exception(cls, inst, tb))))
                self.trace_names[chain].remove(name)


    def connect_model(self, model):
        """Link the Database to the Model instance.

        In case a new database is created from scratch, ``connect_model``
        creates Trace objects for all tallyable pymc objects defined in
        `model`.

        If the database is being loaded from an existing file, ``connect_model``
        restore the objects trace to their stored value.

        :Parameters:
        model : pymc.Model instance
          An instance holding the pymc objects defining a statistical
          model (stochastics, deterministics, data, ...)
        """
        # Changed this to allow non-Model models. -AP
        # We could also remove it altogether. -DH
        if isinstance(model, pymc.Model):
            self.model = model
        else:
            raise AttributeError('Not a Model instance.')

        # Restore the state of the Model from an existing Database.
        # The `load` method will have already created the Trace objects.
        if hasattr(self, '_state_'):
            names = set()
            for morenames in self.trace_names:
                names.update(morenames)
            for name, fun in six.iteritems(model._funs_to_tally):
                if name in self._traces:
                    self._traces[name]._getfunc = fun
                    names.discard(name)
            # if len(names) > 0:
            #     print_("Some objects from the database have not been assigned a getfunc", names)

        # Create a fresh new state.
        # We will be able to remove this when we deprecate traces on objects.
        else:
            for name, fun in six.iteritems(model._funs_to_tally):
                if name not in self._traces:
                    self._traces[name] = self.__Trace__(name=name, getfunc=fun, db=self)

    def _finalize(self, chain=-1):
        """Finalize the chain for all tallyable objects."""
        chain = range(self.chains)[chain]
        for name in self.trace_names[chain]:
            self._traces[name]._finalize(chain)
        self.commit()

    def truncate(self, index, chain=-1):
        """Tell the traces to truncate themselves at the given index."""
        chain = range(self.chains)[chain]
        for name in self.trace_names[chain]:
            self._traces[name].truncate(index, chain)

    def commit(self):
        """Flush data to disk."""
        pass

    def close(self):
        """Close the database."""
        self.commit()

    def savestate(self, state):
        """Store a dictionnary containing the state of the Model and its
        StepMethods."""
        self._state_ = state

    def getstate(self):
        """Return a dictionary containing the state of the Model and its
        StepMethods."""
        return getattr(self, '_state_', {})

    def trace(self, name, chain=-1):
        """Return the trace of a tallyable object stored in the database.

        :Parameters:
        name : string
          The name of the tallyable object.
        chain : int
          The trace index. Setting `chain=i` will return the trace created by
          the ith call to `sample`.
        """
        trace = copy.copy(self._traces[name])
        trace._chain = chain
        return trace
        

def load(dbname):
    """Return a Database instance from the traces stored on disk.

    This function should do the following:

     * Create a Database instance `db`.
     * Assign to `db` Trace instances corresponding to the pymc tallyable
       objects that are stored on disk.
     * Assign to `db` a _state_ attribute storing the stepping methods state.
       If this is not implemented, simply set db._state_ = {}.

    After loading a database `db`, we should be able to call
    the gettrace method of each tallyable object stored in the database.
    """
    pass

    
def batchsd(trace, batches=5):
    """
    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.
    """

    if len(np.shape(trace)) > 1:

        dims = np.shape(trace)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        ttrace = np.transpose([t.ravel() for t in trace])

        return np.reshape([batchsd(t, batches) for t in ttrace], dims[1:])

    else:
        if batches == 1: return np.std(trace)/np.sqrt(len(trace))

        try:
            batched_traces = np.resize(trace, (batches, len(trace)/batches))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(trace) % batches
            batched_traces = np.resize(trace[:-resid], (batches, len(trace)/batches))

        means = np.mean(batched_traces, 1)

        return np.std(means)/np.sqrt(batches)
