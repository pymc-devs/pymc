"""Backends for traces

Available backends
------------------

1. NumPy array (pymc.backends.NDArray)
2. Text files (pymc.backends.Text)
3. SQLite (pymc.backends.SQLite)

The NumPy arrays and text files both hold the entire trace in memory,
whereas SQLite commits the trace to the database while sampling.

Selecting a backend
-------------------

By default, a NumPy array is used as the backend. To specify a different
backend, pass a backend instance to `sample`.

For example, the following would save traces to the file 'test.db'.

    >>> import pymc as pm
    >>> db = pm.backends.SQLite('test.db')
    >>> trace = pm.sample(..., db=db)

Selecting values from a backend
-------------------------------

After a backend is finished sampling, it returns a Trace object. Values
can be accessed in a few ways. The easiest way is to index the backend
object with a variable or variable name.

    >>> trace['x']  # or trace[x]

The call will return a list containing the sampling values for all
chains of `x`. (Each call to `pymc.sample` creates a separate chain of
samples.)

For more control is needed of which values are returned, the
`get_values` method can be used. The call below will return values from
all chains, burning the first 1000 iterations from each chain.

    >>> trace.get_values('x', burn=1000)

Setting the `combined` flag will concatenate the results from all the
chains.

    >>> trace.get_values('x', burn=1000, combine=True)

The `chains` parameter of `get_values` can be used to limit the chains
that are retrieved. To work with a subset of chains without having to
specify `chains` each call, you can set the `active_chains` attribute.

    >>> trace.chains
    [0, 1, 2]
    >>> trace.active_chains = [0, 2]

After this, only chains 0 and 2 will be used in operations that work
with multiple chains.

Similary, the `default_chain` attribute sets which chain is used for
functions that require a single chain (e.g., point).

   >>> trace.point(4)  # or trace[4]

Backends can also suppport slicing the trace object. For example, the
following call would return a new trace object without the first 1000
sampling iterations for all variables.

    >>> sliced_trace = trace[1000:]

Loading a saved backend
-----------------------

Saved backends can be loaded using `load` function in the module for the
specific backend.

    >>> trace = pm.backends.sqlite.load('test.db')

Writing custom backends
-----------------------

Backends consist of two classes: one that handles storing the sample
results (e.g., backends.ndarray.NDArray or backends.sqlite.SQLite) and
one that handles value selection (e.g., backends.ndarray.Trace or
backends.sqlite.Trace).

Three methods of the storage class will be called:

- setup: Before sampling is started, the `setup` method will be called
  with two arguments: the number of draws and the chain number. This is
  useful setting up any structure for storing the sampling values that
  require the above information.

- record: Record the sampling results for the current draw. This method
  will be called with a dictionary of values mapped to the variable
  names. This is the only function that *must* do something to have a
  meaningful backend.

- close: This method is called following sampling and should perform any
  actions necessary for finalizing and cleaning up the backend.

The base storage class `backends.base.Backend` provides model setup that
is used by PyMC backends.

After sampling has completed, the `trace` attribute of the storage
object will be returned. To have a consistent interface with the backend
trace objects in PyMC, this attribute should be an instance of a class
that inherits from pymc.backends.base.Trace, and several methods in the
inherited Trace object should be defined.

- get_values: This is the core method for selecting values from the
  backend. It can be called directly and is used by __getitem__ when the
  backend is indexed with a variable name or object.

- _slice: Defines how the backend returns a slice of itself. This
  is called if the backend is indexed with a slice range.

- point: Returns values for each variables at a single iteration. This
  is called if the backend is indexed with a single integer.

- __len__: This should return the number of draws (for the default
  chain).

- chains: Property that returns a list of chains

In addtion, a `merge_chains` method should be defined if the backend
will be used with parallel sampling. This method describes how to merge
sampling chains from a list of other traces.

As mentioned above, the only method necessary to store the sampling
values is `record`. Other methods in the storage may consist of only a
pass statement. The storage object should have an attribute `trace`
(with a `merge_chains` method for parallel sampling), but this does not
have to do anything if storing the values is all that is desired. The
backends.base.Trace is provided for convenience in setting up a
consistent Trace object.

For specific examples, see pymc.backends.{ndarray,text,sqlite}.py.
"""
from pymc.backends.ndarray import NDArray
from pymc.backends.text import Text
from pymc.backends.sqlite import SQLite
