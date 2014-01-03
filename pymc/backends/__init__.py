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

After a backend is finished sampling, values can be accessed in a few
ways. The easiest way is to index the backend object with a variable or
variable name.

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

To write a custom backend, two base classes should be inherited:
pymc.backends.base.Backend and pymc.backends.base.Trace. The first class
handles sampling, while the second provides access to the sampled
values.

These following sampling-related methods of base.Backend should be
define in the child class:

- _initialize_trace: Return a trace object for to store the sampled
  values.

- _create_trace: Create the trace object for a specific variable and
  chain. For example, the NumPy array backend creates an array of zeros
  shaped according to the number of planned iterations and the shape of
  the given variable.

- _store_value: Store the value for a draw of a particular variable
  (using the trace from `_create_trace`).

- commit: After a set amount of iterations, the sampling results will be
  committed to the backend. In the case of in memory backends (NumPy and
  Text), this doesn't do anything.

- close: This method is called following sampling and should perform any
  actions necessary for finalizing and cleaning up the backend.

If backend-specific initialization is required, redefine `__init__` to
include this and the call the parent `__init__` method.

In addition to sampling methods, several methods in base.Trace should
also be defined.

- get_values: This is the core method for selecting values from the
  backend. It can be called directly and is used by __getitem__ when the
  backend is indexed with a variable name or object.

- _slice: Defines how the backend returns a slice of itself. This
  is called if the backend is indexed with a slice range.

- point: Returns values for each variables at a single iteration. This
  is called if the backend is indexed with a single integer.

For specific examples, see pymc.backends.{ndarray,text,sqlite}.py.
"""
from pymc.backends.ndarray import NDArray
from pymc.backends.text import Text
from pymc.backends.sqlite import SQLite
