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
    >>> trace = pm.sample(..., trace=db)

Selecting values from a backend
-------------------------------

After a backend is finished sampling, it returns a MultiTrace object.
Values can be accessed in a few ways. The easiest way is to index the
backend object with a variable or variable name.

    >>> trace['x']  # or trace[x]

The call will return a list containing the sampling values for all
chains of `x`. (Each call to `pymc.sample` creates a separate chain of
samples.)

For more control of which values are returned, the `get_values` method
can be used. The call below will return values from all chains, burning
the first 1000 iterations from each chain.

    >>> trace.get_values('x', burn=1000)

Setting the `combined` flag will concatenate the results from all the
chains.

    >>> trace.get_values('x', burn=1000, combine=True)

The `chains` parameter of `get_values` can be used to limit the chains
that are retrieved.

    >>> trace.get_values('x', burn=1000, combine=True, chains=[0, 2])

Backends can also suppport slicing the MultiTrace object. For example,
the following call would return a new trace object without the first
1000 sampling iterations for all traces and variables.

    >>> sliced_trace = trace[1000:]

Loading a saved backend
-----------------------

Saved backends can be loaded using `load` function in the module for the
specific backend.

    >>> trace = pm.backends.sqlite.load('test.db')

Writing custom backends
-----------------------

Backends consist of a class that handles sampling storage and value
selection. Three sampling methods of backend will be called:

- setup: Before sampling is started, the `setup` method will be called
  with two arguments: the number of draws and the chain number. This is
  useful setting up any structure for storing the sampling values that
  require the above information.

- record: Record the sampling results for the current draw. This method
  will be called with a dictionary of values mapped to the variable
  names. This is the only sampling function that *must* do something to
  have a meaningful backend.

- close: This method is called following sampling and should perform any
  actions necessary for finalizing and cleaning up the backend.

The base storage class `backends.base.BaseTrace` provides common model
setup that is used by all the PyMC backends.

Several selection methods must also be defined:

- get_values: This is the core method for selecting values from the
  backend. It can be called directly and is used by __getitem__ when the
  backend is indexed with a variable name or object.

- _slice: Defines how the backend returns a slice of itself. This
  is called if the backend is indexed with a slice range.

- point: Returns values for each variable at a single iteration. This is
  called if the backend is indexed with a single integer.

- __len__: This should return the number of draws (for the highest chain
  number).

When `pymc.sample` finishes, it wraps all trace objects in a MultiTrace
object that provides a consistent selection interface for all backends.
If the traces are stored on disk, then a `load` function should also be
defined that returns a MultiTrace object.

For specific examples, see pymc.backends.{ndarray,text,sqlite}.py.
"""
from pymc.backends.ndarray import NDArray
from pymc.backends.text import Text
from pymc.backends.sqlite import SQLite

_shortcuts = {'text': {'backend': Text,
                       'name': 'mcmc'},
              'sqlite': {'backend': SQLite,
                         'name': 'mcmc.sqlite'}}
