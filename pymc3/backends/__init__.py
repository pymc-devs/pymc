"""Backends for traces

Available backends
------------------

1. NumPy array (pymc3.backends.NDArray)
2. Text files (pymc3.backends.Text)
3. SQLite (pymc3.backends.SQLite)

The NDArray backend holds the entire trace in memory, whereas the Text
and SQLite backends store the values while sampling.

Selecting a backend
-------------------

By default, a NumPy array is used as the backend. To specify a different
backend, pass a backend instance to `sample`.

For example, the following would save the sampling values to CSV files
in the directory 'test'.

    >>> import pymc3 as pm
    >>> db = pm.backends.Text('test')
    >>> trace = pm.sample(..., trace=db)

Selecting values from a backend
-------------------------------

After a backend is finished sampling, it returns a MultiTrace object.
Values can be accessed in a few ways. The easiest way is to index the
backend object with a variable or variable name.

    >>> trace['x']  # or trace.x or trace[x]

The call will return the sampling values of `x`, with the values for
all chains concatenated. (For a single call to `sample`, the number of
chains will correspond to the `njobs` argument.)

To discard the first N values of each chain, slicing syntax can be
used.

    >>> trace['x', 1000:]

The `get_values` method offers more control over which values are
returned. The call below will discard the first 1000 iterations
from each chain and keep the values for each chain as separate arrays.

    >>> trace.get_values('x', burn=1000, combine=False)

The `chains` parameter of `get_values` can be used to limit the chains
that are retrieved.

    >>> trace.get_values('x', burn=1000, chains=[0, 2])

MultiTrace objects also support slicing. For example, the following
call would return a new trace object without the first 1000 sampling
iterations for all traces and variables.

    >>> sliced_trace = trace[1000:]

The backend for the new trace is always NDArray, regardless of the
type of original trace.  Only the NDArray backend supports a stop
value in the slice.

Loading a saved backend
-----------------------

Saved backends can be loaded using `load` function in the module for the
specific backend.

    >>> trace = pm.backends.text.load('test')

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

- __len__: This should return the number of draws.

When `pymc3.sample` finishes, it wraps all trace objects in a MultiTrace
object that provides a consistent selection interface for all backends.
If the traces are stored on disk, then a `load` function should also be
defined that returns a MultiTrace object.

For specific examples, see pymc3.backends.{ndarray,text,sqlite}.py.
"""
from ..backends.ndarray import NDArray
from ..backends.text import Text
from ..backends.sqlite import SQLite

_shortcuts = {'text': {'backend': Text,
                       'name': 'mcmc'},
              'sqlite': {'backend': SQLite,
                         'name': 'mcmc.sqlite'}}
