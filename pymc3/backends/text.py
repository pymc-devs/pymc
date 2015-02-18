"""Text file trace backend

After sampling with NDArray backend, save results as text files.

As this other backends, this can be used by passing the backend instance
to `sample`.

    >>> import pymc3 as pm
    >>> db = pm.backends.Text('test')
    >>> trace = pm.sample(..., trace=db)

Or sampling can be performed with the default NDArray backend and then
dumped to text files after.

    >>> from pymc3.backends import text
    >>> trace = pm.sample(...)
    >>> text.dump('test', trace)

Database format
---------------

For each chain, a directory named `chain-N` is created. In this
directory, one file per variable is created containing the values of the
object. To deal with multidimensional variables, the array is reshaped
to one dimension before saving with `numpy.savetxt`. The shape
information is saved in a json file in the same directory and is used to
load the database back again using `numpy.loadtxt`.
"""
import os
import glob
import json
import numpy as np

from ..backends import base
from ..backends.ndarray import NDArray


class Text(NDArray):
    """Text storage

    Parameters
    ----------
    name : str
        Name of directory to store text files
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """
    def __init__(self, name, model=None, vars=None):
        if not os.path.exists(name):
            os.mkdir(name)
        super(Text, self).__init__(name, model, vars)

    def close(self):
        super(Text, self).close()
        _dump_trace(self.name, self)


def dump(name, trace, chains=None):
    """Store NDArray trace as text database.

    Parameters
    ----------
    name : str
        Name of directory to store text files
    trace : MultiTrace of NDArray traces
        Result of MCMC run with default NDArray backend
    chains : list
        Chains to dump. If None, all chains are dumped.
    """
    if not os.path.exists(name):
        os.mkdir(name)
    if chains is None:
        chains = trace.chains
    for chain in chains:
        _dump_trace(name, trace._traces[chain])


def _dump_trace(name, trace):
    """Dump a single-chain trace.
    """
    chain_name = 'chain-{}'.format(trace.chain)
    chain_dir = os.path.join(name, chain_name)
    os.mkdir(chain_dir)

    shapes = {}
    for varname in trace.varnames:
        data = trace.get_values(varname)
        var_file = os.path.join(chain_dir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape
    ## Store shape information for reloading.
    shape_file = os.path.join(chain_dir, 'shapes.json')
    with open(shape_file, 'w') as sfh:
        json.dump(shapes, sfh)


def load(name, chains=None, model=None):
    """Load text database.

    Parameters
    ----------
    name : str
        Path to root directory for text database
    chains : list
        Chains to load. If None, all chains are loaded.
    model : Model
        If None, the model is taken from the `with` context.

    Returns
    -------
    ndarray.Trace instance
    """
    chain_dirs = _get_chain_dirs(name)
    if chains is None:
        chains = list(chain_dirs.keys())

    traces = []
    for chain in chains:
        chain_dir = chain_dirs[chain]
        shape_file = os.path.join(chain_dir, 'shapes.json')
        with open(shape_file, 'r') as sfh:
            shapes = json.load(sfh)
        samples = {}
        for varname, shape in shapes.items():
            var_file = os.path.join(chain_dir, varname + '.txt')
            samples[varname] = np.loadtxt(var_file).reshape(shape)
        trace = NDArray(model=model)
        trace.samples = samples
        trace.chain = chain
        traces.append(trace)
    return base.MultiTrace(traces)


def _get_chain_dirs(name):
    """Return mapping of chain number to directory."""
    return {_chain_dir_to_chain(chain_dir): chain_dir
            for chain_dir in glob.glob(os.path.join(name, 'chain-*'))}


def _chain_dir_to_chain(chain_dir):
    return int(os.path.basename(chain_dir).split('-')[1])
