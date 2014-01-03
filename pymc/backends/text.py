"""Text file trace backend

After sampling with NDArray backend, save results as text files.

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
from contextlib import contextmanager

from pymc.backends import base
from pymc.backends.ndarray import NDArray, Trace


class Text(NDArray):

    def __init__(self, name, model=None, variables=None):
        super(Text, self).__init__(name, model, variables)
        if not os.path.exists(name):
            os.mkdir(name)

    def close(self):
        for chain in self.trace.chains:
            chain_name = 'chain-{}'.format(chain)
            chain_dir = os.path.join(self.name, chain_name)
            os.mkdir(chain_dir)

            shapes = {}
            for var_name in self.var_names:
                data = self.trace.samples[chain][var_name]
                var_file = os.path.join(chain_dir, var_name + '.txt')
                np.savetxt(var_file, data.reshape(-1, data.size))
                shapes[var_name] = data.shape
            ## Store shape information for reloading.
            with _get_shape_fh(chain_dir, 'w') as sfh:
                json.dump(shapes, sfh)


def load(name, chains=None, model=None):
    """Load text database from name

    Parameters
    ----------
    name : str
        Path to root directory for text database
    chains : list or None
        Chains to load. If None, all chains are loaded.
    model : Model
        If None, the model is taken from the `with` context. The trace
        can be loaded without connecting by passing False (although
        connecting to the original model is recommended).

    Returns
    -------
    ndarray.Trace instance
    """
    chain_dirs = _get_chain_dirs(name)
    if chains is None:
        chains = list(chain_dirs.keys())

    trace = Trace(None)

    for chain in chains:
        chain_dir = chain_dirs[chain]
        with _get_shape_fh(chain_dir, 'r') as sfh:
            shapes = json.load(sfh)
        samples = {}
        for var_name, shape in shapes.items():
            var_file = os.path.join(chain_dir, var_name + '.txt')
            samples[var_name] = np.loadtxt(var_file).reshape(shape)
        trace.samples[chain] = samples
    trace.var_names = list(trace.samples[chain].keys())
    return trace


## Not opening json directory in `Text.close` and `load` for testing
## convenience
@contextmanager
def _get_shape_fh(chain_dir, mode='r'):
    fh = open(os.path.join(chain_dir, 'shapes.json'), mode)
    try:
        yield fh
    finally:
        fh.close()


def _get_chain_dirs(name):
    """Return mapping of chain number to directory"""
    return {_chain_dir_to_chain(chain_dir): chain_dir
            for chain_dir in glob.glob(os.path.join(name, 'chain-*'))}


def _chain_dir_to_chain(chain_dir):
    return int(os.path.basename(chain_dir).split('-')[1])
