"""NumPy trace backend

Store sampling values in memory as a NumPy array.
"""
import numpy as np
from pymc.backends import base


class NDArray(base.Backend):
    """NDArray storage

    Parameters
    ----------
    name : str
        Name of backend.
    model : Model
        If None, the model is taken from the `with` context.
    variables : list of variable objects
        Sampling values will be stored for these variables
    """
    ## make `name` an optional argument for NDArray
    def __init__(self, name=None, model=None, variables=None):
        super(NDArray, self).__init__(name, model, variables)

        self.trace = Trace(self.var_names)
        self.draw_idx = 0
        self.draws = None

    def setup(self, draws, chain):
        """Perform chain-specific setup

        draws : int
            Expected number of draws
        chain : int
            chain number
        """
        self.draws = draws
        self.chain = chain
        ## Make array of zeros for each variable
        var_arrays = {}
        for var_name, shape in self.var_shapes.items():
            var_arrays[var_name] = np.zeros((draws, ) + shape)
        self.trace.samples[chain] = var_arrays

    def record(self, point):
        """Record results of a sampling iteration

        point : dict
            Values mappled to variable names
        """
        for var_name, value in zip(self.var_names, self.fn(point)):
            self.trace.samples[self.chain][var_name][self.draw_idx] = value
        self.draw_idx += 1

    def close(self):
        if self.draw_idx == self.draws - 1:
            return
        ## Remove trailing zeros if interrupted before completed all draws
        traces = self.trace.samples[self.chain]
        traces = {var: trace[:self.draw_idx] for var, trace in traces.items()}
        self.trace.samples[self.chain] = traces


class Trace(base.Trace):

    __doc__ = 'NumPy array trace\n' + base.Trace.__doc__

    def __init__(self, var_names, backend=None):
        super(Trace, self).__init__(var_names, backend)
        self.samples = {}  # chain -> var name -> values

    def __len__(self):
        var_name = self.var_names[0]
        return self.samples[self.default_chain][var_name].shape[0]

    @property
    def chains(self):
        """All chains in trace"""
        return list(self.samples.keys())

    def get_values(self, var_name, burn=0, thin=1, combine=False, chains=None,
                   squeeze=True):
        """Get values from samples

        Parameters
        ----------
        var_name : str
        burn : int
        thin : int
        combine : bool
            If True, results from all chains will be concatenated.
        chains : list
            Chains to retrieve. If None, `active_chains` is used.
        squeeze : bool
            If `combine` is False, return a single array element if the
            resulting list of values only has one element (even if
            `combine` is True).

        Returns
        -------
        A list of NumPy array of values
        """
        if chains is None:
            chains = self.active_chains

        var_name = str(var_name)
        results = (self.samples[chain][var_name] for chain in chains)
        results = [arr[burn::thin] for arr in results]

        return base._squeeze_cat(results, combine, squeeze)

    def _slice(self, idx):
        sliced = Trace(self.var_names)
        sliced.backend = self.backend
        sliced._active_chains = sliced._active_chains
        sliced._default_chain = sliced._default_chain

        sliced.samples = {}
        for chain, trace in self.samples.items():
            sliced_values = {var_name: values[idx]
                             for var_name, values in trace.items()}
            sliced.samples[chain] = sliced_values
        return sliced

    def point(self, idx, chain=None):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.

        If `chain` is not specified, `default_chain` is used.
        """
        if chain is None:
            chain = self.default_chain
        return {var_name: values[idx]
                for var_name, values in self.samples[chain].items()}

    def merge_chains(self, traces):
        """Merge chains from trace instances

        Parameters
        ----------
        traces : list
            Backend trace instances. Each instance should have only one
            chain, and all chain numbers should be unique.

        Raises
        ------
        ValueError is raised if any traces have the same current chain
        number.

        Returns
        -------
        Backend instance with merge chains
        """
        for new_trace in traces:
            new_chain = new_trace.chains[0]
            if new_chain in self.samples:
                raise ValueError('Trace chain numbers conflict.')
            self.samples[new_chain] = new_trace.samples[new_chain]
