"""NumPy trace backend

Store sampling values in memory as a NumPy array.
"""
import numpy as np
from pymc.backends import base


class NDArray(base.Backend):

    ## make `name` an optional argument for NDArray
    def __init__(self, name=None, model=None, variables=None):
        super(NDArray, self).__init__(name, model, variables)

    def _initialize_trace(self):
        return Trace(self.var_names)

    def _create_trace(self, chain, var_name, shape):
        return np.zeros(shape)

    def _store_value(self, draw, var_trace, value):
        var_trace[draw] = value

    def commit(self):
        pass

    def close(self):
        pass

    def clean_interrupt(self, current_draw):
        super(NDArray, self).clean_interrupt(current_draw)
        traces = self.trace.samples[self.chain]
        ## get rid of trailing zeros
        traces = {var: trace[:current_draw] for var, trace in traces.items()}
        self.trace.samples[self.chain] = traces


class Trace(base.Trace):

    __doc__ = 'NumPy array trace\n' + base.Trace.__doc__

    def __len__(self):
        try:
            return super(Trace, self).__len__()
        except KeyError:
            var_name = self.var_names[0]
            draws = self.samples[self.default_chain][var_name].shape[0]
            self._draws[self.default_chain] = draws
            return draws

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
        sliced._draws = {}
        for chain, trace in self.samples.items():
            sliced_values = {var_name: values[idx]
                             for var_name, values in trace.items()}
            sliced.samples[chain] = sliced_values
            sliced._draws[chain] = sliced_values[self.var_names[0]].shape[0]
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
