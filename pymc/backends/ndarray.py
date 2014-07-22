"""NumPy array trace backend

Store sampling values in memory as a NumPy array.
"""
import numpy as np
from ..backends import base


class NDArray(base.BaseTrace):
    """NDArray trace object

    Parameters
    ----------
    name : str
        Name of backend. This has no meaning for the NDArray backend.
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """
    def __init__(self, name=None, model=None, vars=None):
        super(NDArray, self).__init__(name, model, vars)
        self.draw_idx = 0
        self.draws = None
        self.samples = {}

    ## Sampling methods

    def setup(self, draws, chain):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        self.chain = chain
        if self.samples:  # Concatenate new array if chain is already present.
            old_draws = len(self)
            self.draws = old_draws + draws
            self.draws_idx = old_draws
            for varname, shape in self.var_shapes.items():
                old_trace = self.samples[varname]
                new_trace = np.zeros((draws, ) + shape)
                self.samples[varname] = np.concatenate((old_trace, new_trace),
                                                       axis=0)
        else:  # Otherwise, make array of zeros for each variable.
            self.draws = draws
            for varname, shape in self.var_shapes.items():
                self.samples[varname] = np.zeros((draws, ) + shape)

    def record(self, point):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point : dict
            Values mapped to variable names
        """
        for varname, value in zip(self.varnames, self.fn(point)):
            self.samples[varname][self.draw_idx] = value
        self.draw_idx += 1

    def close(self):
        if self.draw_idx == self.draws:
            return
        ## Remove trailing zeros if interrupted before completed all
        ## draws.
        self.samples = {var: trace[:self.draw_idx]
                        for var, trace in self.samples.items()}

    ## Selection methods

    def __len__(self):
        if not self.samples:  # `setup` has not been called.
            return 0
        varname = self.varnames[0]
        return self.samples[varname].shape[0]

    def get_values(self, varname, burn=0, thin=1):
        """Get values from trace.

        Parameters
        ----------
        varname : str
        burn : int
        thin : int

        Returns
        -------
        A NumPy array
        """
        return self.samples[varname][burn::thin]

    def _slice(self, idx):
        sliced = NDArray(model=self.model, vars=self.vars)
        sliced.chain = self.chain
        sliced.samples = {varname: values[idx]
                          for varname, values in self.samples.items()}
        return sliced

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.
        """
        idx = int(idx)
        return {varname: values[idx]
                for varname, values in self.samples.items()}
