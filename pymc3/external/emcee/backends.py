from pymc3.backends import NDArray, base
from theano.gradient import np


class EnsembleNDArray(NDArray):
    def __init__(self, name=None, model=None, vars=None, nparticles=None):
        super(EnsembleNDArray, self).__init__(name, model, vars)
        self.nparticles = nparticles

    def setup(self, draws, chain, sampler_vars=None):
        base.BaseTrace.setup(self, draws, chain, sampler_vars)

        self.chain = chain
        if self.samples:  # Concatenate new array if chain is already present.
            old_draws = len(self)
            self.draws = old_draws + draws
            self.draws_idx = old_draws
            for varname, shape in self.var_shapes.items():
                old_var_samples = self.samples[varname]
                new_var_samples = np.zeros((draws, self.nparticles) + shape,
                                           self.var_dtypes[varname])
                self.samples[varname] = np.concatenate((old_var_samples,
                                                        new_var_samples),
                                                       axis=0)
        else:  # Otherwise, make array of zeros for each variable.
            self.draws = draws
            for varname, shape in self.var_shapes.items():
                self.samples[varname] = np.zeros((draws, self.nparticles) + shape,
                                                 dtype=self.var_dtypes[varname])

        if sampler_vars is None:
            return

        if self._stats is None:
            self._stats = []
            for sampler in sampler_vars:
                data = dict()
                self._stats.append(data)
                for varname, dtype in sampler.items():
                    data[varname] = np.zeros((self.nparticles, draws), dtype=dtype)
        else:
            for data, vars in zip(self._stats, sampler_vars):
                if vars.keys() != data.keys():
                    raise ValueError("Sampler vars can't change")
                old_draws = len(self)
                for varname, dtype in vars.items():
                    old = data[varname]
                    new = np.zeros((draws, self.nparticles), dtype=dtype)
                    data[varname] = np.concatenate([old, new])

    def multi_fn(self, point):
        l = [self.fn({k: point[k][i] for k in self.varnames}) for i in range(self.nparticles)]
        return map(np.asarray, zip(*l))


    def record(self, point, sampler_stats=None):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point : dict
            Values mapped to variable names
        """
        for varname, value in zip(self.varnames, self.multi_fn(point)):
            self.samples[varname][self.draw_idx] = value

        if self._stats is not None and sampler_stats is None:
            raise ValueError("Expected sampler_stats")
        if self._stats is None and sampler_stats is not None:
            raise ValueError("Unknown sampler_stats")
        if sampler_stats is not None:
            for data, vars in zip(self._stats, sampler_stats):
                for key, val in vars.items():
                    data[key][self.draw_idx] = val
        self.draw_idx += 1

    def _get_sampler_stats(self, varname, sampler_idx, burn, thin):
        return self._stats[sampler_idx][varname][burn::thin]

    def close(self):
        if self.draw_idx == self.draws:
            return
        # Remove trailing zeros if interrupted before completed all
        # draws.
        self.samples = {var: vtrace[:self.draw_idx]
                        for var, vtrace in self.samples.items()}
        if self._stats is not None:
            self._stats = [{var: trace[:self.draw_idx] for var, trace in stats.items()}
                           for stats in self._stats]

    # Selection methods

    def __len__(self):
        if not self.samples:  # `setup` has not been called.
            return 0
        return self.draw_idx

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
        # Slicing directly instead of using _slice_as_ndarray to
        # support stop value in slice (which is needed by
        # iter_sample).

        # Only the first `draw_idx` value are valid because of preallocation
        idx = slice(*idx.indices(len(self)))

        sliced = NDArray(model=self.model, vars=self.vars)
        sliced.chain = self.chain
        sliced.samples = {varname: values[idx]
                          for varname, values in self.samples.items()}
        sliced.sampler_vars = self.sampler_vars
        sliced.draw_idx = (idx.stop - idx.start) // idx.step

        if self._stats is None:
            return sliced
        sliced._stats = []
        for vars in self._stats:
            var_sliced = {}
            sliced._stats.append(var_sliced)
            for key, vals in vars.items():
                var_sliced[key] = vals[idx]

        return sliced

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variable names as keys.
        """
        idx = int(idx)
        return {varname: values[idx]
                for varname, values in self.samples.items()}

