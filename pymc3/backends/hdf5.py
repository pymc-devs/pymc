from ndarray import NDArray
import h5py
import numpy as np
from contextlib import contextmanager

@contextmanager
def activator(instance):
    with instance.h5file as f:
        instance.samples = f
        yield
    instance.samples = None


class HDF5(NDArray):
    """HDF5 trace object

        Parameters
        ----------
        name : str
            Name of backend. This has no meaning for the HDF5 backend.
        model : Model
            If None, the model is taken from the `with` context.
        vars : list of variables
            Sampling values will be stored for these variables. If None,
            `model.unobserved_RVs` is used.
        """
    @property
    def activate_file(self):
        return activator(self)

    @property
    def h5file(self):
        return h5py.File(self.name, 'a')

    @property
    def is_setup(self):
        with self.h5file as f:
            return 'chains' in f.attrs.keys()

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
        if self.is_setup:  # Concatenate new array if chain is already present.
            old_draws = len(self)
            self.draws = old_draws + draws
            self.draws_idx = old_draws
            with self.h5file as file:
                chs = file.attrs['chains']
                if self.chain not in chs:
                    file.attrs['chains'] = np.append(chs, self.chain)
                    for v in file.itervalues():
                        v.resize(len(chs), axis=0)
                for v in file.itervalues():
                    v.resize(self.draws, axis=1)
                self.chain_dict = {c: i for i, c in enumerate(chs)}

        else:  # Otherwise, make array of zeros for each variable.
            self.draws = draws
            with self.h5file as file:
                file.attrs['chains'] = [self.chain]
                for varname, shape in self.var_shapes.items():
                    ds = file.create_dataset(varname, (1, draws) + shape, dtype=self.var_dtypes[varname], maxshape=(None, None)+shape)
                self.chain_dict = {self.chain: 0}

    def close(self):
        with self.activate_file:
            if self.draw_idx == self.draws:
                return
            # Remove trailing zeros if interrupted before completed all
            # draws.
            for ds in self.samples.itervalues():
                self.samples.resize(self.draws_idx+1, axis=1)

    def record(self, point):
        with self.activate_file:
            for varname, value in zip(self.varnames, self.fn(point)):
                self.samples[varname][self.chain_dict[self.chain], self.draw_idx] = value
            self.draw_idx += 1

    def get_values(self, varname, burn=0, thin=1):
        with self.activate_file:
            return super(HDF5, self).get_values(varname, burn, thin)

    def _slice(self, idx):
        with self.activate_file:
            return super(HDF5, self)._slice(idx)

    def point(self, idx):
        with self.activate_file:
            return super(HDF5, self).point(idx)
