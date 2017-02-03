from ..backends import base, ndarray
import h5py
import numpy as np
from contextlib import contextmanager

@contextmanager
def activator(instance):
    if isinstance(instance.samples, h5py.File):
        if instance.samples.id:  # if file is open, keep open
            yield
            return
    # if file is closed/not referenced: open, do job, then close
    instance.samples = h5py.File(instance.name, 'a')
    yield
    instance.samples.close()
    return

class HDF5(ndarray.NDArray):
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
    def chain_is_setup(self):
        with self.activate_file:
            try:
                return self.chain in self.samples.attrs['chains']
            except KeyError:
                return False

    @property
    def is_new_file(self):
        with self.activate_file:
            return len(self.samples.keys()) == 0

    @property
    def chain_dict(self):
        with self.activate_file:
            try:
                l = self.samples.attrs['chains']
                d = {k: i for i, k in enumerate(l)}
                d[None] = 0
                return d
            except KeyError:
                return {}

    @property
    def nchains(self):
        with self.activate_file:
            try:
                return len(self.samples.attrs['chains'])
            except KeyError:
                return 0

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
        with self.activate_file:
            if self.is_new_file:
                self.draws = draws
                for varname, shape in self.var_shapes.items():
                    ds = self.samples.create_dataset(varname, (1, draws) + shape, dtype=self.var_dtypes[varname],
                                                     maxshape=(None, None) + shape)
                    self.samples.attrs['chains'] = [self.chain]
            else:
                if not self.chain_is_setup:
                    self.draws = draws
                    for v in self.samples.itervalues():
                        v.resize(self.nchains+1, axis=0)
                    self.samples.attrs['chains'] = np.append(self.samples.attrs['chains'], self.chain)
                else:
                    old_draws = len(self)
                    self.draws = old_draws + draws
                    self.draw_idx = old_draws  # tag onto end of chain
                for v in self.samples.itervalues():
                    v.resize(self.draws, axis=1)


    def close(self):
        with self.activate_file:
            if self.draw_idx == self.draws:
                return
            # Remove trailing zeros if interrupted before completed all
            # draws.
            for ds in self.samples.itervalues():
                ds.resize(self.draw_idx, axis=1)

    def record(self, point):
        with self.activate_file:
            for varname, value in zip(self.varnames, self.fn(point)):
                self.samples[varname][self.chain_dict[self.chain], self.draw_idx] = value
            self.draw_idx += 1

    def get_values(self, varname, burn=0, thin=1):
        with self.activate_file:
            return self.samples[varname][self.chain_dict[self.chain], burn::thin]

    def _slice(self, idx):
        with self.activate_file:
            if idx.start is None:
                burn = 0
            else:
                burn = idx.start
            if idx.step is None:
                thin = 1
            else:
                thin = idx.step

            sliced = ndarray.NDArray(model=self.model, vars=self.vars)
            sliced.chain = self.chain_dict[self.chain]
            sliced.samples = {v: self.get_values(v, burn=burn, thin=thin)
                              for v in self.varnames}
            return sliced

    def point(self, idx):
        with self.activate_file:
            idx = int(idx)
            r = {}
            for varname, values in self.samples.iteritems():
                r[varname] = values[self.chain_dict[self.chain], idx]
            return r

    def __len__(self):
        if self.chain_is_setup:
            with self.activate_file:
                return self.samples.values()[0].shape[1]
        return 0


def load(name, model=None):
    """Load HDF5 arrays.

    Parameters
    ----------
    name : str
        Path to HDF5 arrays file
    model : Model
        If None, the model is taken from the `with` context.

    Returns
    -------
    A MultiTrace instance
    """
    straces = []
    with h5py.File(name, 'r') as file:
        chains = file.attrs['chains']
    for chain in chains:
        trace = HDF5(name, model=model)
        trace.chain = chain
        straces.append(trace)
    r = base.MultiTrace(straces)
    return r