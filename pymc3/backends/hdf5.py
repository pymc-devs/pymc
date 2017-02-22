from ..backends import base, ndarray
import h5py
from contextlib import contextmanager

@contextmanager
def activator(instance):
    if isinstance(instance.hdf5_file, h5py.File):
        if instance.hdf5_file.id:  # if file is open, keep open
            yield
            return
    # if file is closed/not referenced: open, do job, then close
    instance.hdf5_file = h5py.File(instance.name, 'a')
    yield
    instance.hdf5_file.close()
    return


class HDF5(base.BaseTrace):
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

    supports_sampler_stats = True

    def __init__(self, name=None, model=None, vars=None):
        self.hdf5_file = None
        self.draw_idx = 0
        self.draws = None
        super(HDF5, self).__init__(name, model, vars)

    def _get_sampler_stats(self, varname, sampler_idx, burn, thin):
        with self.activate_file:
            return self.stats[str(sampler_idx)][varname][burn::thin]

    @property
    def activate_file(self):
        return activator(self)

    @property
    def samples(self):
        g = self.hdf5_file.require_group(str(self.chain))
        if 'name' not in g.attrs:
            g.attrs['name'] = self.chain
        return g.require_group('samples')

    @property
    def stats(self):
        g = self.hdf5_file.require_group(str(self.chain))
        if 'name' not in g.attrs:
            g.attrs['name'] = self.chain
        return g.require_group('stats')

    @property
    def chains(self):
        with self.activate_file:
            return [v.attrs['name'] for v in self.hdf5_file.values()]

    @property
    def is_new_file(self):
        with self.activate_file:
            return len(self.samples.keys()) == 0

    @property
    def chain_is_setup(self):
        with self.activate_file:
            return self.chain in self.chains

    @property
    def nchains(self):
        with self.activate_file:
            return len(self.chains)

    @property
    def records_stats(self):
        with self.activate_file:
            return self.hdf5_file.attrs['records_stats']

    @records_stats.setter
    def records_stats(self, v):
        with self.activate_file:
            self.hdf5_file.attrs['records_stats'] = bool(v)

    def _resize(self, n):
        for v in self.samples.values():
            v.resize(n, axis=0)
        for key, group in self.stats.items():
            for statds in group.values():
                statds.resize((n, ))

    @property
    def sampler_vars(self):
        with self.activate_file:
            l = []
            for i, sampler in sorted(self.stats.items(), key=lambda x: int(x[0])):
                d = {}
                for varname, ds in sampler.items():
                    d[varname] = ds.dtype
                l.append(d)
        if not l:
            return None
        return l

    @sampler_vars.setter
    def sampler_vars(self, values):
        with self.activate_file:
            if values is None:
                self.records_stats = False
                return

            for i, sampler in enumerate(values):
                data = self.stats.require_group(str(i))
                if not data.keys():  # no pre-recorded stats
                    for varname, dtype in sampler.items():
                        if varname not in data:
                            data.create_dataset(varname, (self.draws,), dtype=dtype, maxshape=(None,))
                elif data.keys() != sampler.keys():
                    raise ValueError(
                        "Sampler vars can't change, names incompatible: {} != {}".format(data.keys(), sampler.keys()))
            self.records_stats = True

    def setup(self, draws, chain, sampler_vars=None):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        sampler_vars : list of dicts
            Names and dtypes of the variables that are
            exported by the samplers.
        """
        self.chain = chain
        with self.activate_file:
            for varname, shape in self.var_shapes.items():
                if varname not in self.samples:
                    self.samples.create_dataset(name=varname, shape=(draws, ) + shape,
                                                dtype=self.var_dtypes[varname],
                                                maxshape=(None, ) + shape)
            self.draw_idx = len(self)
            self.draws = self.draw_idx + draws
            self._set_sampler_vars(sampler_vars)
            self._is_base_setup = True
            self._resize(self.draws)


    def close(self):
        with self.activate_file:
            if self.draw_idx == self.draws:
                return
            # Remove trailing zeros if interrupted before completed all
            # draws.
            self._resize(self.draw_idx)

    def record(self, point, sampler_stats=None):
        with self.activate_file:
            for varname, value in zip(self.varnames, self.fn(point)):
                self.samples[varname][self.draw_idx] = value

            if self.records_stats and sampler_stats is None:
                raise ValueError("Expected sampler_stats")
            if not self.records_stats and sampler_stats is not None:
                raise ValueError("Unknown sampler_stats")
            if sampler_stats is not None:
                for i, vars in enumerate(sampler_stats):
                    data = self.stats[str(i)]
                    for key, val in vars.items():
                        data[key][self.draw_idx] = val

            self.draw_idx += 1

    def get_values(self, varname, burn=0, thin=1):
        with self.activate_file:
            return self.samples[varname][burn::thin]

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
            sliced.chain = self.chain
            sliced.samples = {v: self.get_values(v, burn=burn, thin=thin)
                              for v in self.varnames}
            return sliced

    def point(self, idx):
        with self.activate_file:
            idx = int(idx)
            r = {}
            for varname, values in self.samples.items():
                r[varname] = values[idx]
            return r

    def __len__(self):
        if self.chain_is_setup:
            return self.draw_idx
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
    for chain in HDF5(name, model=model).chains:
        trace = HDF5(name, model=model)
        trace.chain = chain
        straces.append(trace)
    return base.MultiTrace(straces)