import collections
import pymc3 as pm
import theano
from pymc3.backends import NDArray, base
from pymc3.backends.base import MultiTrace
from pymc3.blocking import Compose, VarMap

from pymc3.step_methods.arraystep import ArrayStepShared
from theano.gradient import np

import emcee

LARGENUMBER = 100000

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


class EnsembleArrayOrdering(object):
    """
    An ordering for an array space
    """

    def __init__(self, vars, nparticles):
        self.vmap = []
        dim = 0

        for var in vars:
            slc = slice(dim, dim + var.dsize)
            self.vmap.append(VarMap(str(var), slc, var.dshape, var.dtype))
            dim += var.dsize

        self.dimensions = (nparticles, dim)

    @property
    def nparticles(self):
        return self.dimensions[0]


class EnsembleDictToArrayBijection(object):
    """
    A mapping between a dict space and an array space
    """

    def __init__(self, ordering, dpoint):
        self.ordering = ordering
        self.dpt = dpoint

        # determine smallest float dtype that will fit all data
        if all([x.dtyp == 'float16' for x in ordering.vmap]):
            self.array_dtype = 'float16'
        elif all([x.dtyp == 'float32' for x in ordering.vmap]):
            self.array_dtype = 'float32'
        else:
            self.array_dtype = 'float64'

    def map(self, dpt):
        """
        Maps value from dict space to array space

        Parameters
        ----------
        dpt : dict
        """
        apt = np.empty(self.ordering.dimensions, dtype=self.array_dtype)
        for var, slc, shp, dty in self.ordering.vmap:
            apt[:, slc] = dpt[var].reshape(self.ordering.nparticles, -1)
        return apt

    def rmap(self, apt):
        """
        Maps value from array space to dict space

        Parameters
        ----------
        apt : array
        """
        dpt = self.dpt.copy()

        for var, slc, shp, dtyp in self.ordering.vmap:
            dpt[var] = np.atleast_2d(apt)[:, slc].reshape(self.ordering.nparticles, *shp).astype(dtyp)

        return dpt

    def mapf(self, f):
        """
         function f : DictSpace -> T to ArraySpace -> T

        Parameters
        ----------

        f : dict -> T

        Returns
        -------
        f : array -> T
        """
        return Compose(f, self.rmap)


class ExternalEnsembleStepShared(ArrayStepShared):
    """Ensemble version of ArrayStepShared for exportation to external step functions"""

    def __init__(self, vars, shared, blocked=True, nparticles=None):
        """
        Parameters
        ----------
        vars : list of sampling variables
        shared : dict of theano variable -> shared variable
        blocked : Boolean (default True)
        """
        self.vars = vars
        self.dimensions = sum(v.dsize for v in vars)
        if nparticles is None:
            nparticles = self.dimensions * 2
        self.nparticles = nparticles
        self.ordering = EnsembleArrayOrdering(vars, nparticles)
        self.shared = {str(var): shared for var, shared in shared.items()}
        self.blocked = blocked

    def step(self, points):
        for var, share in self.shared.items():
            share.set_value(points[var])

        bij = EnsembleDictToArrayBijection(self.ordering, points)

        if self.generates_stats:
            apoints, stats = self.astep(bij.map(points))
            return bij.rmap(apoints), stats
        else:
            apoints = self.astep(bij.map(points))
            return bij.rmap(apoints)


def logpostfn(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f


class AffineInvariantEnsemble(ExternalEnsembleStepShared):
    """
    Affine Invariant ensemble sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    tune : bool
        Flag for tuning. Defaults to True.
    tune_interval : int
        The frequency of tuning. Defaults to 100 iterations.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode :  string or `Mode` instance.
        compilation mode passed to Theano functions
    """

    default_blocked = True
    generates_stats = False

    def __init__(self, vars=None, nparticles=None, tune=True, tune_interval=100, model=None, mode=None, **kwargs):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # Determine type of variables
        self.discrete = np.concatenate([[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)
        self.logp = logpostfn(model.logpt, vars, shared)
        self.emcee_kwargs = kwargs.pop('emcee_kwargs', {})
        super(AffineInvariantEnsemble, self).__init__(vars, shared, nparticles=nparticles)
        self.sample_generator = None

    def setup_step(self, point_array):
        # very hacky way to make emcee keep going without `iterations` information
        self.emcee_sampler = emcee.EnsembleSampler(self.nparticles, self.dimensions, self.logp, **self.emcee_kwargs)
        self.sample_generator = self.emcee_sampler.sample(point_array, storechain=False, iterations=100000)

    def astep(self, point_array):
        if self.sample_generator is None:
            self.setup_step(point_array)
        try:
            q, lnprob, state = next(self.sample_generator)
        except StopIteration:
            self.sample_generator = None
            return self.astep(point_array)
        return q


def get_random_starters(nparticles, model):
    return {v.name: np.asarray([v.distribution.random() for i in range(nparticles)]) for v in model.vars}


def build_start_point(nparticles, method='random', model=None, **kwargs):
    if method == 'random':
        return get_random_starters(nparticles, model)
    else:
        start, _ = pm.init_nuts(method, nparticles, model=model, **kwargs)
        return {v: start.get_values(v) for v in start.varnames}


def sample(draws=500, step=AffineInvariantEnsemble, init='random', n_init=200000, start=None,
           trace=None, nparticles=None, tune=500,
           step_kwargs=None, progressbar=True, model=None, random_seed=-1,
           live_plot=False, discard_tuned_samples=True, **kwargs):

    model = pm.modelcontext(model)

    vars = model.vars
    vars = pm.inputvars(vars)

    sampler = step(vars, nparticles, tune>0, tune, model, **kwargs)
    nparticles = sampler.nparticles
    if trace is None:
        trace = EnsembleNDArray('mcmc', model, vars, nparticles)
    elif not isinstance(trace, EnsembleNDArray):
        raise TypeError("trace must be of type EnsembleNDArray")

    start = build_start_point(nparticles, init, model)
    trace = pm.sample(draws, sampler, init, n_init, start, trace, 0, 1, tune, None, None, progressbar, model,
                      random_seed, live_plot, discard_tuned_samples, **kwargs)

    traces = []
    for i in range(nparticles):
        tr = NDArray('mcmc', model, vars)
        tr.setup(len(trace), i)
        for varname in trace.varnames:
            tr.samples[varname] = trace[varname][:, i]
            tr.draw_idx = len(trace)
        traces.append(tr)
    return MultiTrace(traces)
