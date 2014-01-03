"""Base backend for traces

These are the base classes for all trace backends. They define all the
required methods for sampling and value selection that should be
overridden or implementented in children classes. See the docstring for
pymc.backends for more information (includng creating custom backends).
"""
import numpy as np
from pymc.model import modelcontext


class Backend(object):

    def __init__(self, name, model=None, variables=None):
        self.name = name

        ## model attributes
        self.variables = None
        self.var_names = None
        self.var_shapes = None
        self._fn = None

        model = modelcontext(model)
        self.model = model
        if model:
            self._setup_model(model, variables)

        ## set by setup_samples
        self.chain = None
        self.trace = None

        self._draws = {}

    def _setup_model(self, model, variables):
        if variables is None:
            variables = model.unobserved_RVs
        self.variables = variables
        self.var_names = [str(var) for var in variables]
        self._fn = model.fastfn(variables)

        var_values = zip(self.var_names, self._fn(model.test_point))
        self.var_shapes = {var: value.shape
                           for var, value in var_values}

    def setup_samples(self, draws, chain):
        """Prepare structure to store traces

        Parameters
        ----------
        draws : int
            Number of sampling iterations
        chain : int
            Chain number to store trace under
        """
        self.chain = chain
        self._draws[chain] = draws

        if self.trace is None:
            self.trace = self._initialize_trace()
        trace = self.trace
        trace._draws[chain] = draws
        trace.backend = self

        trace.samples[chain] = {}
        for var_name, var_shape in self.var_shapes.items():
            trace_shape = [draws] + list(var_shape)
            trace.samples[chain][var_name] = self._create_trace(chain,
                                                                var_name,
                                                                trace_shape)

    def record(self, point, draw):
        """Record the value of the current iteration

        Parameters
        ----------
        point : dict
            Map of point values to variable names
        draw : int
            Current sampling iteration
        """
        for var_name, value in zip(self.var_names, self._fn(point)):
            self._store_value(draw,
                              self.trace.samples[self.chain][var_name],
                              value)

    def clean_interrupt(self, current_draw):
        """Clean up sampling after interruption

        Perform any clean up not taken care of by `close`. After
        KeyboardInterrupt, `sample` calls `close`, so `close` should not
        be called here.
        """
        self.trace._draws[self.chain] = current_draw

    ## Sampling methods that children must define

    def _initialize_trace(self):
        raise NotImplementedError

    def _create_trace(self, chain, var_name, shape):
        """Create trace for a variable

        Parameters
        ----------
        chain : int
            Current chain number
        var_name : str
            Name of variable
        shape : tuple
            Shape of the trace. The first element corresponds to the
            number of draws.
        """
        raise NotImplementedError

    def _store_value(self, draw, var_trace, value):
        raise NotImplementedError

    def commit(self):
        """Commit samples to backend

        This is called at set intervals during sampling.
        """
        raise NotImplementedError

    def close(self):
        """Close the database backend

        This is called after sampling has finished.
        """
        raise NotImplementedError


class Trace(object):
    """
    Parameters
    ----------
    var_names : list of strs
        Sample variables names
    backend : Backend object

    Attributes
    ----------
    backend : Backend object
    var_names
    var_shapes : dict
        Map of variables shape to variable names
    samples : dict of dicts
        Sample values keyed by chain and variable name
    nchains : int
        Number of sampling chains
    chains : list of ints
        List of sampling chain numbers
    default_chain : int
        Chain to be used if single chain requested
    active_chains : list of ints
        Values from chains to be used operations
    """
    def __init__(self, var_names, backend=None):
        self.var_names = var_names

        self.samples = {}
        self._draws = {}
        self.backend = backend
        self._active_chains = []
        self._default_chain = None

    @property
    def nchains(self):
        """Number of chains

        A chain is created for each sample call (including parallel
        threads).
        """
        return len(self.samples)

    @property
    def chains(self):
        """All chains in trace"""
        return list(self.samples.keys())

    @property
    def default_chain(self):
        """Default chain to use for operations that require one chain (e.g.,
        `point`)
        """
        if self._default_chain is None:
            return self.active_chains[-1]
        return self._default_chain

    @default_chain.setter
    def default_chain(self, value):
        self._default_chain = value

    @property
    def active_chains(self):
        """List of chains to be used. Defaults to all.
        """
        if not self._active_chains:
            return self.chains
        return self._active_chains

    @active_chains.setter
    def active_chains(self, values):
        try:
            self._active_chains = [chain for chain in values]
        except TypeError:
            self._active_chains = [values]

    def __len__(self):
        return self._draws[self.default_chain]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)

        try:
            return self.point(idx)
        except ValueError:
            pass
        except TypeError:
            pass
        return self.get_values(idx)

    ## Selection methods that children must define

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
        raise NotImplementedError

    def _slice(self, idx):
        """Slice trace object"""
        raise NotImplementedError

    def point(self, idx, chain=None):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.

        If `chain` is not specified, `default_chain` is used.
        """
        raise NotImplementedError


def merge_chains(traces):
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
    base_trace = traces[0]
    for new_trace in traces[1:]:
        new_chain = new_trace.chains[0]
        if new_chain in base_trace.samples:
            raise ValueError('Trace chain numbers conflict.')
        base_trace.samples[new_chain] = new_trace.samples[new_chain]
    return base_trace


def _squeeze_cat(results, combine, squeeze):
    """Squeeze and concatenate the results dependending on values of
    `combine` and `squeeze`"""
    if combine:
        results = np.concatenate(results)
        if not squeeze:
            results = [results]
    else:
        if squeeze and len(results) == 1:
            results = results[0]
    return results
