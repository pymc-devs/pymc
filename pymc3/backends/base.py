"""Base backend for traces

See the docstring for pymc3.backends for more information (including
creating custom backends).
"""
import numpy as np
from ..model import modelcontext, TransformedRV


class BackendError(Exception):
    pass


class BaseTrace(object):
    """Base trace object

    Parameters
    ----------
    name : str
        Name of backend
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """

    def __init__(self, name, model=None, vars=None):
        self.name = name

        model = modelcontext(model)
        self.model = model
        if vars is None:
            vars = model.unobserved_RVs
        self.vars = vars
        self.varnames = [var.name for var in vars]
        self.fn = model.fastfn(vars)

        # Get variable shapes. Most backends will need this
        # information.
        var_values = list(zip(self.varnames, self.fn(model.test_point)))
        self.var_shapes = {var: value.shape
                           for var, value in var_values}
        self.var_dtypes = {var: value.dtype
                           for var, value in var_values}
        self.chain = None

    # Sampling methods

    def setup(self, draws, chain):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        pass

    def record(self, point):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point : dict
            Values mapped to variable names
        """
        raise NotImplementedError

    def close(self):
        """Close the database backend.

        This is called after sampling has finished.
        """
        pass

    # Selection methods

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)

        try:
            return self.point(int(idx))
        except (ValueError, TypeError):  # Passed variable or variable name.
            raise ValueError('Can only index with slice or integer')

    def __len__(self):
        raise NotImplementedError

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
        raise NotImplementedError

    def _slice(self, idx):
        """Slice trace object."""
        raise NotImplementedError

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.
        """
        raise NotImplementedError


class MultiTrace(object):
    """Main interface for accessing values from MCMC results

    The core method to select values is `get_values`. Values can also be
    accessed by indexing the MultiTrace object. Indexing can behave in
    three ways:

    1. Indexing with a variable or variable name (str) returns all
       values for that variable, combining values for all chains.

       >>> trace[varname]

       Slicing after the variable name can be used to burn and thin
       the samples.

       >>> trace[varname, 1000:]

       For convenience during interactive use, values can also be
       accessed using the variable an attribute.

       >>> trace.varname

    2. Indexing with an integer returns a dictionary with values for
       each variable at the given index (corresponding to a single
       sampling iteration).

    3. Slicing with a range returns a new trace with the number of draws
       corresponding to the range.

    For any methods that require a single trace (e.g., taking the length
    of the MultiTrace instance, which returns the number of draws), the
    trace with the highest chain number is always used.
    """

    def __init__(self, straces):
        self._straces = {}
        for strace in straces:
            if strace.chain in self._straces:
                raise ValueError("Chains are not unique.")
            self._straces[strace.chain] = strace

    def __repr__(self):
        template = '<{}: {} chains, {} iterations, {} variables>'
        return template.format(self.__class__.__name__,
                               self.nchains, len(self), len(self.varnames))

    @property
    def nchains(self):
        return len(self._straces)

    @property
    def chains(self):
        return list(sorted(self._straces.keys()))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)

        try:
            return self.point(int(idx))
        except (ValueError, TypeError):  # Passed variable or variable name.
            pass

        if isinstance(idx, tuple):
            var, vslice = idx
            burn, thin = vslice.start, vslice.step
            if burn is None:
                burn = 0
            if thin is None:
                thin = 1
        else:
            var = idx
            burn, thin = 0, 1
        return self.get_values(var, burn=burn, thin=thin)

    _attrs = set(['_straces', 'varnames', 'chains'])

    def __getattr__(self, name):
        # Avoid infinite recursion when called before __init__
        # variables are set up (e.g., when pickling).
        if name in self._attrs:
            raise AttributeError

        if name in self.varnames:
            return self[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __len__(self):
        chain = self.chains[-1]
        return len(self._straces[chain])

    @property
    def varnames(self):
        chain = self.chains[-1]
        return self._straces[chain].varnames

    def get_values(self, varname, burn=0, thin=1, combine=True, chains=None,
                   squeeze=True):
        """Get values from traces.

        Parameters
        ----------
        varname : str
        burn : int
        thin : int
        combine : bool
            If True, results from `chains` will be concatenated.
        chains : int or list of ints
            Chains to retrieve. If None, all chains are used. A single
            chain value can also be given.
        squeeze : bool
            Return a single array element if the resulting list of
            values only has one element. If False, the result will
            always be a list of arrays, even if `combine` is True.

        Returns
        -------
        A list of NumPy arrays or a single NumPy array (depending on
        `squeeze`).
        """
        if chains is None:
            chains = self.chains
        varname = str(varname)
        try:
            results = [self._straces[chain].get_values(varname, burn, thin)
                       for chain in chains]
        except TypeError:  # Single chain passed.
            results = [self._straces[chains].get_values(varname, burn, thin)]
        return _squeeze_cat(results, combine, squeeze)

    def _slice(self, idx):
        """Return a new MultiTrace object sliced according to `idx`."""
        new_traces = [trace._slice(idx) for trace in self._straces.values()]
        return MultiTrace(new_traces)

    def point(self, idx, chain=None):
        """Return a dictionary of point values at `idx`.

        Parameters
        ----------
        idx : int
        chain : int
            If a chain is not given, the highest chain number is used.
        """
        if chain is None:
            chain = self.chains[-1]
        return self._straces[chain].point(idx)


def merge_traces(mtraces):
    """Merge MultiTrace objects.

    Parameters
    ----------
    mtraces : list of MultiTraces
        Each instance should have unique chain numbers.

    Raises
    ------
    A ValueError is raised if any traces have overlapping chain numbers.

    Returns
    -------
    A MultiTrace instance with merged chains
    """
    base_mtrace = mtraces[0]
    for new_mtrace in mtraces[1:]:
        for new_chain, strace in new_mtrace._straces.items():
            if new_chain in base_mtrace._straces:
                raise ValueError("Chains are not unique.")
            base_mtrace._straces[new_chain] = strace
    return base_mtrace


def _squeeze_cat(results, combine, squeeze):
    """Squeeze and concatenate the results depending on values of
    `combine` and `squeeze`."""
    if combine:
        results = np.concatenate(results)
        if not squeeze:
            results = [results]
    else:
        if squeeze and len(results) == 1:
            results = results[0]
    return results
