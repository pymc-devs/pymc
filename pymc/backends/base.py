"""Base backend for traces

See the docstring for pymc.backends for more information (includng
creating custom backends).
"""
import numpy as np
from pymc.model import modelcontext


class Backend(object):
    """Base storage class

    Parameters
    ----------
    name : str
        Name of backend.
    model : Model
        If None, the model is taken from the `with` context.
    variables : list of variable objects
        Sampling values will be stored for these variables
    """
    def __init__(self, name, model=None, variables=None):
        self.name = name

        model = modelcontext(model)
        if variables is None:
            variables = model.unobserved_RVs
        self.variables = variables
        self.var_names = [str(var) for var in variables]
        self.fn = model.fastfn(variables)

        ## get variable shapes. common enough that I think most backends
        ## will use this
        var_values = zip(self.var_names, self.fn(model.test_point))
        self.var_shapes = {var: value.shape
                           for var, value in var_values}
        self.chain = None
        self.trace = None

    def setup(self, draws, chain):
        """Perform chain-specific setup

        draws : int
            Expected number of draws
        chain : int
            chain number
        """
        pass

    def record(self, point):
        """Record results of a sampling iteration

        point : dict
            Values mappled to variable names
        """
        raise NotImplementedError

    def close(self):
        """Close the database backend

        This is called after sampling has finished.
        """
        pass


class Trace(object):
    """
    Parameters
    ----------
    var_names : list of strs
        Sample variables names
    backend : Backend object

    Attributes
    ----------
    var_names
    backend : Backend object
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
        self.backend = backend
        self._active_chains = []
        self._default_chain = None

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

    @property
    def nchains(self):
        """Number of chains

        A chain is created for each sample call (including parallel
        threads).
        """
        return len(self.chains)

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

    @property
    def chains(self):
        """All chains in trace"""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

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
        raise NotImplementedError


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
