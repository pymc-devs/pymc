#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Base backend for traces

See the docstring for pymc3.backends for more information
"""
import itertools as itl
import logging
import warnings

from abc import ABC
from typing import List

import numpy as np
import theano.tensor as tt

from pymc3.backends.report import SamplerReport, merge_reports
from pymc3.model import modelcontext
from pymc3.util import get_var_name

logger = logging.getLogger("pymc3")


class BackendError(Exception):
    pass


class BaseTrace(ABC):
    """Base trace object

    Parameters
    ----------
    name: str
        Name of backend
    model: Model
        If None, the model is taken from the `with` context.
    vars: list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    test_point: dict
        use different test point that might be with changed variables shapes
    """

    supports_sampler_stats = False

    def __init__(self, name, model=None, vars=None, test_point=None):
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
        if test_point is None:
            test_point = model.test_point
        else:
            test_point_ = model.test_point.copy()
            test_point_.update(test_point)
            test_point = test_point_
        var_values = list(zip(self.varnames, self.fn(test_point)))
        self.var_shapes = {var: value.shape for var, value in var_values}
        self.var_dtypes = {var: value.dtype for var, value in var_values}
        self.chain = None
        self._is_base_setup = False
        self.sampler_vars = None
        self._warnings = []

    def _add_warnings(self, warnings):
        self._warnings.extend(warnings)

    # Sampling methods

    def _set_sampler_vars(self, sampler_vars):
        if sampler_vars is not None and not self.supports_sampler_stats:
            raise ValueError("Backend does not support sampler stats.")

        if self._is_base_setup and self.sampler_vars != sampler_vars:
            raise ValueError("Can't change sampler_vars")

        if sampler_vars is None:
            self.sampler_vars = None
            return

        dtypes = {}
        for stats in sampler_vars:
            for key, dtype in stats.items():
                if dtypes.setdefault(key, dtype) != dtype:
                    raise ValueError("Sampler statistic %s appears with " "different types." % key)

        self.sampler_vars = sampler_vars

    # pylint: disable=unused-argument
    def setup(self, draws, chain, sampler_vars=None) -> None:
        """Perform chain-specific setup.

        Parameters
        ----------
        draws: int
            Expected number of draws
        chain: int
            Chain number
        sampler_vars: list of dictionaries (name -> dtype), optional
            Diagnostics / statistics for each sampler. Before passing this
            to a backend, you should check, that the `supports_sampler_state`
            flag is set.
        """
        self._set_sampler_vars(sampler_vars)
        self._is_base_setup = True

    def record(self, point, sampler_states=None):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point: dict
            Values mapped to variable names
        sampler_states: list of dicts
            The diagnostic values for each sampler
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
            raise ValueError("Can only index with slice or integer")

    def __len__(self):
        raise NotImplementedError

    def get_values(self, varname, burn=0, thin=1):
        """Get values from trace.

        Parameters
        ----------
        varname: str
        burn: int
        thin: int

        Returns
        -------
        A NumPy array
        """
        raise NotImplementedError

    def get_sampler_stats(self, stat_name, sampler_idx=None, burn=0, thin=1):
        """Get sampler statistics from the trace.

        Parameters
        ----------
        stat_name: str
        sampler_idx: int or None
        burn: int
        thin: int

        Returns
        -------
        If the `sampler_idx` is specified, return the statistic with
        the given name in a numpy array. If it is not specified and there
        is more than one sampler that provides this statistic, return
        a numpy array of shape (m, n), where `m` is the number of
        such samplers, and `n` is the number of samples.
        """
        if not self.supports_sampler_stats:
            raise ValueError("This backend does not support sampler stats")

        if sampler_idx is not None:
            return self._get_sampler_stats(stat_name, sampler_idx, burn, thin)

        sampler_idxs = [i for i, s in enumerate(self.sampler_vars) if stat_name in s]
        if not sampler_idxs:
            raise KeyError("Unknown sampler stat %s" % stat_name)

        vals = np.stack(
            [self._get_sampler_stats(stat_name, i, burn, thin) for i in sampler_idxs], axis=-1
        )
        if vals.shape[-1] == 1:
            return vals[..., 0]
        else:
            return vals

    def _get_sampler_stats(self, stat_name, sampler_idx, burn, thin):
        """Get sampler statistics."""
        raise NotImplementedError()

    def _slice(self, idx):
        """Slice trace object."""
        raise NotImplementedError()

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.
        """
        raise NotImplementedError()

    @property
    def stat_names(self):
        if self.supports_sampler_stats:
            names = set()
            for vars in self.sampler_vars or []:
                names.update(vars.keys())
            return names
        else:
            return set()


class MultiTrace:
    """Main interface for accessing values from MCMC results.

    The core method to select values is `get_values`. The method
    to select sampler statistics is `get_sampler_stats`. Both kinds of
    values can also be accessed by indexing the MultiTrace object.
    Indexing can behave in four ways:

    1. Indexing with a variable or variable name (str) returns all
       values for that variable, combining values for all chains.

       >>> trace[varname]

       Slicing after the variable name can be used to burn and thin
       the samples.

       >>> trace[varname, 1000:]

       For convenience during interactive use, values can also be
       accessed using the variable as an attribute.

       >>> trace.varname

    2. Indexing with an integer returns a dictionary with values for
       each variable at the given index (corresponding to a single
       sampling iteration).

    3. Slicing with a range returns a new trace with the number of draws
       corresponding to the range.

    4. Indexing with the name of a sampler statistic that is not also
       the name of a variable returns those values from all chains.
       If there is more than one sampler that provides that statistic,
       the values are concatenated along a new axis.

    For any methods that require a single trace (e.g., taking the length
    of the MultiTrace instance, which returns the number of draws), the
    trace with the highest chain number is always used.

    Attributes
    ----------
    nchains: int
        Number of chains in the `MultiTrace`.
    chains: `List[int]`
        List of chain indices
    report: str
        Report on the sampling process.
    varnames: `List[str]`
        List of variable names in the trace(s)
    """

    def __init__(self, straces):
        self._straces = {}
        for strace in straces:
            if strace.chain in self._straces:
                raise ValueError("Chains are not unique.")
            self._straces[strace.chain] = strace

        self._report = SamplerReport()
        for strace in straces:
            if hasattr(strace, "_warnings"):
                self._report._add_warnings(strace._warnings, strace.chain)

    def __repr__(self):
        template = "<{}: {} chains, {} iterations, {} variables>"
        return template.format(self.__class__.__name__, self.nchains, len(self), len(self.varnames))

    @property
    def nchains(self):
        return len(self._straces)

    @property
    def chains(self):
        return list(sorted(self._straces.keys()))

    @property
    def report(self):
        return self._report

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

        var = get_var_name(var)
        if var in self.varnames:
            if var in self.stat_names:
                warnings.warn(
                    "Attribute access on a trace object is ambigous. "
                    "Sampler statistic and model variable share a name. Use "
                    "trace.get_values or trace.get_sampler_stats."
                )
            return self.get_values(var, burn=burn, thin=thin)
        if var in self.stat_names:
            return self.get_sampler_stats(var, burn=burn, thin=thin)
        raise KeyError("Unknown variable %s" % var)

    _attrs = {"_straces", "varnames", "chains", "stat_names", "supports_sampler_stats", "_report"}

    def __getattr__(self, name):
        # Avoid infinite recursion when called before __init__
        # variables are set up (e.g., when pickling).
        if name in self._attrs:
            raise AttributeError

        name = get_var_name(name)
        if name in self.varnames:
            if name in self.stat_names:
                warnings.warn(
                    "Attribute access on a trace object is ambigous. "
                    "Sampler statistic and model variable share a name. Use "
                    "trace.get_values or trace.get_sampler_stats."
                )
            return self.get_values(name)
        if name in self.stat_names:
            return self.get_sampler_stats(name)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __len__(self):
        chain = self.chains[-1]
        return len(self._straces[chain])

    @property
    def varnames(self):
        chain = self.chains[-1]
        return self._straces[chain].varnames

    @property
    def stat_names(self):
        if not self._straces:
            return set()
        sampler_vars = [s.sampler_vars for s in self._straces.values()]
        if not all(svars == sampler_vars[0] for svars in sampler_vars):
            raise ValueError("Inividual chains contain different sampler stats")
        names = set()
        for trace in self._straces.values():
            if trace.sampler_vars is None:
                continue
            for vars in trace.sampler_vars:
                names.update(vars.keys())
        return names

    def add_values(self, vals, overwrite=False) -> None:
        """Add variables to traces.

        Parameters
        ----------
        vals: dict (str: array-like)
             The keys should be the names of the new variables. The values are expected to be
             array-like objects. For traces with more than one chain the length of each value
             should match the number of total samples already in the trace `(chains * iterations)`,
             otherwise a warning is raised.
        overwrite: bool
            If `False` (default) a ValueError is raised if the variable already exists.
            Change to `True` to overwrite the values of variables

        Returns
        -------
            None.
        """
        for k, v in vals.items():
            new_var = 1
            if k in self.varnames:
                if overwrite:
                    self.varnames.remove(k)
                    new_var = 0
                else:
                    raise ValueError(f"Variable name {k} already exists.")

            self.varnames.append(k)

            chains = self._straces
            l_samples = len(self) * len(self.chains)
            l_v = len(v)
            if l_v != l_samples:
                warnings.warn(
                    "The length of the values you are trying to "
                    "add ({}) does not match the number ({}) of "
                    "total samples in the trace "
                    "(chains * iterations)".format(l_v, l_samples)
                )

            v = np.squeeze(v.reshape(len(chains), len(self), -1))

            for idx, chain in enumerate(chains.values()):
                if new_var:
                    dummy = tt.as_tensor_variable([], k)
                    chain.vars.append(dummy)
                chain.samples[k] = v[idx]

    def remove_values(self, name):
        """remove variables from traces.

        Parameters
        ----------
        name: str
            Name of the variable to remove. Raises KeyError if the variable is not present
        """
        varnames = self.varnames
        if name not in varnames:
            raise KeyError(f"Unknown variable {name}")
        self.varnames.remove(name)
        chains = self._straces
        for chain in chains.values():
            for va in chain.vars:
                if va.name == name:
                    chain.vars.remove(va)
                    del chain.samples[name]

    def get_values(self, varname, burn=0, thin=1, combine=True, chains=None, squeeze=True):
        """Get values from traces.

        Parameters
        ----------
        varname: str
        burn: int
        thin: int
        combine: bool
            If True, results from `chains` will be concatenated.
        chains: int or list of ints
            Chains to retrieve. If None, all chains are used. A single
            chain value can also be given.
        squeeze: bool
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
        varname = get_var_name(varname)
        try:
            results = [self._straces[chain].get_values(varname, burn, thin) for chain in chains]
        except TypeError:  # Single chain passed.
            results = [self._straces[chains].get_values(varname, burn, thin)]
        return _squeeze_cat(results, combine, squeeze)

    def get_sampler_stats(self, stat_name, burn=0, thin=1, combine=True, chains=None, squeeze=True):
        """Get sampler statistics from the trace.

        Parameters
        ----------
        stat_name: str
        sampler_idx: int or None
        burn: int
        thin: int

        Returns
        -------
        If the `sampler_idx` is specified, return the statistic with
        the given name in a numpy array. If it is not specified and there
        is more than one sampler that provides this statistic, return
        a numpy array of shape (m, n), where `m` is the number of
        such samplers, and `n` is the number of samples.
        """
        if stat_name not in self.stat_names:
            raise KeyError("Unknown sampler statistic %s" % stat_name)

        if chains is None:
            chains = self.chains
        try:
            chains = iter(chains)
        except TypeError:
            chains = [chains]

        results = [
            self._straces[chain].get_sampler_stats(stat_name, None, burn, thin) for chain in chains
        ]
        return _squeeze_cat(results, combine, squeeze)

    def _slice(self, slice):
        """Return a new MultiTrace object sliced according to `slice`."""
        new_traces = [trace._slice(slice) for trace in self._straces.values()]
        trace = MultiTrace(new_traces)
        idxs = slice.indices(len(self))
        trace._report = self._report._slice(*idxs)
        return trace

    def point(self, idx, chain=None):
        """Return a dictionary of point values at `idx`.

        Parameters
        ----------
        idx: int
        chain: int
            If a chain is not given, the highest chain number is used.
        """
        if chain is None:
            chain = self.chains[-1]
        return self._straces[chain].point(idx)

    def points(self, chains=None):
        """Return an iterator over all or some of the sample points

        Parameters
        ----------
        chains: list of int or N
            The chains whose points should be inlcuded in the iterator.  If
            chains is not given, include points from all chains.
        """
        if chains is None:
            chains = self.chains

        return itl.chain.from_iterable(self._straces[chain] for chain in chains)


def merge_traces(mtraces: List[MultiTrace]) -> MultiTrace:
    """Merge MultiTrace objects.

    Parameters
    ----------
    mtraces: list of MultiTraces
        Each instance should have unique chain numbers.

    Raises
    ------
    A ValueError is raised if any traces have overlapping chain numbers,
    or if chains are of different lengths.

    Returns
    -------
    A MultiTrace instance with merged chains
    """
    if len(mtraces) == 0:
        raise ValueError("Cannot merge an empty set of traces.")
    base_mtrace = mtraces[0]
    chain_len = len(base_mtrace)
    # check base trace
    if any(
        len(st) != chain_len for _, st in base_mtrace._straces.items()
    ):  # pylint: disable=line-too-long
        raise ValueError("Chains are of different lengths.")
    for new_mtrace in mtraces[1:]:
        for new_chain, strace in new_mtrace._straces.items():
            if new_chain in base_mtrace._straces:
                raise ValueError("Chains are not unique.")
            if len(strace) != chain_len:
                raise ValueError("Chains are of different lengths.")
            base_mtrace._straces[new_chain] = strace
    base_mtrace._report = merge_reports([trace.report for trace in mtraces])
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
