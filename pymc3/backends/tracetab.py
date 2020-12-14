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

"""Functions for converting traces into a table-like format
"""

import warnings

import numpy as np
import pandas as pd

from pymc3.util import get_default_varnames

__all__ = ["trace_to_dataframe"]


def trace_to_dataframe(trace, chains=None, varnames=None, include_transformed=False):
    """Convert trace to pandas DataFrame.

    Parameters
    ----------
    trace: NDarray trace
    chains: int or list of ints
        Chains to include. If None, all chains are used. A single
        chain value can also be given.
    varnames: list of variable names
        Variables to be included in the DataFrame, if None all variable are
        included.
    include_transformed: boolean
        If true transformed variables will be included in the resulting
        DataFrame.
    """
    warnings.warn(
        "The `trace_to_dataframe` function will soon be removed. "
        "Please use ArviZ to save traces. "
        "If you have good reasons for using the `trace_to_dataframe` function, file an issue and tell us about them. ",
        DeprecationWarning,
    )
    var_shapes = trace._straces[0].var_shapes

    if varnames is None:
        varnames = get_default_varnames(var_shapes.keys(), include_transformed=include_transformed)

    flat_names = {v: create_flat_names(v, var_shapes[v]) for v in varnames}

    var_dfs = []
    for v in varnames:
        vals = trace.get_values(v, combine=True, chains=chains)
        flat_vals = vals.reshape(vals.shape[0], -1)
        var_dfs.append(pd.DataFrame(flat_vals, columns=flat_names[v]))
    return pd.concat(var_dfs, axis=1)


def create_flat_names(varname, shape):
    """Return flat variable names for `varname` of `shape`.

    Examples
    --------
    >>> create_flat_names('x', (5,))
    ['x__0', 'x__1', 'x__2', 'x__3', 'x__4']

    >>> create_flat_names('x', (2, 2))
    ['x__0_0', 'x__0_1', 'x__1_0', 'x__1_1']
    """
    if not shape:
        return [varname]
    labels = (np.ravel(xs).tolist() for xs in np.indices(shape))
    labels = (map(str, xs) for xs in labels)
    return ["{}__{}".format(varname, "_".join(idxs)) for idxs in zip(*labels)]


def _create_shape(flat_names):
    """Determine shape from `create_flat_names` output."""
    try:
        _, shape_str = flat_names[-1].rsplit("__", 1)
    except ValueError:
        return ()
    return tuple(int(i) + 1 for i in shape_str.split("_"))
