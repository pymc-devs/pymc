"""Functions for converting traces into a table-like format
"""

import numpy as np
import pandas as pd

__all__ = ['trace_to_dataframe']


def trace_to_dataframe(trace, chains=None, varnames=None, hide_transformed_vars=True):
    """Convert trace to Pandas DataFrame.

    Parameters
    ----------
    trace : NDarray trace
    chains : int or list of ints
        Chains to include. If None, all chains are used. A single
        chain value can also be given.
    varnames : list of variable names
        Variables to be included in the DataFrame, if None all variable are
        included.
    hide_transformed_vars: boolean
        If true transformed variables will not be included in the resulting
        DataFrame.
    """
    var_shapes = trace._straces[0].var_shapes

    if varnames is None:
        varnames = var_shapes.keys()

    flat_names = {v: create_flat_names(v, shape)
                    for v, shape in var_shapes.items()
                    if not (hide_transformed_vars and v.endswith('_'))}

    var_dfs = []
    for v in var_shapes:
        if v in varnames:
            if not hide_transformed_vars or not v.endswith('_'):
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
    return ['{}__{}'.format(varname, '_'.join(idxs)) for idxs in zip(*labels)]


def _create_shape(flat_names):
    "Determine shape from `create_flat_names` output."
    try:
        _, shape_str = flat_names[-1].rsplit('__', 1)
    except ValueError:
        return ()
    return tuple(int(i) + 1 for i in shape_str.split('_'))
