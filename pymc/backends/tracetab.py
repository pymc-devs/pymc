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


import numpy as np


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
