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

"""
pymc3.blocking

Classes for working with subsets of parameters.
"""
import collections

from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["ArrayOrdering", "DictToArrayBijection"]

# `point_map_info` is a tuple of tuples containing `(name, shape, dtype)` for
# each of the raveled variables.
RaveledVars = collections.namedtuple("RaveledVars", "data, point_map_info")
VarMap = collections.namedtuple("VarMap", "var, slc, shp, dtyp")
DataMap = collections.namedtuple("DataMap", "list_ind, slc, shp, dtype, name")


class ArrayOrdering:
    """
    An ordering for an array space
    """

    def __init__(self, vars):
        self.vmap = []
        self.by_name = {}
        self.size = 0

        for var in vars:
            name = var.name
            if name is None:
                raise ValueError("Unnamed variable in ArrayOrdering.")
            if name in self.by_name:
                raise ValueError("Name of variable not unique: %s." % name)
            if not hasattr(var, "dshape") or not hasattr(var, "dsize"):
                raise ValueError("Shape of variable not known %s" % name)

            slc = slice(self.size, self.size + var.dsize)
            varmap = VarMap(name, slc, var.dshape, var.dtype)
            self.vmap.append(varmap)
            self.by_name[name] = varmap
            self.size += var.dsize

    def __getitem__(self, key):
        return self.by_name[key]


class DictToArrayBijection:
    """Map between a `dict`s of variables to an array space.

    Said array space consists of all the vars raveled and then concatenated.

    """

    @staticmethod
    def map(var_dict: Dict[str, np.ndarray]) -> RaveledVars:
        """Map a dictionary of names and variables to a concatenated 1D array space."""
        vars_info = tuple((v, k, v.shape, v.dtype) for k, v in var_dict.items())
        res = np.concatenate([v[0].ravel() for v in vars_info])
        return RaveledVars(res, tuple(v[1:] for v in vars_info))

    @staticmethod
    def rmap(
        array: RaveledVars, as_list: Optional[bool] = False
    ) -> Union[Dict[str, np.ndarray], List[np.ndarray]]:
        """Map 1D concatenated array to a dictionary of variables in their original spaces.

        Parameters
        ==========
        array
            The array to map.
        as_list
            When ``True``, return a list of the original variables instead of a
            ``dict`` keyed each variable's name.
        """
        if as_list:
            res = []
        else:
            res = {}

        if not isinstance(array, RaveledVars):
            raise TypeError("`apt` must be a `RaveledVars` type")

        last_idx = 0
        for name, shape, dtype in array.point_map_info:
            arr_len = np.prod(shape, dtype=int)
            var = array.data[last_idx : last_idx + arr_len].reshape(shape).astype(dtype)
            if as_list:
                res.append(var)
            else:
                res[name] = var
            last_idx += arr_len

        return res

    @classmethod
    def mapf(cls, f):
        """
         function f: DictSpace -> T to ArraySpace -> T

        Parameters
        ----------
        f: dict -> T

        Returns
        -------
        f: array -> T
        """
        return Compose(f, cls.rmap)


class Compose:
    """
    Compose two functions in a pickleable way
    """

    def __init__(self, fa, fb):
        self.fa = fa
        self.fb = fb

    def __call__(self, x):
        return self.fa(self.fb(x))
