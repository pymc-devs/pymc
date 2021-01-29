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

from typing import Dict, List, Text, Union

import numpy as np

__all__ = ["ArrayOrdering", "DictToArrayBijection"]

RaveledVars = collections.namedtuple("RaveledVars", "data, point_map_info")
VarMap = collections.namedtuple("VarMap", "var, slc, shp, dtyp")
DataMap = collections.namedtuple("DataMap", "list_ind, slc, shp, dtype, name")


# TODO Classes and methods need to be fully documented.


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
    """
    A mapping between a dict space and an array space
    """

    def __init__(self, ordering: List[Text]):
        # TODO: Should just use `OrderedDict`s and remove this state entirely
        self.ordering = ordering

    def map(self, dpt: Dict[Text, np.ndarray]):
        """
        Maps value from dict space to array space

        Parameters
        ----------
        dpt: dict
        """
        vars_info = tuple((dpt[n], n, dpt[n].shape, dpt[n].dtype) for n in self.ordering)
        res = np.concatenate([v[0].ravel() for v in vars_info])
        return RaveledVars(res, tuple(v[1:] for v in vars_info))

    @classmethod
    def rmap(
        cls, apt: RaveledVars, as_list=False
    ) -> Union[Dict[Text, np.ndarray], List[np.ndarray]]:
        """
        Maps value from array space to dict space

        Parameters
        ----------
        apt: array
        """
        if as_list:
            res = []
        else:
            res = {}

        if not isinstance(apt, RaveledVars):
            raise TypeError("`apt` must be a `RaveledVars` type")

        last_idx = 0
        for name, shape, dtype in apt.point_map_info:
            arr_len = np.prod(shape, dtype=int)
            var = apt.data[last_idx : last_idx + arr_len].reshape(shape).astype(dtype)
            if as_list:
                res.append(var)
            else:
                res[name] = var
            last_idx += arr_len

        return res

    def mapf(self, f):
        """
         function f: DictSpace -> T to ArraySpace -> T

        Parameters
        ----------

        f: dict -> T

        Returns
        -------
        f: array -> T
        """
        return Compose(f, self.rmap)


class Compose:
    """
    Compose two functions in a pickleable way
    """

    def __init__(self, fa, fb):
        self.fa = fa
        self.fb = fb

    def __call__(self, x):
        return self.fa(self.fb(x))
