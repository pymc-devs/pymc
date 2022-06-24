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
pymc.blocking

Classes for working with subsets of parameters.
"""
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np

try:
    from math import prod
except ImportError:

    def prod(vals):
        product = 1
        for val in vals:
            product *= val
        return product


__all__ = ["DictToArrayBijection"]


T = TypeVar("T")
PointType = Dict[str, np.ndarray]


class PointMapItem(NamedTuple):
    name: str
    shape: Tuple[int]
    dtype: np.dtype
    data_slice: slice


PointMapInfo = List[PointMapItem]


class RaveledVars(NamedTuple):
    data: np.ndarray
    point_map_info: PointMapInfo


class Compose:
    """
    Compose two functions in a pickleable way
    """

    def __init__(self, fa: Callable[[PointType], T], fb: Callable[[RaveledVars], PointType]):
        self.fa = fa
        self.fb = fb

    def __call__(self, x: RaveledVars) -> T:
        return self.fa(self.fb(x))


class DictToArrayBijection:
    """Map between a `dict`s of variables to an array space.

    Said array space consists of all the vars raveled and then concatenated.

    """

    @staticmethod
    def map(var_dict: PointType) -> RaveledVars:
        """Map a dictionary of names and variables to a concatenated 1D array space."""

        count = 0
        infos = []
        raveled = []
        for name, variable in var_dict.items():
            size = prod(variable.shape)
            data_slice = slice(count, count + size)
            count += size
            infos.append(PointMapItem(name, variable.shape, variable.dtype, data_slice))
            raveled.append(variable.ravel())

        if raveled:
            joined = np.concatenate(raveled)
        else:
            joined = np.array([], dtpye=np.float64)

        return RaveledVars(joined, infos)

    @staticmethod
    def rmap(
        array: RaveledVars,
        start_point: Optional[PointType] = None,
    ) -> PointType:
        """Map 1D concatenated array to a dictionary of variables in their original spaces.

        Parameters
        ==========
        array
            The array to map.
        start_point
            An optional dictionary of initial values.

        """
        if start_point:
            result = dict(start_point)
        else:
            result = {}

        if not isinstance(array, RaveledVars):
            raise TypeError("`array` must be a `RaveledVars` type")

        for info in array.point_map_info:
            values = array.data[info.data_slice].reshape(info.shape).astype(info.dtype)
            result[info.name] = values

        return result

    @classmethod
    def mapf(
        cls, f: Callable[[PointType], T], start_point: Optional[PointType] = None
    ) -> Callable[[RaveledVars], T]:
        """Create a callable that first maps back to ``dict`` inputs and then applies a function.

        function f: DictSpace -> T to ArraySpace -> T

        Parameters
        ----------
        f: dict -> T

        Returns
        -------
        f: array -> T
        """
        return Compose(f, partial(cls.rmap, start_point=start_point))
