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
import collections

from functools import partial
from typing import Callable, Dict, Optional, TypeVar

import numpy as np

__all__ = ["DictToArrayBijection"]


T = TypeVar("T")
PointType = Dict[str, np.ndarray]

# `point_map_info` is a tuple of tuples containing `(name, shape, dtype)` for
# each of the raveled variables.
RaveledVars = collections.namedtuple("RaveledVars", "data, point_map_info")


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
        vars_info = tuple((v, k, v.shape, v.dtype) for k, v in var_dict.items())
        raveled_vars = [v[0].ravel() for v in vars_info]
        if raveled_vars:
            result = np.concatenate(raveled_vars)
        else:
            result = np.array([])
        return RaveledVars(result, tuple(v[1:] for v in vars_info))

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

        last_idx = 0
        for name, shape, dtype in array.point_map_info:
            arr_len = np.prod(shape, dtype=int)
            var = array.data[last_idx : last_idx + arr_len].reshape(shape).astype(dtype)
            result[name] = var
            last_idx += arr_len

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
