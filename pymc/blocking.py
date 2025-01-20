#   Copyright 2024 - present The PyMC Developers
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

"""Classes for working with subsets of parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import (
    Any,
    Generic,
    NamedTuple,
    TypeAlias,
    TypeVar,
)

import numpy as np

__all__ = ["DictToArrayBijection"]


T = TypeVar("T")
PointType: TypeAlias = dict[str, np.ndarray]
StatsDict: TypeAlias = dict[str, Any]
StatsType: TypeAlias = list[StatsDict]
StatDtype: TypeAlias = type | np.dtype
StatShape: TypeAlias = Sequence[int | None] | None


# `point_map_info` is a tuple of tuples containing `(name, shape, size, dtype)` for
# each of the raveled variables.
class RaveledVars(NamedTuple):
    data: np.ndarray
    point_map_info: tuple[tuple[str, tuple[int, ...], int, np.dtype], ...]


class Compose(Generic[T]):
    """Compose two functions in a pickleable way."""

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
        vars_info = tuple((v, k, v.shape, v.size, v.dtype) for k, v in var_dict.items())
        if vars_info:
            result = np.concatenate(tuple(v[0].ravel() for v in vars_info))
        else:
            result = np.array([])
        return RaveledVars(result, tuple(v[1:] for v in vars_info))

    @staticmethod
    def rmap(
        array: RaveledVars,
        start_point: PointType | None = None,
    ) -> PointType:
        """Map 1D concatenated array to a dictionary of variables in their original spaces.

        Parameters
        ----------
        array
            The array to map.
        start_point
            An optional dictionary of initial values.

        """
        if start_point:
            result = start_point.copy()
        else:
            result = {}

        last_idx = 0
        for name, shape, size, dtype in array.point_map_info:
            end = last_idx + size
            result[name] = array.data[last_idx:end].reshape(shape).astype(dtype)
            last_idx = end

        return result

    @classmethod
    def mapf(
        cls, f: Callable[[PointType], T], start_point: PointType | None = None
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
