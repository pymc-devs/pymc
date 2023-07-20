#   Copyright 2023 The PyMC Developers
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
Created on Mar 7, 2011

@author: johnsalvatier
"""

import warnings

from abc import ABC, abstractmethod
from enum import IntEnum, unique
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np

from pytensor.graph.basic import Variable

from pymc.blocking import PointType, StatDtype, StatsDict, StatShape, StatsType
from pymc.model import modelcontext

__all__ = ("Competence", "CompoundStep")


@unique
class Competence(IntEnum):
    """Enum for characterizing competence classes of step methods.
    Values include:
    0: INCOMPATIBLE
    1: COMPATIBLE
    2: PREFERRED
    3: IDEAL
    """

    INCOMPATIBLE = 0
    COMPATIBLE = 1
    PREFERRED = 2
    IDEAL = 3


def infer_warn_stats_info(
    stats_dtypes: List[Dict[str, StatDtype]],
    sds: Dict[str, Tuple[StatDtype, StatShape]],
    stepname: str,
) -> Tuple[List[Dict[str, StatDtype]], Dict[str, Tuple[StatDtype, StatShape]]]:
    """Helper function to get `stats_dtypes` and `stats_dtypes_shapes` from either of them."""
    # Avoid side-effects on the original lists/dicts
    stats_dtypes = [d.copy() for d in stats_dtypes]
    sds = sds.copy()
    # Disallow specification of both attributes
    if stats_dtypes and sds:
        raise TypeError(
            "Only one of `stats_dtypes_shapes` or `stats_dtypes` must be specified."
            f" `{stepname}.stats_dtypes` should be removed."
        )

    # Infer one from the other
    if not sds and stats_dtypes:
        warnings.warn(
            f"`{stepname}.stats_dtypes` is deprecated."
            " Please update it to specify `stats_dtypes_shapes` instead.",
            DeprecationWarning,
        )
        if len(stats_dtypes) > 1:
            raise TypeError(
                f"`{stepname}.stats_dtypes` must be a list containing at most one dict."
            )
        for sd in stats_dtypes:
            for sname, dtype in sd.items():
                sds[sname] = (dtype, None)
    elif sds:
        stats_dtypes.append({sname: dtype for sname, (dtype, _) in sds.items()})
    return stats_dtypes, sds


class BlockedStep(ABC):
    stats_dtypes: List[Dict[str, type]] = []
    """A list containing <=1 dictionary that maps stat names to dtypes.

    This attribute is deprecated.
    Use `stats_dtypes_shapes` instead.
    """

    stats_dtypes_shapes: Dict[str, Tuple[StatDtype, StatShape]] = {}
    """Maps stat names to dtypes and shapes.

    Shapes are interpreted in the following ways:
    - `[]` is a scalar.
    - `[3,]` is a length-3 vector.
    - `[4, None]` is a matrix with 4 rows and a dynamic number of columns.
    - `None` is a sparse stat (i.e. not always present) or a NumPy array with varying `ndim`.
    """

    vars: List[Variable] = []
    """Variables that the step method is assigned to."""

    def __new__(cls, *args, **kwargs):
        blocked = kwargs.get("blocked")
        if blocked is None:
            # Try to look up default value from class
            blocked = getattr(cls, "default_blocked", True)
            kwargs["blocked"] = blocked

        model = modelcontext(kwargs.get("model"))
        kwargs.update({"model": model})

        # vars can either be first arg or a kwarg
        if "vars" not in kwargs and len(args) >= 1:
            vars = args[0]
            args = args[1:]
        elif "vars" in kwargs:
            vars = kwargs.pop("vars")
        else:  # Assume all model variables
            vars = model.value_vars

        if not isinstance(vars, (tuple, list)):
            vars = [vars]

        if len(vars) == 0:
            raise ValueError("No free random variables to sample.")

        # Auto-fill stats metadata attributes from whichever was given.
        stats_dtypes, stats_dtypes_shapes = infer_warn_stats_info(
            cls.stats_dtypes,
            cls.stats_dtypes_shapes,
            cls.__name__,
        )

        if not blocked and len(vars) > 1:
            # In this case we create a separate sampler for each var
            # and append them to a CompoundStep
            steps = []
            for var in vars:
                step = super().__new__(cls)
                step.stats_dtypes = stats_dtypes
                step.stats_dtypes_shapes = stats_dtypes_shapes
                # If we don't return the instance we have to manually
                # call __init__
                step.__init__([var], *args, **kwargs)
                # Hack for creating the class correctly when unpickling.
                step.__newargs = ([var],) + args, kwargs
                steps.append(step)

            return CompoundStep(steps)
        else:
            step = super().__new__(cls)
            step.stats_dtypes = stats_dtypes
            step.stats_dtypes_shapes = stats_dtypes_shapes
            # Hack for creating the class correctly when unpickling.
            step.__newargs = (vars,) + args, kwargs
            return step

    # Hack for creating the class correctly when unpickling.
    def __getnewargs_ex__(self):
        return self.__newargs

    @abstractmethod
    def step(self, point: PointType) -> Tuple[PointType, StatsType]:
        """Perform a single step of the sampler."""

    @staticmethod
    def competence(var, has_grad):
        return Competence.INCOMPATIBLE

    @classmethod
    def _competence(cls, vars, have_grad):
        vars = np.atleast_1d(vars)
        have_grad = np.atleast_1d(have_grad)
        competences = []
        for var, has_grad in zip(vars, have_grad):
            try:
                competences.append(cls.competence(var, has_grad))
            except TypeError:
                competences.append(cls.competence(var))
        return competences

    def stop_tuning(self):
        if hasattr(self, "tune"):
            self.tune = False


def flat_statname(sampler_idx: int, sname: str) -> str:
    """Get the flat-stats name for a samplers stat."""
    return f"sampler_{sampler_idx}__{sname}"


def get_stats_dtypes_shapes_from_steps(
    steps: Iterable[BlockedStep],
) -> Dict[str, Tuple[StatDtype, StatShape]]:
    """Combines stats dtype shape dictionaries from multiple step methods.

    In the resulting stats dict, each sampler stat is prefixed by `sampler_#__`.
    """
    result = {}
    for s, step in enumerate(steps):
        for sname, (dtype, shape) in step.stats_dtypes_shapes.items():
            result[flat_statname(s, sname)] = (dtype, shape)
    return result


class CompoundStep:
    """Step method composed of a list of several other step
    methods applied in sequence."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.stats_dtypes = []
        for method in self.methods:
            self.stats_dtypes.extend(method.stats_dtypes)
        self.stats_dtypes_shapes = get_stats_dtypes_shapes_from_steps(methods)
        self.name = (
            f"Compound[{', '.join(getattr(m, 'name', 'UNNAMED_STEP') for m in self.methods)}]"
        )
        self.tune = True

    def step(self, point) -> Tuple[PointType, StatsType]:
        stats = []
        for method in self.methods:
            point, sts = method.step(point)
            stats.extend(sts)
        # Model logp can only be the logp of the _last_ stats,
        # if there is one. Pop all others.
        for sts in stats[:-1]:
            sts.pop("model_logp", None)
        return point, stats

    def stop_tuning(self):
        for method in self.methods:
            method.stop_tuning()

    def reset_tuning(self):
        for method in self.methods:
            if hasattr(method, "reset_tuning"):
                method.reset_tuning()

    @property
    def vars(self) -> List[Variable]:
        return [var for method in self.methods for var in method.vars]


def flatten_steps(step: Union[BlockedStep, CompoundStep]) -> List[BlockedStep]:
    """Flatten a hierarchy of step methods to a list."""
    if isinstance(step, BlockedStep):
        return [step]
    steps = []
    if not isinstance(step, CompoundStep):
        raise ValueError(f"Unexpected type of step method: {step}")
    for sm in step.methods:
        steps += flatten_steps(sm)
    return steps


def check_step_emits_tune(step: Union[CompoundStep, BlockedStep]):
    if isinstance(step, BlockedStep) and "tune" not in step.stats_dtypes_shapes:
        raise TypeError(f"{type(step)} does not emit the required 'tune' stat.")
    elif isinstance(step, CompoundStep):
        for sstep in step.methods:
            if "tune" not in sstep.stats_dtypes_shapes:
                raise TypeError(f"{type(sstep)} does not emit the required 'tune' stat.")
    return


class StatsBijection:
    """Map between a `list` of stats to `dict` of stats."""

    def __init__(self, sampler_stats_dtypes: Sequence[Mapping[str, type]]) -> None:
        # Keep a list of flat vs. original stat names
        stat_groups = []
        for s, names_dtypes in enumerate(sampler_stats_dtypes):
            group = []
            for statname, dtype in names_dtypes.items():
                flatname = flat_statname(s, statname)
                is_obj = np.dtype(dtype) == np.dtype(object)
                group.append((flatname, statname, is_obj))
            stat_groups.append(group)
        self._stat_groups: List[List[Tuple[str, str, bool]]] = stat_groups
        self.object_stats = {
            fname: (s, sname)
            for s, group in enumerate(self._stat_groups)
            for fname, sname, is_obj in group
            if is_obj
        }

    @property
    def n_samplers(self) -> int:
        return len(self._stat_groups)

    def map(self, stats_list: Sequence[Mapping[str, Any]]) -> StatsDict:
        """Combine stats dicts of multiple samplers into one dict."""
        stats_dict = {}
        for s, sts in enumerate(stats_list):
            for fname, sname, is_obj in self._stat_groups[s]:
                if sname not in sts:
                    continue
                stats_dict[fname] = sts[sname]
        return stats_dict

    def rmap(self, stats_dict: Mapping[str, Any]) -> StatsType:
        """Split a global stats dict into a list of sampler-wise stats dicts.

        The ``stats_dict`` can be a subset of all sampler stats.
        """
        stats_list = []
        for group in self._stat_groups:
            d = {}
            for fname, sname, is_obj in group:
                if fname not in stats_dict:
                    continue
                d[sname] = stats_dict[fname]
            stats_list.append(d)
        return stats_list
