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
Created on Mar 7, 2011

@author: johnsalvatier
"""


from abc import ABC, abstractmethod
from enum import IntEnum, unique
from typing import Dict, List, Tuple

import numpy as np

from pytensor.graph.basic import Variable

from pymc.blocking import PointType, StatsType
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


class BlockedStep(ABC):

    stats_dtypes: List[Dict[str, type]] = []
    vars: List[Variable] = []

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

        if not blocked and len(vars) > 1:
            # In this case we create a separate sampler for each var
            # and append them to a CompoundStep
            steps = []
            for var in vars:
                step = super().__new__(cls)
                # If we don't return the instance we have to manually
                # call __init__
                step.__init__([var], *args, **kwargs)
                # Hack for creating the class correctly when unpickling.
                step.__newargs = ([var],) + args, kwargs
                steps.append(step)

            return CompoundStep(steps)
        else:
            step = super().__new__(cls)
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


class CompoundStep:
    """Step method composed of a list of several other step
    methods applied in sequence."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.stats_dtypes = []
        for method in self.methods:
            self.stats_dtypes.extend(method.stats_dtypes)
        self.name = (
            f"Compound[{', '.join(getattr(m, 'name', 'UNNAMED_STEP') for m in self.methods)}]"
        )

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
    def vars(self):
        return [var for method in self.methods for var in method.vars]
