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


from typing import Tuple

from pymc.blocking import PointType, StatsType


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
    def vars(self):
        return [var for method in self.methods for var in method.vars]
