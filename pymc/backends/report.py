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

import dataclasses
import logging

from typing import Dict, List, Optional

from pymc.stats.convergence import _LEVELS, SamplerWarning

logger = logging.getLogger(__name__)


class SamplerReport:
    """Bundle warnings, convergence stats and metadata of a sampling run."""

    def __init__(self) -> None:
        self._chain_warnings: Dict[int, List[SamplerWarning]] = {}
        self._global_warnings: List[SamplerWarning] = []
        self._n_tune = None
        self._n_draws = None
        self._t_sampling = None

    @property
    def _warnings(self):
        chains = sum(self._chain_warnings.values(), [])
        return chains + self._global_warnings

    @property
    def ok(self):
        """Whether the automatic convergence checks found serious problems."""
        return all(_LEVELS[warn.level] < _LEVELS["warn"] for warn in self._warnings)

    @property
    def n_tune(self) -> Optional[int]:
        """Number of tune iterations - not necessarily kept in trace!"""
        return self._n_tune

    @property
    def n_draws(self) -> Optional[int]:
        """Number of draw iterations."""
        return self._n_draws

    @property
    def t_sampling(self) -> Optional[float]:
        """
        Number of seconds that the sampling procedure took.

        (Includes parallelization overhead.)
        """
        return self._t_sampling

    def raise_ok(self, level="error"):
        errors = [warn for warn in self._warnings if _LEVELS[warn.level] >= _LEVELS[level]]
        if errors:
            raise ValueError("Serious convergence issues during sampling.")

    def _add_warnings(self, warnings, chain=None):
        if chain is None:
            warn_list = self._global_warnings
        else:
            warn_list = self._chain_warnings.setdefault(chain, [])
        warn_list.extend(warnings)

    def _slice(self, start, stop, step):
        report = SamplerReport()

        def filter_warns(warnings):
            filtered = []
            for warn in warnings:
                if warn.step is None:
                    filtered.append(warn)
                elif start <= warn.step < stop and (warn.step - start) % step == 0:
                    warn = dataclasses.replace(warn, step=warn.step - start)
                    filtered.append(warn)
            return filtered

        report._add_warnings(filter_warns(self._global_warnings))
        for chain in self._chain_warnings:
            report._add_warnings(filter_warns(self._chain_warnings[chain]), chain)

        return report
