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

import dataclasses
import enum
import logging

from typing import Any, Optional

import arviz

from pymc3.util import get_untransformed_name, is_transformed_name

logger = logging.getLogger("pymc3")


@enum.unique
class WarningType(enum.Enum):
    # For HMC and NUTS
    DIVERGENCE = 1
    TUNING_DIVERGENCE = 2
    DIVERGENCES = 3
    TREEDEPTH = 4
    # Problematic sampler parameters
    BAD_PARAMS = 5
    # Indications that chains did not converge, eg Rhat
    CONVERGENCE = 6
    BAD_ACCEPTANCE = 7
    BAD_ENERGY = 8


@dataclasses.dataclass
class SamplerWarning:
    kind: WarningType
    message: str
    level: str
    step: Optional[int] = None
    exec_info: Optional[Any] = None
    extra: Optional[Any] = None
    divergence_point_source: Optional[dict] = None
    divergence_point_dest: Optional[dict] = None
    divergence_info: Optional[Any] = None


_LEVELS = {
    "info": logging.INFO,
    "error": logging.ERROR,
    "warn": logging.WARN,
    "debug": logging.DEBUG,
    "critical": logging.CRITICAL,
}


class SamplerReport:
    """Bundle warnings, convergence stats and metadata of a sampling run."""

    def __init__(self):
        self._chain_warnings = {}
        self._global_warnings = []
        self._ess = None
        self._rhat = None
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

    def _run_convergence_checks(self, idata: arviz.InferenceData, model):
        if not hasattr(idata, "posterior"):
            msg = "No posterior samples. Unable to run convergence checks"
            warn = SamplerWarning(WarningType.BAD_PARAMS, msg, "info", None, None, None)
            self._add_warnings([warn])
            return

        if idata.posterior.sizes["chain"] == 1:
            msg = (
                "Only one chain was sampled, this makes it impossible to "
                "run some convergence checks"
            )
            warn = SamplerWarning(WarningType.BAD_PARAMS, msg, "info")
            self._add_warnings([warn])
            return

        valid_name = [rv.name for rv in model.free_RVs + model.deterministics]
        varnames = []
        for rv in model.free_RVs:
            rv_name = rv.name
            if is_transformed_name(rv_name):
                rv_name2 = get_untransformed_name(rv_name)
                rv_name = rv_name2 if rv_name2 in valid_name else rv_name
            if rv_name in idata.posterior:
                varnames.append(rv_name)

        self._ess = ess = arviz.ess(idata, var_names=varnames)
        self._rhat = rhat = arviz.rhat(idata, var_names=varnames)

        warnings = []
        rhat_max = max(val.max() for val in rhat.values())
        if rhat_max > 1.4:
            msg = (
                "The rhat statistic is larger than 1.4 for some "
                "parameters. The sampler did not converge."
            )
            warn = SamplerWarning(WarningType.CONVERGENCE, msg, "error", extra=rhat)
            warnings.append(warn)
        elif rhat_max > 1.2:
            msg = "The rhat statistic is larger than 1.2 for some " "parameters."
            warn = SamplerWarning(WarningType.CONVERGENCE, msg, "warn", extra=rhat)
            warnings.append(warn)
        elif rhat_max > 1.05:
            msg = (
                "The rhat statistic is larger than 1.05 for some "
                "parameters. This indicates slight problems during "
                "sampling."
            )
            warn = SamplerWarning(WarningType.CONVERGENCE, msg, "info", extra=rhat)
            warnings.append(warn)

        eff_min = min(val.min() for val in ess.values())
        sizes = idata.posterior.sizes
        n_samples = sizes["chain"] * sizes["draw"]
        if eff_min < 200 and n_samples >= 500:
            msg = (
                "The estimated number of effective samples is smaller than "
                "200 for some parameters."
            )
            warn = SamplerWarning(WarningType.CONVERGENCE, msg, "error", extra=ess)
            warnings.append(warn)
        elif eff_min / n_samples < 0.1:
            msg = "The number of effective samples is smaller than " "10% for some parameters."
            warn = SamplerWarning(WarningType.CONVERGENCE, msg, "warn", extra=ess)
            warnings.append(warn)
        elif eff_min / n_samples < 0.25:
            msg = "The number of effective samples is smaller than " "25% for some parameters."
            warn = SamplerWarning(WarningType.CONVERGENCE, msg, "info", extra=ess)
            warnings.append(warn)

        self._add_warnings(warnings)

    def _add_warnings(self, warnings, chain=None):
        if chain is None:
            warn_list = self._global_warnings
        else:
            warn_list = self._chain_warnings.setdefault(chain, [])
        warn_list.extend(warnings)

    def _log_summary(self):
        def log_warning(warn):
            level = _LEVELS[warn.level]
            logger.log(level, warn.message)

        for chain, warns in self._chain_warnings.items():
            for warn in warns:
                log_warning(warn)
        for warn in self._global_warnings:
            log_warning(warn)

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


def merge_reports(reports):
    report = SamplerReport()
    for rep in reports:
        report._add_warnings(rep._global_warnings)
        for chain in rep._chain_warnings:
            report._add_warnings(rep._chain_warnings[chain], chain)
    return report
