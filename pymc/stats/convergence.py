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
import enum
import logging

from typing import Any, Dict, List, Optional, Sequence

import arviz

from pymc.util import get_untransformed_name, is_transformed_name

_LEVELS = {
    "info": logging.INFO,
    "error": logging.ERROR,
    "warn": logging.WARN,
    "debug": logging.DEBUG,
    "critical": logging.CRITICAL,
}

logger = logging.getLogger(__name__)


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


def run_convergence_checks(idata: arviz.InferenceData, model) -> List[SamplerWarning]:
    if not hasattr(idata, "posterior"):
        msg = "No posterior samples. Unable to run convergence checks"
        warn = SamplerWarning(WarningType.BAD_PARAMS, msg, "info", None, None, None)
        return [warn]

    if idata["posterior"].sizes["draw"] < 100:
        msg = "The number of samples is too small to check convergence reliably."
        warn = SamplerWarning(WarningType.BAD_PARAMS, msg, "info", None, None, None)
        return [warn]

    if idata["posterior"].sizes["chain"] == 1:
        msg = "Only one chain was sampled, this makes it impossible to run some convergence checks"
        warn = SamplerWarning(WarningType.BAD_PARAMS, msg, "info")
        return [warn]

    elif idata["posterior"].sizes["chain"] < 4:
        msg = (
            "We recommend running at least 4 chains for robust computation of "
            "convergence diagnostics"
        )
        warn = SamplerWarning(WarningType.BAD_PARAMS, msg, "info")
        return [warn]

    warnings: List[SamplerWarning] = []
    valid_name = [rv.name for rv in model.free_RVs + model.deterministics]
    varnames = []
    for rv in model.free_RVs:
        rv_name = rv.name
        if is_transformed_name(rv_name):
            rv_name2 = get_untransformed_name(rv_name)
            rv_name = rv_name2 if rv_name2 in valid_name else rv_name
        if rv_name in idata["posterior"]:
            varnames.append(rv_name)

    ess = arviz.ess(idata, var_names=varnames)
    rhat = arviz.rhat(idata, var_names=varnames)

    warnings = []
    rhat_max = max(val.max() for val in rhat.values())
    if rhat_max > 1.01:
        msg = (
            "The rhat statistic is larger than 1.01 for some "
            "parameters. This indicates problems during sampling. "
            "See https://arxiv.org/abs/1903.08008 for details"
        )
        warn = SamplerWarning(WarningType.CONVERGENCE, msg, "info", extra=rhat)
        warnings.append(warn)

    eff_min = min(val.min() for val in ess.values())
    eff_per_chain = eff_min / idata["posterior"].sizes["chain"]
    if eff_per_chain < 100:
        msg = (
            "The effective sample size per chain is smaller than 100 for some parameters. "
            " A higher number is needed for reliable rhat and ess computation. "
            "See https://arxiv.org/abs/1903.08008 for details"
        )
        warn = SamplerWarning(WarningType.CONVERGENCE, msg, "error", extra=ess)
        warnings.append(warn)

    warnings += warn_divergences(idata)
    warnings += warn_treedepth(idata)

    return warnings


def warn_divergences(idata: arviz.InferenceData) -> List[SamplerWarning]:
    """Checks sampler stats and creates a list of warnings about divergences."""
    sampler_stats = idata.get("sample_stats", None)
    if sampler_stats is None:
        return []

    diverging = sampler_stats.get("diverging", None)
    if diverging is None:
        return []

    # Warn about divergences
    n_div = int(diverging.sum())
    if n_div == 0:
        return []
    warning = SamplerWarning(
        WarningType.DIVERGENCES,
        f"There were {n_div} divergences after tuning. Increase `target_accept` or reparameterize.",
        "error",
    )
    return [warning]


def warn_treedepth(idata: arviz.InferenceData) -> List[SamplerWarning]:
    """Checks sampler stats and creates a list of warnings about tree depth."""
    sampler_stats = idata.get("sample_stats", None)
    if sampler_stats is None:
        return []

    rmtd = sampler_stats.get("reached_max_treedepth", None)
    if rmtd is None:
        return []

    warnings = []
    for c in rmtd.chain:
        if sum(rmtd.sel(chain=c)) / rmtd.sizes["draw"] > 0.05:
            warnings.append(
                SamplerWarning(
                    WarningType.TREEDEPTH,
                    f"Chain {int(c)} reached the maximum tree depth."
                    " Increase `max_treedepth`, increase `target_accept` or reparameterize.",
                    "warn",
                )
            )
    return warnings


def log_warning(warn: SamplerWarning):
    level = _LEVELS.get(warn.level, logging.WARNING)
    logger.log(level, warn.message)


def log_warnings(warnings: Sequence[SamplerWarning]):
    for warn in warnings:
        log_warning(warn)


def log_warning_stats(stats: Sequence[Dict[str, Any]]):
    """Logs 'warning' stats if present."""
    if stats is None:
        return

    for sts in stats:
        warn = sts.get("warning", None)
        if warn is None:
            continue
        if isinstance(warn, SamplerWarning):
            log_warning(warn)
        else:
            logger.warning(warn)
    return
