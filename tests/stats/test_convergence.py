#   Copyright 2024 The PyMC Developers
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

import logging

import arviz
import numpy as np

from pymc.stats import convergence


def test_warn_divergences():
    idata = arviz.from_dict(
        sample_stats={
            "diverging": np.array([[1, 0, 1, 0], [0, 0, 0, 0]]).astype(bool),
        }
    )
    warns = convergence.warn_divergences(idata)
    assert len(warns) == 1
    assert "2 divergences after tuning" in warns[0].message


def test_warn_treedepth():
    idata = arviz.from_dict(
        sample_stats={
            "reached_max_treedepth": np.array([[0, 0, 0], [0, 1, 0]]).astype(bool),
        }
    )
    warns = convergence.warn_treedepth(idata)
    assert len(warns) == 1
    assert "Chain 1 reached the maximum tree depth" in warns[0].message


def test_warn_treedepth_multiple_samplers():
    """Check we handle cases when sampling with multiple NUTS samplers, each of which reports max_treedepth."""
    max_treedepth = np.zeros((3, 2, 2), dtype=bool)
    max_treedepth[0, 0, 0] = True
    max_treedepth[2, 1, 1] = True
    idata = arviz.from_dict(
        sample_stats={
            "reached_max_treedepth": max_treedepth,
        }
    )
    warns = convergence.warn_treedepth(idata)
    assert len(warns) == 2
    assert "Chain 0 reached the maximum tree depth" in warns[0].message
    assert "Chain 2 reached the maximum tree depth" in warns[1].message


def test_log_warning_stats(caplog):
    s1 = {"warning": "Temperature too low!"}
    s2 = {"warning": "Temperature too high!"}
    stats = [s1, s2]

    with caplog.at_level(logging.WARNING):
        convergence.log_warning_stats(stats)

    # We have a list of stats dicts, because there might be several samplers involved.
    assert "too low" in caplog.records[0].message
    assert "too high" in caplog.records[1].message


def test_log_warning_stats_knows_SamplerWarning(caplog):
    """Checks that SamplerWarning "warning" stats get special treatment."""
    warn = convergence.SamplerWarning(
        convergence.WarningType.BAD_ENERGY,
        "Not that interesting",
        "debug",
    )
    stats = [{"warning": warn}]

    with caplog.at_level(logging.DEBUG, logger="pymc"):
        convergence.log_warning_stats(stats)

    assert "Not that interesting" in caplog.records[0].message
