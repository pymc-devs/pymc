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
import numpy as np
import pytest

import pymc as pm

from pymc.backends import _init_trace
from pymc.backends.base import _choose_chains


@pytest.mark.parametrize(
    "n_points, tune, expected_length, expected_n_traces",
    [
        ((5, 2, 2), 0, 2, 3),
        ((6, 1, 1), 1, 6, 1),
    ],
)
def test_choose_chains(n_points, tune, expected_length, expected_n_traces):
    trace_0 = np.arange(n_points[0])
    trace_1 = np.arange(n_points[1])
    trace_2 = np.arange(n_points[2])
    traces, length = _choose_chains([trace_0, trace_1, trace_2], tune=tune)
    assert length == expected_length
    assert expected_n_traces == len(traces)


class TestInitTrace:
    def test_init_trace_continuation_unsupported(self):
        with pm.Model() as pmodel:
            A = pm.Normal("A")
            B = pm.Uniform("B")
            strace = pm.backends.ndarray.NDArray(vars=[A, B])
            strace.setup(10, 0)
            strace.record({"A": 2, "B_interval__": 0.1})
            assert len(strace) == 1
            with pytest.raises(ValueError, match="Continuation of traces"):
                _init_trace(
                    expected_length=20,
                    stats_dtypes=pm.Metropolis().stats_dtypes,
                    chain_number=0,
                    trace=strace,
                    model=pmodel,
                )
