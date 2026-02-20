#   Copyright 2026 - present The PyMC Developers
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
import pytensor.tensor as pt
import pytest

from pytensor.compile.function import function

import pymc as pm

from pymc.logprob.basic import logp
from pymc.logprob.utils import ParameterValueError


def test_switch_non_overlapping_logp_matches_change_of_variables():
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1, size=(3,))
    y = pt.switch(x > 0, x, scale * x)

    vv = pt.vector("vv")

    logp_y = logp(y, vv)
    inv = pt.switch(pt.gt(vv, 0), vv, vv / scale)
    expected = logp(x, inv) + pt.switch(pt.gt(vv, 0), 0.0, -pt.log(scale))

    logp_y_fn = function([vv, scale], logp_y)
    expected_fn = function([vv, scale], expected)

    v = np.array([-2.0, 0.0, 1.5])
    np.testing.assert_allclose(logp_y_fn(v, 0.5), expected_fn(v, 0.5))

    with pytest.raises(ParameterValueError, match="switch non-overlapping scale > 0"):
        logp_y_fn(v, -0.5)

    with pytest.raises(ParameterValueError, match="switch non-overlapping scale > 0"):
        logp_y_fn(v, 0.0)


def test_switch_non_overlapping_does_not_rewrite_if_x_replicated_by_condition():
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1, size=(3,))
    cond = (x[None, :] > 0) & pt.ones((2, 1), dtype="bool")
    y = pt.switch(cond, x, scale * x)

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for Switch"):
        logp(y, np.zeros((2, 3)))


def test_switch_non_overlapping_does_not_rewrite_if_scale_broadcasts_x():
    x = pm.Normal.dist(mu=0, sigma=1)
    scale = pt.vector("scale")
    y = pt.switch(x > 0, x, scale * x)

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for Switch"):
        logp(y, np.zeros((3,)))


def test_switch_non_overlapping_does_not_apply_to_discrete_rv():
    a = pt.scalar("a")
    x = pm.Poisson.dist(3)
    y = pt.switch(x > 0, x, a * x)

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for Switch"):
        logp(y, 1)
