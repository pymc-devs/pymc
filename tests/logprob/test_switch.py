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


def _cond_x_gt_0(x):
    return x > 0


def _cond_x_ge_0(x):
    return x >= 0


def _cond_0_lt_x(x):
    return 0 < x


def _cond_0_le_x(x):
    return 0 <= x


def _cond_x_lt_0(x):
    return x < 0


def _cond_x_le_0(x):
    return x <= 0


def _cond_0_gt_x(x):
    return 0 > x


def _cond_0_ge_x(x):
    return 0 >= x


@pytest.mark.parametrize(
    "cond_builder,includes_zero_in_true,true_is_positive_side",
    [
        (_cond_x_gt_0, False, True),
        (_cond_x_ge_0, True, True),
        (_cond_0_lt_x, False, True),
        (_cond_0_le_x, True, True),
        (_cond_x_lt_0, False, False),
        (_cond_x_le_0, True, False),
        (_cond_0_gt_x, False, False),
        (_cond_0_ge_x, True, False),
    ],
    ids=[
        "x_gt_0",
        "x_ge_0",
        "0_lt_x",
        "0_le_x",
        "x_lt_0",
        "x_le_0",
        "0_gt_x",
        "0_ge_x",
    ],
)
@pytest.mark.parametrize("true_branch_is_scaled", [False, True], ids=["x_true", "scaled_true"])
def test_switch_non_overlapping_logp_matches_change_of_variables(
    cond_builder, includes_zero_in_true, true_is_positive_side, true_branch_is_scaled
):
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1, size=(3,))

    cond = cond_builder(x)

    unscaled = x
    scaled = scale * x
    true_branch = scaled if true_branch_is_scaled else unscaled
    false_branch = unscaled if true_branch_is_scaled else scaled
    y = pt.switch(cond, true_branch, false_branch)

    vv = pt.vector("vv")

    logp_y = logp(y, vv)

    if true_is_positive_side:
        cond_v = pt.ge(vv, 0) if includes_zero_in_true else pt.gt(vv, 0)
    else:
        cond_v = pt.le(vv, 0) if includes_zero_in_true else pt.lt(vv, 0)

    inv_true = vv / scale if true_branch_is_scaled else vv
    inv_false = vv if true_branch_is_scaled else vv / scale
    inv = pt.switch(cond_v, inv_true, inv_false)

    jac_true = -pt.log(scale) if true_branch_is_scaled else 0.0
    jac_false = 0.0 if true_branch_is_scaled else -pt.log(scale)
    jac = pt.switch(cond_v, jac_true, jac_false)

    expected = logp(x, inv) + jac

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
