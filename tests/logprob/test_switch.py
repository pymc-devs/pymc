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


@pytest.mark.parametrize(
    "cond_variant,true_branch_is_scaled",
    [
        ("x_gt_0", False),
        ("x_ge_0", False),
        ("0_lt_x", False),
        ("0_le_x", False),
        ("x_lt_0", False),
        ("x_le_0", False),
        ("0_gt_x", False),
        ("0_ge_x", False),
        ("x_gt_0", True),
        ("x_ge_0", True),
        ("0_lt_x", True),
        ("0_le_x", True),
        ("x_lt_0", True),
        ("x_le_0", True),
        ("0_gt_x", True),
        ("0_ge_x", True),
    ],
)
def test_switch_non_overlapping_logp_matches_change_of_variables(
    cond_variant, true_branch_is_scaled
):
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1, size=(3,))

    if cond_variant == "x_gt_0":
        cond = x > 0
        op = "gt"
    elif cond_variant == "x_ge_0":
        cond = x >= 0
        op = "ge"
    elif cond_variant == "0_lt_x":
        cond = 0 < x
        op = "gt"
    elif cond_variant == "0_le_x":
        cond = 0 <= x
        op = "ge"
    elif cond_variant == "x_lt_0":
        cond = x < 0
        op = "lt"
    elif cond_variant == "x_le_0":
        cond = x <= 0
        op = "le"
    elif cond_variant == "0_gt_x":
        cond = 0 > x
        op = "lt"
    elif cond_variant == "0_ge_x":
        cond = 0 >= x
        op = "le"
    else:
        raise AssertionError(f"Unexpected cond_variant: {cond_variant}")

    unscaled = x
    scaled = scale * x
    true_branch = scaled if true_branch_is_scaled else unscaled
    false_branch = unscaled if true_branch_is_scaled else scaled
    y = pt.switch(cond, true_branch, false_branch)

    vv = pt.vector("vv")

    logp_y = logp(y, vv)

    if op == "gt":
        cond_v = pt.gt(vv, 0)
    elif op == "ge":
        cond_v = pt.ge(vv, 0)
    elif op == "lt":
        cond_v = pt.lt(vv, 0)
    elif op == "le":
        cond_v = pt.le(vv, 0)
    else:
        raise AssertionError(f"Unexpected op: {op}")

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
