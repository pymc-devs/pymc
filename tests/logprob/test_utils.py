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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import function
from pytensor.compile import get_default_mode
from pytensor.tensor.random.basic import normal, uniform

import pymc as pm

from pymc.logprob.abstract import MeasurableVariable
from pymc.logprob.basic import logp, transformed_conditional_logp
from pymc.logprob.utils import (
    ParameterValueError,
    check_potential_measurability,
    dirac_delta,
    rvs_to_value_vars,
    walk_model,
)
from pymc.testing import assert_no_rvs
from tests.logprob.utils import create_pytensor_params, scipy_logprob_tester


def test_walk_model():
    d = pt.vector("d")
    b = pt.vector("b")
    c = uniform(0.0, d)
    c.name = "c"
    e = pt.log(c)
    a = normal(e, b)
    a.name = "a"

    test_graph = pt.exp(a + 1)
    res = list(walk_model((test_graph,)))
    assert a in res
    assert c not in res

    res = list(walk_model((test_graph,), walk_past_rvs=True))
    assert a in res
    assert c in res

    res = list(walk_model((test_graph,), walk_past_rvs=True, stop_at_vars={e}))
    assert a in res
    assert c not in res


def test_rvs_to_value_vars():
    a = pt.random.uniform(0.0, 1.0)
    a.name = "a"
    a.tag.value_var = a_value_var = a.clone()

    b = pt.random.uniform(0, a + 1.0)
    b.name = "b"
    b.tag.value_var = b_value_var = b.clone()

    c = pt.random.normal()
    c.name = "c"
    c.tag.value_var = c_value_var = c.clone()

    d = pt.log(c + b) + 2.0

    initial_replacements = {b: b_value_var, c: c_value_var}
    (res,), replaced = rvs_to_value_vars((d,), initial_replacements=initial_replacements)

    assert res.owner.op == pt.add
    log_output = res.owner.inputs[0]
    assert log_output.owner.op == pt.log
    log_add_output = res.owner.inputs[0].owner.inputs[0]
    assert log_add_output.owner.op == pt.add
    c_output = log_add_output.owner.inputs[0]

    # We make sure that the random variables were replaced
    # with their value variables
    assert c_output == c_value_var
    b_output = log_add_output.owner.inputs[1]
    assert b_output == b_value_var

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(res)

    res_ancestors = list(walk_model((res,), walk_past_rvs=True))

    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var not in res_ancestors


def test_rvs_to_value_vars_intermediate_rv():
    """Test that function replaces values above an intermediate RV."""
    a = pt.random.uniform(0.0, 1.0)
    a.name = "a"
    a.tag.value_var = a_value_var = a.clone()

    b = pt.random.uniform(0, a + 1.0)
    b.name = "b"
    b.tag.value_var = b.clone()

    c = pt.random.normal()
    c.name = "c"
    c.tag.value_var = c_value_var = c.clone()

    d = pt.log(c + b) + 2.0

    initial_replacements = {a: a_value_var, c: c_value_var}
    (res,), replaced = rvs_to_value_vars((d,), initial_replacements=initial_replacements)

    # Assert that the only RandomVariable that remains in the graph is `b`
    res_ancestors = list(walk_model((res,), walk_past_rvs=True))

    assert (
        len(
            list(n for n in res_ancestors if n.owner and isinstance(n.owner.op, MeasurableVariable))
        )
        == 1
    )

    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors


def test_CheckParameter():
    mu = pt.constant(0)
    sigma = pt.scalar("sigma")
    x_rv = pt.random.normal(mu, sigma, name="x")
    x_vv = pt.constant(0)
    x_logp = logp(x_rv, x_vv)

    x_logp_fn = function([sigma], x_logp)
    with pytest.raises(ParameterValueError, match="sigma > 0"):
        x_logp_fn(-1)


def test_dirac_delta():
    fn = pytensor.function(
        [], dirac_delta(pt.as_tensor(1)), mode=get_default_mode().excluding("useless")
    )
    with pytest.warns(UserWarning, match=".*DiracDelta.*"):
        assert np.array_equal(fn(), 1)


@pytest.mark.parametrize(
    "dist_params, obs",
    [
        ((np.array([0, 0, 0, 0], dtype=np.float64),), np.array([0, 0.5, 1, -1], dtype=np.float64)),
        ((np.array(0, dtype=np.int64),), np.array(0, dtype=np.int64)),
    ],
)
def test_dirac_delta_logprob(dist_params, obs):
    dist_params_at, obs_at, _ = create_pytensor_params(dist_params, obs, ())
    dist_params = dict(zip(dist_params_at, dist_params))

    x = dirac_delta(*dist_params_at)

    @np.vectorize
    def scipy_logprob(obs, c):
        return 0.0 if obs == c else -np.inf

    scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


def test_check_potential_measurability():
    x1 = pt.random.normal()
    x2 = pt.random.normal()
    x3 = pt.scalar("x3")
    y = pt.exp(x1 + x2 + x3)

    # In the first three cases, y is potentially measurable, because it has at least on unvalued RV input
    assert check_potential_measurability([y], {})
    assert check_potential_measurability([y], {x1})
    assert check_potential_measurability([y], {x2})
    # y is not potentially measurable because both RV inputs are valued
    assert not check_potential_measurability([y], {x1, x2})
