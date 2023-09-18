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

import numpy as np
import pytensor
import pytest

from pytensor import function
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.graph.basic import ancestors, equal_computations
from pytensor.tensor.random.op import RandomVariable

import pymc as pm

from pymc import SymbolicRandomVariable
from pymc.distributions.transforms import Interval
from pymc.logprob.abstract import MeasurableVariable, valued_rv
from pymc.logprob.basic import logp
from pymc.logprob.utils import (
    ParameterValueError,
    check_potential_measurability,
    dirac_delta,
    replace_rvs_by_values,
)
from pymc.testing import assert_no_rvs
from tests.logprob.utils import create_pytensor_params, scipy_logprob_tester


class TestReplaceRVsByValues:
    @pytest.mark.parametrize("symbolic_rv", (False, True))
    @pytest.mark.parametrize("apply_transforms", (True, False))
    def test_basic(self, symbolic_rv, apply_transforms):
        # Interval transform between last two arguments
        interval = (
            Interval(bounds_fn=lambda *args: (args[-2], args[-1])) if apply_transforms else None
        )

        with pm.Model() as m:
            a = pm.Uniform("a", 0.0, 1.0)
            if symbolic_rv:
                raw_b = pm.Uniform.dist(0, a + 1.0)
                b = pm.Censored("b", raw_b, lower=0, upper=a + 1.0, transform=interval)
                # If not True, another distribution has to be used
                assert isinstance(b.owner.op, SymbolicRandomVariable)
            else:
                b = pm.Uniform("b", 0, a + 1.0, transform=interval)
            c = pm.Normal("c")
            d = pt.log(c + b) + 2.0

        a_value_var = m.rvs_to_values[a]
        assert m.rvs_to_transforms[a] is not None

        b_value_var = m.rvs_to_values[b]
        c_value_var = m.rvs_to_values[c]

        (res,) = replace_rvs_by_values(
            (d,),
            rvs_to_values=m.rvs_to_values,
            rvs_to_transforms=m.rvs_to_transforms,
        )

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
        # When transforms are applied, the input is the back-transformation of the value_var,
        # otherwise it is the value_var itself
        if apply_transforms:
            assert b_output != b_value_var
        else:
            assert b_output == b_value_var

        res_ancestors = list(ancestors((res,)))
        res_rv_ancestors = [
            v for v in res_ancestors if v.owner and isinstance(v.owner.op, RandomVariable)
        ]

        # There shouldn't be any `RandomVariable`s in the resulting graph
        assert len(res_rv_ancestors) == 0
        assert b_value_var in res_ancestors
        assert c_value_var in res_ancestors
        # When transforms are used, `d` depends on `a` through the back-transformation of
        # `b`, otherwise there is no direct connection between `d` and `a`
        if apply_transforms:
            assert a_value_var in res_ancestors
        else:
            assert a_value_var not in res_ancestors

    def test_intermediate_rv(self):
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
        (res,) = replace_rvs_by_values((d,), rvs_to_values=initial_replacements)

        # Assert that the only RandomVariable that remains in the graph is `b`
        res_ancestors = list(ancestors((res,)))

        assert (
            len(
                list(
                    n
                    for n in res_ancestors
                    if n.owner and isinstance(n.owner.op, MeasurableVariable)
                )
            )
            == 1
        )

        assert c_value_var in res_ancestors
        assert a_value_var in res_ancestors

    def test_unvalued_rv_model(self):
        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal.dist(x)
            z = pm.Normal("z", y)
            out = z + y

        x_value = m.rvs_to_values[x]
        z_value = m.rvs_to_values[z]

        (res,) = replace_rvs_by_values(
            (out,),
            rvs_to_values=m.rvs_to_values,
            rvs_to_transforms=m.rvs_to_transforms,
        )

        assert res.owner.op == pt.add
        assert res.owner.inputs[0] is z_value
        res_y = res.owner.inputs[1]
        # Graph should have be cloned, and therefore y and res_y should have different ids
        assert res_y is not y
        assert res_y.owner.op == pt.random.normal
        assert res_y.owner.inputs[3] is x_value

    def test_no_change_inplace(self):
        # Test that calling rvs_to_value_vars in models with nested transformations
        # does not change the original rvs in place. See issue #5172
        with pm.Model() as m:
            one = pm.LogNormal("one", mu=0)
            two = pm.LogNormal("two", mu=pt.log(one))

            # We add potentials or deterministics that are not in topological order
            pm.Potential("two_pot", two)
            pm.Potential("one_pot", one)

        before = pytensor.clone_replace(m.free_RVs)

        # This call would change the model free_RVs in place in #5172
        replace_rvs_by_values(
            m.potentials,
            rvs_to_values=m.rvs_to_values,
            rvs_to_transforms=m.rvs_to_transforms,
        )

        after = pytensor.clone_replace(m.free_RVs)
        assert equal_computations(before, after)

    @pytest.mark.parametrize("reversed", (False, True))
    def test_interdependent_transformed_rvs(self, reversed):
        # Test that nested transformed variables, whose transformed values depend on other
        # RVs are properly replaced
        with pm.Model() as m:
            transform = pm.distributions.transforms.Interval(
                bounds_fn=lambda *inputs: (inputs[-2], inputs[-1])
            )
            x = pm.Uniform("x", lower=0, upper=1, transform=transform)
            y = pm.Uniform("y", lower=0, upper=x, transform=transform)
            z = pm.Uniform("z", lower=0, upper=y, transform=transform)
            w = pm.Uniform("w", lower=0, upper=z, transform=transform)

        rvs = [x, y, z, w]
        if reversed:
            rvs = rvs[::-1]

        transform_values = replace_rvs_by_values(
            rvs,
            rvs_to_values=m.rvs_to_values,
            rvs_to_transforms=m.rvs_to_transforms,
        )

        for transform_value in transform_values:
            assert_no_rvs(transform_value)

        if reversed:
            transform_values = transform_values[::-1]
        transform_values_fn = m.compile_fn(transform_values, point_fn=False)

        x_interval_test_value = np.random.rand()
        y_interval_test_value = np.random.rand()
        z_interval_test_value = np.random.rand()
        w_interval_test_value = np.random.rand()

        # The 3 Nones correspond to unused rng, dtype and size arguments
        expected_x = transform.backward(x_interval_test_value, None, None, None, 0, 1).eval()
        expected_y = transform.backward(
            y_interval_test_value, None, None, None, 0, expected_x
        ).eval()
        expected_z = transform.backward(
            z_interval_test_value, None, None, None, 0, expected_y
        ).eval()
        expected_w = transform.backward(
            w_interval_test_value, None, None, None, 0, expected_z
        ).eval()

        np.testing.assert_allclose(
            transform_values_fn(
                x_interval__=x_interval_test_value,
                y_interval__=y_interval_test_value,
                z_interval__=z_interval_test_value,
                w_interval__=w_interval_test_value,
            ),
            [expected_x, expected_y, expected_z, expected_w],
        )


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
    x1_valued = valued_rv(x1, x1.type())
    x2 = pt.random.normal()
    x2_valued = valued_rv(x2, x2.type())
    x3 = pt.scalar("x3")

    # In the first three cases, y is potentially measurable, because it has at least on unvalued RV input
    y = pt.exp(x1 + x2 + x3)
    assert check_potential_measurability([y])
    y = pt.exp(x1_valued + x2 + x3)
    assert check_potential_measurability([y])
    y = pt.exp(x1 + x2_valued + x3)
    assert check_potential_measurability([y])
    # y is not potentially measurable because both RV inputs are valued
    y = pt.exp(x1_valued + x2_valued + x3)
    assert not check_potential_measurability([y])
