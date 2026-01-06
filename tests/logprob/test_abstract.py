#   Copyright 2024 - present The PyMC Developers
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

import re

import numpy as np
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pytensor.scalar import Exp, exp

import pymc as pm

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableOp,
    _logccdf_helper,
    _logcdf_helper,
)
from pymc.logprob.basic import logccdf, logcdf


def assert_equal_hash(classA, classB):
    assert hash(classA) == hash(classA.id_obj)
    assert hash(classB) == hash(classB.id_obj)
    assert classA == classB
    assert hash(classA) == hash(classB)


def test_measurable_elemwise():
    # Default does not accept any scalar_op
    with pytest.raises(TypeError, match=re.escape("scalar_op exp is not valid")):
        MeasurableElemwise(exp)

    class TestMeasurableElemwise(MeasurableElemwise):
        valid_scalar_types = (Exp,)

    measurable_exp_op = TestMeasurableElemwise(scalar_op=exp)
    measurable_exp = measurable_exp_op(0.0)
    assert isinstance(measurable_exp.owner.op, MeasurableOp)


def test_logcdf_helper():
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    x_logcdf = _logcdf_helper(x, value)
    np.testing.assert_almost_equal(x_logcdf.eval({value: [0, 1]}), sp.norm(0, 1).logcdf([0, 1]))

    x_logcdf = _logcdf_helper(x, [0, 1])
    np.testing.assert_almost_equal(x_logcdf.eval(), sp.norm(0, 1).logcdf([0, 1]))


def test_logccdf_helper():
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    x_logccdf = _logccdf_helper(x, value)
    np.testing.assert_almost_equal(x_logccdf.eval({value: [0, 1]}), sp.norm(0, 1).logsf([0, 1]))

    x_logccdf = _logccdf_helper(x, [0, 1])
    np.testing.assert_almost_equal(x_logccdf.eval(), sp.norm(0, 1).logsf([0, 1]))


def test_logcdf_transformed_argument():
    with pm.Model() as m:
        sigma = pm.HalfFlat("sigma")
        x = pm.Normal("x", 0, sigma)
        pm.Potential("norm_term", -logcdf(x, 1.0))

    sigma_value_log = -1.0
    sigma_value = np.exp(sigma_value_log)
    x_value = 0.5

    observed = m.compile_logp(jacobian=False)({"sigma_log__": sigma_value_log, "x": x_value})
    expected = pm.logp(
        pm.TruncatedNormal.dist(0, sigma_value, lower=None, upper=1.0), x_value
    ).eval()
    assert np.isclose(observed, expected)


def test_logccdf():
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    x_logccdf = logccdf(x, value)
    np.testing.assert_almost_equal(x_logccdf.eval({value: [0, 1]}), sp.norm(0, 1).logsf([0, 1]))


def test_logccdf_numerical_stability():
    """Logccdf at 100 sigma should be finite, not -inf."""
    x = pm.Normal.dist(0, 1)

    result = logccdf(x, 100.0).eval()
    expected = sp.norm(0, 1).logsf(100.0)

    assert np.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_logccdf_helper_fallback():
    """Distributions without logccdf should fall back to log1mexp(logcdf)."""
    from pytensor.scalar.math import Log1mexp
    from pytensor.tensor.elemwise import Elemwise

    def graph_contains_log1mexp(var, depth=0, visited=None):
        if visited is None:
            visited = set()
        if id(var) in visited or depth > 20:
            return False
        visited.add(id(var))
        if var.owner:
            op = var.owner.op
            if isinstance(op, Elemwise) and isinstance(op.scalar_op, Log1mexp):
                return True
            for inp in var.owner.inputs:
                if graph_contains_log1mexp(inp, depth + 1, visited):
                    return True
        return False

    # Uniform has no logccdf - should use fallback
    uniform_logccdf = _logccdf_helper(pm.Uniform.dist(0, 1), 0.5)
    assert graph_contains_log1mexp(uniform_logccdf)

    # Normal has logccdf - should NOT use fallback
    normal_logccdf = _logccdf_helper(pm.Normal.dist(0, 1), 0.5)
    assert not graph_contains_log1mexp(normal_logccdf)


def test_logccdf_transformed_argument():
    with pm.Model() as m:
        sigma = pm.HalfFlat("sigma")
        x = pm.Normal("x", 0, sigma)
        pm.Potential("norm_term", logccdf(x, 1.0))

    sigma_value_log = -1.0
    sigma_value = np.exp(sigma_value_log)  # sigma ≈ 0.368
    x_value = 0.5

    observed = m.compile_logp(jacobian=False)({"sigma_log__": sigma_value_log, "x": x_value})

    # Expected = logp(x | sigma) + logccdf(Normal(0, sigma), 1.0)
    expected_logp = pm.logp(pm.Normal.dist(0, sigma_value), x_value).eval()
    expected_logsf = sp.norm(0, sigma_value).logsf(1.0)
    expected = expected_logp + expected_logsf

    assert np.isclose(observed, expected)


def test_logccdf_helper_discrete():
    """Test that logccdf computes P(X > x), not P(X >= x), for discrete RVs."""
    p = 0.7
    x = pm.Bernoulli.dist(p=p)

    np.testing.assert_almost_equal(_logccdf_helper(x, -1).eval(), 0.0)  # P(X > -1) = 1
    np.testing.assert_almost_equal(_logccdf_helper(x, 0).eval(), np.log(p))  # P(X > 0) = p
    assert _logccdf_helper(x, 1).eval() == -np.inf  # P(X > 1) = 0


def test_logccdf_discrete():
    mu = 3.0
    x = pm.Poisson.dist(mu=mu)

    test_values = np.array([0, 1, 2, 3, 5, 10])
    result = logccdf(x, test_values).eval()
    expected = sp.poisson(mu).logsf(test_values)

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_logccdf_negated_discrete():
    """Test logccdf on Y = -Bernoulli (decreasing transform)."""
    p = 0.7
    rv = -pm.Bernoulli.dist(p=p)

    np.testing.assert_almost_equal(logccdf(rv, -2).eval(), 0.0)  # P(Y > -2) = 1
    np.testing.assert_almost_equal(logccdf(rv, -1).eval(), np.log(1 - p))  # P(Y > -1) = 1-p
    assert logccdf(rv, 0).eval() == -np.inf  # P(Y > 0) = 0


def test_logccdf_shifted_discrete():
    """Test logccdf on Y = Bernoulli + 5 (increasing transform)."""
    p = 0.7
    rv = pm.Bernoulli.dist(p=p) + 5

    np.testing.assert_almost_equal(logccdf(rv, 4).eval(), 0.0)  # P(Y > 4) = 1
    np.testing.assert_almost_equal(logccdf(rv, 5).eval(), np.log(p))  # P(Y > 5) = p
    assert logccdf(rv, 6).eval() == -np.inf  # P(Y > 6) = 0
