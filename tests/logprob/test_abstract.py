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

from pymc.logprob.abstract import MeasurableElemwise, MeasurableOp, _logcdf_helper
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


def test_logccdf_fallback():
    """Distributions without logccdf should fall back to log1mexp(logcdf).

    This test assumes Uniform does not implement logccdf. Implementing one would
    not be very useful since the logcdf is very simple and there are no numerical
    stability concerns. If Uniform ever gets a logccdf implementation, this test
    should be updated to use a different distribution without one.

    Before rewrites, the logccdf graph for Uniform should contain log1mexp.

    Normal implements a specialized logccdf using erfc/erfcx, so its graph, even
    before rewrites, should not contain log1mexp.
    """
    from pytensor.graph.traversal import ancestors
    from pytensor.scalar.math import Log1mexp
    from pytensor.tensor.elemwise import Elemwise

    def graph_contains_log1mexp(var):
        return any(
            v.owner
            and isinstance(v.owner.op, Elemwise)
            and isinstance(v.owner.op.scalar_op, Log1mexp)
            for v in ancestors([var])
        )

    # Uniform has no logccdf - should use fallback
    uniform_logccdf = logccdf(pm.Uniform.dist(0, 1), 0.5)
    assert graph_contains_log1mexp(uniform_logccdf)

    # Normal has logccdf - should NOT use fallback
    normal_logccdf = logccdf(pm.Normal.dist(0, 1), 0.5)
    assert not graph_contains_log1mexp(normal_logccdf)


def test_logccdf_discrete():
    mu = 3.0
    x = pm.Poisson.dist(mu=mu)

    test_values = np.array([0, 1, 2, 3, 5, 10])
    result = logccdf(x, test_values).eval()
    expected = sp.poisson(mu).logsf(test_values)

    np.testing.assert_allclose(result, expected, rtol=1e-6)
