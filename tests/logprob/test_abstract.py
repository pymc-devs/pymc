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


def test_logccdf_helper_numerical_stability():
    """Test that logccdf is numerically stable in the far right tail.

    This is where log(1 - exp(logcdf)) would lose precision because CDF is very close to 1.
    """
    x = pm.Normal.dist(0, 1)

    # Test value far in the right tail where CDF is essentially 1
    far_tail_value = 10.0

    x_logccdf = _logccdf_helper(x, far_tail_value)
    result = x_logccdf.eval()

    # scipy.stats.norm.logsf uses a numerically stable implementation
    expected = sp.norm(0, 1).logsf(far_tail_value)

    # The naive computation would give log(1 - 1) = -inf or very wrong values
    # The stable implementation should match scipy's logsf closely
    np.testing.assert_almost_equal(result, expected, decimal=6)


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
    """Test the public logccdf function."""
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    x_logccdf = logccdf(x, value)
    np.testing.assert_almost_equal(x_logccdf.eval({value: [0, 1]}), sp.norm(0, 1).logsf([0, 1]))


def test_logccdf_numerical_stability():
    """Test that pm.logccdf is numerically stable in the extreme right tail.

    For a normal distribution, the log survival function at x=10 is very negative
    (around -52). Using log(1 - exp(logcdf)) would fail because CDF(10) is essentially 1.
    """
    x = pm.Normal.dist(0, 1)

    # Test value far in the right tail
    far_tail_value = 10.0

    result = logccdf(x, far_tail_value).eval()
    expected = sp.norm(0, 1).logsf(far_tail_value)

    # Should be around -52, not -inf or nan
    assert np.isfinite(result)
    np.testing.assert_almost_equal(result, expected, decimal=6)
