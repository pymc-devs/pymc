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
    """Test the internal _logccdf_helper function for basic correctness.

    What: Tests that _logccdf_helper correctly computes log(1 - CDF(x)),
    also known as the log survival function (logsf).

    Why: The _logccdf_helper is the internal dispatcher that routes logccdf
    computations to distribution-specific implementations. It needs to work
    with both symbolic (TensorVariable) and concrete values.

    How: Creates a Normal(0, 1) distribution and computes logccdf at values
    [0, 1]. Compares against scipy's logsf which is the reference implementation.
    Tests both symbolic input (pt.vector) and concrete input ([0, 1]).
    """
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    # Test with symbolic value input
    x_logccdf = _logccdf_helper(x, value)
    np.testing.assert_almost_equal(x_logccdf.eval({value: [0, 1]}), sp.norm(0, 1).logsf([0, 1]))

    # Test with concrete value input
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
    """Test the public pm.logccdf function for basic correctness.

    What: Tests that the public logccdf API correctly computes the log
    complementary CDF (log survival function) for a Normal distribution.

    Why: pm.logccdf is the user-facing function that wraps _logccdf_helper
    and handles IR graph rewriting when needed. It should produce correct
    results for direct RandomVariable inputs.

    How: Creates Normal(0, 1), computes logccdf at [0, 1], and compares
    against scipy.stats.norm.logsf reference values.
    - logsf(0) = log(0.5) ≈ -0.693 (50% probability of exceeding 0)
    - logsf(1) ≈ -1.84 (about 15.9% probability of exceeding 1)
    """
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    x_logccdf = logccdf(x, value)
    np.testing.assert_almost_equal(x_logccdf.eval({value: [0, 1]}), sp.norm(0, 1).logsf([0, 1]))


def test_logccdf_numerical_stability():
    """Test numerical stability of pm.logccdf in the extreme right tail.

    What: Verifies the public logccdf function is numerically stable when
    evaluating far in the distribution's tail.

    Why: This is the primary use case that motivated adding logccdf support.
    In censored/truncated distributions, we need log(1 - CDF(bound)) at the
    censoring/truncation point. When this point is far in the tail:
    - Naive: log(1 - exp(logcdf)) = log(1 - 1) = log(0) = -inf
    - Stable: Uses erfcx-based computation → correct finite value

    How: Evaluates logccdf at x=100 for Normal(0,1) and verifies:
    1. Result is finite (not -inf or nan)
    2. Result matches scipy.logsf within relative tolerance

    The expected value is approximately -5005.5, representing the log
    probability of a standard normal exceeding 100 sigma.
    Using 100 sigma future-proofs against any improvements in naive methods.
    """
    x = pm.Normal.dist(0, 1)

    far_tail_value = 100.0

    result = logccdf(x, far_tail_value).eval()
    expected = sp.norm(0, 1).logsf(far_tail_value)  # ≈ -5005.5

    # Must be finite, not -inf (which naive computation would give)
    assert np.isfinite(result)
    # Use rtol for relative tolerance (float32 has ~7 significant digits)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_logccdf_helper_fallback():
    """Test that _logccdf_helper falls back to log1mexp(logcdf) for distributions without logccdf.

    What: Verifies that the helper's NotImplementedError fallback branch is exercised
    and produces the correct graph structure.

    Why: Distributions without a registered _logccdf method should still work via
    the fallback computation log(1 - exp(logcdf)) = log1mexp(logcdf).

    How: Uses Uniform distribution (which has logcdf but no logccdf) and inspects
    the resulting computation graph. For Uniform, the graph should contain log1mexp.
    For Normal (which has logccdf), the graph should NOT contain log1mexp.
    """
    from pytensor.scalar.math import Log1mexp
    from pytensor.tensor.elemwise import Elemwise

    def graph_contains_log1mexp(var, depth=0, visited=None):
        """Recursively check if computation graph contains Log1mexp scalar op."""
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

    # Uniform has logcdf but no logccdf - should use log1mexp fallback
    uniform_rv = pm.Uniform.dist(0, 1)
    uniform_logccdf = _logccdf_helper(uniform_rv, 0.5)
    assert graph_contains_log1mexp(uniform_logccdf), "Uniform logccdf should use log1mexp fallback"

    # Normal has logccdf - should NOT use log1mexp fallback
    normal_rv = pm.Normal.dist(0, 1)
    normal_logccdf = _logccdf_helper(normal_rv, 0.5)
    assert not graph_contains_log1mexp(normal_logccdf), (
        "Normal logccdf should use specialized implementation"
    )
