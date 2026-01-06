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


def test_logccdf_transformed_argument():
    """Test logccdf with a transformed random variable requiring IR graph rewriting.

    What: Tests that pm.logccdf works when the random variable has been
    transformed (e.g., sigma is log-transformed), which requires the
    IR (intermediate representation) graph rewriting path.

    Why: When a random variable depends on transformed parameters, the
    direct _logccdf_helper call fails because the RV isn't in the expected
    form. The public logccdf function catches this and rewrites the graph
    using construct_ir_fgraph to make it work. This test ensures that
    fallback path is covered and correct.

    How:
    1. Creates a model where x ~ Normal(0, sigma) with sigma ~ HalfFlat
       (HalfFlat gets log-transformed automatically)
    2. Adds a Potential using logccdf(x, 1.0)
    3. Compiles and evaluates the model's logp
    4. Verifies the result equals:
       logp(Normal(0, sigma), x_value) + logsf(1.0; 0, sigma)

    The IR rewriting is triggered because x's distribution depends on
    the transformed sigma parameter.
    """
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
    """Test _logccdf_helper semantics for discrete distributions.

    What: Verifies that _logccdf_helper computes log(P(X > x)) for discrete RVs,
    which is the survival function P(X > x), NOT P(X >= x).

    Why: For discrete distributions, P(X > x) != P(X >= x). The difference is:
    - P(X > x) = 1 - P(X <= x) = 1 - CDF(x)
    - P(X >= x) = 1 - P(X < x) = 1 - P(X <= x-1) = 1 - CDF(x-1)

    The _logccdf_helper consistently computes log(P(X > x)) for both continuous
    and discrete distributions. This test verifies that behavior.

    How: Uses Bernoulli(p=0.7) where X in {0, 1}:
    - P(X > -1) = 1 (all values exceed -1)
    - P(X > 0) = P(X = 1) = 0.7 (probability of exceeding 0)
    - P(X > 1) = 0 (no value exceeds 1)
    """
    p = 0.7
    x = pm.Bernoulli.dist(p=p)

    # Test at various points
    # P(X > -1) = P(X=0) + P(X=1) = 1
    logccdf_minus1 = _logccdf_helper(x, -1).eval()
    np.testing.assert_almost_equal(logccdf_minus1, 0.0)  # log(1)

    # P(X > 0) = P(X=1) = p = 0.7
    logccdf_0 = _logccdf_helper(x, 0).eval()
    np.testing.assert_almost_equal(logccdf_0, np.log(p))

    # P(X > 1) = 0
    logccdf_1 = _logccdf_helper(x, 1).eval()
    assert logccdf_1 == -np.inf


def test_logccdf_discrete():
    """Test the public logccdf function for discrete distributions.

    What: Tests that the public pm.logccdf API works correctly for discrete
    distributions (Poisson) and computes log(P(X > x)).

    Why: Discrete distributions need log(1 - CDF(x)) computed via fallback
    since most don't have specialized logccdf implementations. This tests
    the fallback path with a distribution that has more than 2 values.

    How: Uses Poisson(mu=3) and verifies logccdf at several points against
    scipy's survival function.
    """
    mu = 3.0
    x = pm.Poisson.dist(mu=mu)

    test_values = np.array([0, 1, 2, 3, 5, 10])
    result = logccdf(x, test_values).eval()
    expected = sp.poisson(mu).logsf(test_values)

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_logccdf_negated_discrete():
    """Test logccdf on negated discrete random variable.

    What: Tests logccdf for a transformed discrete RV: Y = -X where X ~ Bernoulli.
    This exercises the special handling in measurable_transform_logcdf.

    Why: For a decreasing transform like negation, the CDF relationship is:
    P(Y <= y) = P(-X <= y) = P(X >= -y)

    For discrete X, P(X >= t) = 1 - CDF(t-1), NOT 1 - CDF(t).
    This is why measurable_transform_logcdf uses backward_value - 1 for
    discrete distributions when computing logccdf.

    How: For Y = -Bernoulli(p=0.7), Y in {-1, 0}:
    - P(Y = -1) = p = 0.7
    - P(Y = 0) = 1 - p = 0.3

    CCDF (survival function) P(Y > y):
    - P(Y > -2) = 1 (all values exceed -2)
    - P(Y > -1) = P(Y = 0) = 0.3
    - P(Y > 0) = 0 (no value exceeds 0)
    """
    p = 0.7
    rv = -pm.Bernoulli.dist(p=p)

    # P(Y > -2) = 1
    np.testing.assert_almost_equal(logccdf(rv, -2).eval(), 0.0)

    # P(Y > -1) = P(Y = 0) = 1 - p = 0.3
    np.testing.assert_almost_equal(logccdf(rv, -1).eval(), np.log(1 - p))

    # P(Y > 0) = 0
    assert logccdf(rv, 0).eval() == -np.inf


def test_logccdf_shifted_discrete():
    """Test logccdf on shifted discrete random variable.

    What: Tests logccdf for Y = X + 5 where X ~ Bernoulli (an increasing transform).

    Why: For an increasing transform (addition), no special discrete handling
    is needed since P(Y <= y) = P(X <= y - 5) = CDF_X(y - 5) directly.
    The CCDF is just 1 - CDF.

    How: For Y = Bernoulli(p=0.7) + 5, Y in {5, 6}:
    - P(Y = 5) = 1 - p = 0.3
    - P(Y = 6) = p = 0.7

    CCDF P(Y > y):
    - P(Y > 4) = 1
    - P(Y > 5) = P(Y = 6) = 0.7
    - P(Y > 6) = 0
    """
    p = 0.7
    rv = pm.Bernoulli.dist(p=p) + 5

    # P(Y > 4) = 1
    np.testing.assert_almost_equal(logccdf(rv, 4).eval(), 0.0)

    # P(Y > 5) = P(Y = 6) = p = 0.7
    np.testing.assert_almost_equal(logccdf(rv, 5).eval(), np.log(p))

    # P(Y > 6) = 0
    assert logccdf(rv, 6).eval() == -np.inf
