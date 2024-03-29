#   Copyright 2024 The PyMC Developers
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

import warnings

import numpy as np
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest

from pymc.math import (
    LogDet,
    cartesian,
    expand_packed_triangular,
    invprobit,
    kron_dot,
    kron_solve_lower,
    kronecker,
    log1mexp,
    log1mexp_numpy,
    logdet,
    logdiffexp,
    logdiffexp_numpy,
    probit,
)
from pymc.pytensorf import floatX
from tests.helpers import verify_grad


def test_kronecker():
    np.random.seed(1)
    # Create random matrices
    [a, b, c] = [np.random.rand(3, 3 + i) for i in range(3)]

    custom = kronecker(a, b, c)  # Custom version
    nested = pt.slinalg.kron(a, pt.slinalg.kron(b, c))
    np.testing.assert_array_almost_equal(custom.eval(), nested.eval())  # Standard nested version


def test_cartesian():
    np.random.seed(1)
    a = [1, 2, 3]
    b = [0, 2]
    c = [5, 6]
    manual_cartesian = np.array(
        [
            [1, 0, 5],
            [1, 0, 6],
            [1, 2, 5],
            [1, 2, 6],
            [2, 0, 5],
            [2, 0, 6],
            [2, 2, 5],
            [2, 2, 6],
            [3, 0, 5],
            [3, 0, 6],
            [3, 2, 5],
            [3, 2, 6],
        ]
    )
    auto_cart = cartesian(a, b, c)
    np.testing.assert_array_equal(manual_cartesian, auto_cart)


def test_cartesian_2d():
    np.random.seed(1)
    a = [[1, 2], [3, 4]]
    b = [5, 6]
    c = [0]
    manual_cartesian = np.array(
        [
            [1, 2, 5, 0],
            [1, 2, 6, 0],
            [3, 4, 5, 0],
            [3, 4, 6, 0],
        ]
    )
    auto_cart = cartesian(a, b, c)
    np.testing.assert_array_equal(manual_cartesian, auto_cart)


def test_kron_dot():
    np.random.seed(1)
    # Create random matrices
    Ks = [np.random.rand(3, 3) for i in range(3)]
    # Create random vector with correct shape
    tot_size = np.prod([k.shape[1] for k in Ks])
    x = np.random.rand(tot_size).reshape((tot_size, 1))
    # Construct entire kronecker product then multiply
    big = kronecker(*Ks)
    slow_ans = pt.dot(big, x)
    # Use tricks to avoid construction of entire kronecker product
    fast_ans = kron_dot(Ks, x)
    np.testing.assert_array_almost_equal(slow_ans.eval(), fast_ans.eval())


def test_kron_solve_lower():
    np.random.seed(1)
    # Create random matrices
    Ls = [np.tril(np.random.rand(3, 3)) for i in range(3)]
    # Create random vector with correct shape
    tot_size = np.prod([L.shape[1] for L in Ls])
    x = np.random.rand(tot_size).reshape((tot_size, 1))
    # Construct entire kronecker product then solve
    big = kronecker(*Ls)
    slow_ans = pt.slinalg.solve_triangular(big, x, lower=True)
    # Use tricks to avoid construction of entire kronecker product
    fast_ans = kron_solve_lower(Ls, x)
    np.testing.assert_array_almost_equal(slow_ans.eval(), fast_ans.eval())


def test_probit():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(invprobit(probit(p)).eval(), p, atol=1e-5)


def test_log1mexp():
    vals = np.array([-1, 0, 1e-20, 1e-4, 10, 100, 1e20])
    vals_ = vals.copy()
    # import mpmath
    # mpmath.mp.dps = 1000
    # [float(mpmath.log(1 - mpmath.exp(-x))) for x in vals]
    expected = np.array(
        [
            np.nan,
            -np.inf,
            -46.051701859880914,
            -9.210390371559516,
            -4.540096037048921e-05,
            -3.720075976020836e-44,
            0.0,
        ]
    )
    actual = pt.log1mexp(-vals).eval()
    npt.assert_allclose(actual, expected)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value encountered in log", RuntimeWarning)
        with pytest.warns(FutureWarning, match="deprecated"):
            actual_ = log1mexp_numpy(-vals, negative_input=True)
    npt.assert_allclose(actual_, expected)
    # Check that input was not changed in place
    npt.assert_allclose(vals, vals_)


@pytest.mark.filterwarnings("error")
def test_log1mexp_numpy_no_warning():
    """Assert RuntimeWarning is not raised for very small numbers"""
    with pytest.warns(FutureWarning, match="deprecated"):
        log1mexp_numpy(-1e-25, negative_input=True)


def test_log1mexp_numpy_integer_input():
    with pytest.warns(FutureWarning, match="deprecated"):
        assert np.isclose(log1mexp_numpy(-2, negative_input=True), pt.log1mexp(-2).eval())


@pytest.mark.filterwarnings("error")
def test_log1mexp_deprecation_warnings():
    with pytest.warns(FutureWarning, match="deprecated"):
        with pytest.warns(
            FutureWarning,
            match="pymc.math.log1mexp_numpy will expect a negative input",
        ):
            res_pos = log1mexp_numpy(2)

        res_neg = log1mexp_numpy(-2, negative_input=True)

        with pytest.warns(
            FutureWarning,
            match="pymc.math.log1mexp will expect a negative input",
        ):
            res_pos_at = log1mexp(2).eval()

        res_neg_at = log1mexp(-2, negative_input=True).eval()

    assert np.isclose(res_pos, res_neg)
    assert np.isclose(res_pos_at, res_neg)
    assert np.isclose(res_neg_at, res_neg)


def test_logdiffexp():
    a = np.log([1, 2, 3, 4])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
        b = np.log([0, 1, 2, 3])
    with pytest.warns(FutureWarning, match="deprecated"):
        assert np.allclose(logdiffexp_numpy(a, b), 0)
    assert np.allclose(logdiffexp(a, b).eval(), 0)


class TestLogDet:
    def setup_method(self):
        np.random.seed(899853)
        self.op_class = LogDet
        self.op = logdet

    @pytensor.config.change_flags(compute_test_value="ignore")
    def validate(self, input_mat):
        x = pytensor.tensor.matrix()
        f = pytensor.function([x], self.op(x))
        out = f(input_mat)
        svd_diag = np.linalg.svd(input_mat, compute_uv=False)
        numpy_out = np.sum(np.log(np.abs(svd_diag)))

        # Compare the result computed to the expected value.
        np.allclose(numpy_out, out)

        # Test gradient:
        verify_grad(self.op, [input_mat])

    @pytest.mark.skipif(
        pytensor.config.device in ["cuda", "gpu"],
        reason="No logDet implementation on GPU.",
    )
    def test_basic(self):
        # Calls validate with different params
        test_case_1 = np.random.randn(3, 3) / np.sqrt(3)
        test_case_2 = np.random.randn(10, 10) / np.sqrt(10)
        self.validate(test_case_1.astype(pytensor.config.floatX))
        self.validate(test_case_2.astype(pytensor.config.floatX))


def test_expand_packed_triangular():
    with pytest.raises(ValueError):
        x = pt.matrix("x")
        x.tag.test_value = np.array([[1.0]], dtype=pytensor.config.floatX)
        expand_packed_triangular(5, x)
    N = 5
    packed = pt.vector("packed")
    packed.tag.test_value = floatX(np.zeros(N * (N + 1) // 2))
    with pytest.raises(TypeError):
        expand_packed_triangular(packed.shape[0], packed)
    np.random.seed(42)
    vals = np.random.randn(N, N)
    lower = floatX(np.tril(vals))
    lower_packed = floatX(vals[lower != 0])
    upper = floatX(np.triu(vals))
    upper_packed = floatX(vals[upper != 0])
    expand_lower = expand_packed_triangular(N, packed, lower=True)
    expand_upper = expand_packed_triangular(N, packed, lower=False)
    expand_diag_lower = expand_packed_triangular(N, packed, lower=True, diagonal_only=True)
    expand_diag_upper = expand_packed_triangular(N, packed, lower=False, diagonal_only=True)
    assert np.all(expand_lower.eval({packed: lower_packed}) == lower)
    assert np.all(expand_upper.eval({packed: upper_packed}) == upper)
    assert np.all(expand_diag_lower.eval({packed: lower_packed}) == floatX(np.diag(vals)))
    assert np.all(expand_diag_upper.eval({packed: upper_packed}) == floatX(np.diag(vals)))
