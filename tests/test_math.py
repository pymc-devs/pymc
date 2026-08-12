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


import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor.graph.basic import equal_computations

from pymc.math import (
    cartesian,
    expand_packed_triangular,
    invprobit,
    kron_dot,
    kron_solve_lower,
    kronecker,
    log1mexp,
    logdet,
    logdiffexp,
    probit,
)
from pymc.pytensorf import floatX

pytestmark = pytest.mark.filterwarnings("error")


def test_kronecker():
    np.random.seed(1)
    # Create random matrices
    [a, b, c] = [np.random.rand(3, 3 + i) for i in range(3)]

    custom = kronecker(a, b, c)  # Custom version
    nested = pt.linalg.kron(a, pt.linalg.kron(b, c))
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
    slow_ans = pt.linalg.solve_triangular(big, x, lower=True)
    # Use tricks to avoid construction of entire kronecker product
    fast_ans = kron_solve_lower(Ls, x)
    np.testing.assert_array_almost_equal(slow_ans.eval(), fast_ans.eval())


def test_probit():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(invprobit(probit(p)).eval(), p, atol=1e-5)


@pytest.mark.filterwarnings("error")
def test_log1mexp_deprecation_warnings():
    with pytest.raises(
        ValueError,
        match="log1mexp with negative_input=False is no longer supported",
    ):
        log1mexp(2, negative_input=False).eval()

    with pytest.warns(FutureWarning):
        res_1 = log1mexp(-2, negative_input=True).eval()

    res_2 = log1mexp(-2).eval()
    res_ref = pt.log1mexp(-2).eval()

    assert np.isclose(res_ref, res_1)
    assert np.isclose(res_ref, res_2)


def test_logdiffexp():
    a = pt.vector("a")
    b = pt.vector("b")
    a_test = np.log([1, 2, 3, 4])
    with np.errstate(divide="ignore"):
        b_test = np.log([0, 1, 2, 3])
    np.testing.assert_allclose(logdiffexp(a, b).eval({a: a_test, b: b_test}), 0, atol=1e-15)

    np.testing.assert_array_equal(
        logdiffexp(a, b).eval({a: [-np.inf, -np.inf, -1], b: [-1, -np.inf, -np.inf]}),
        [np.nan, -np.inf, -1],
    )


def test_logdet():
    x = pt.matrix("x")
    assert equal_computations([logdet(x)], [pt.linalg.slogdet(x)[1]])


def test_expand_packed_triangular():
    with pytest.raises(ValueError):
        x = pt.matrix("x")
        expand_packed_triangular(5, x)
    N = 5
    packed = pt.vector("packed")
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
