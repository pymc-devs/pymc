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
import pytest

from pytensor import tensor as pt

from pymc.tuning import scipy_interface
from pymc.tuning.scipy_interface import (
    scipy_optimize_funcs_from_loss,
    set_optimizer_function_defaults,
)


@pytest.fixture
def simple_loss_and_inputs():
    x = pt.vector("x")
    return pt.sum(x**2), [x]


def test_set_optimizer_function_defaults_warns_and_prefers_hessp():
    with pytest.warns(UserWarning, match="Only one of `use_hess` and `use_hessp`"):
        flags = set_optimizer_function_defaults("trust-ncg", True, True, True)
    assert flags == (True, False, True)


@pytest.mark.parametrize(
    "method, use_hess, use_hessp, expected",
    [
        ("trust-ncg", None, None, (True, False, True)),
        ("trust-ncg", None, True, (True, False, True)),
        ("trust-ncg", True, None, (True, True, False)),
        ("trust-ncg", False, None, (True, False, True)),
        ("L-BFGS-B", None, None, (True, False, False)),
        # setting one flag must not flip the other on for a method that cannot use it
        ("L-BFGS-B", None, False, (True, False, False)),
        ("L-BFGS-B", False, None, (True, False, False)),
        ("trust-exact", None, False, (True, True, False)),
        ("powell", None, None, (False, False, False)),
    ],
)
def test_set_optimizer_function_defaults(method, use_hess, use_hessp, expected):
    assert set_optimizer_function_defaults(method, None, use_hess, use_hessp) == expected


@pytest.mark.parametrize(
    "method, flags, expected",
    [
        ("trust-exact", (None, None, True), (True, True, False)),  # keeps the Hessian it needs
        ("L-BFGS-B", (None, True, None), (True, False, False)),
        ("powell", (True, None, None), (False, False, False)),
    ],
)
def test_set_optimizer_function_defaults_ignores_unused_flags(method, flags, expected):
    with pytest.warns(UserWarning, match=f"Method '{method}' does not use"):
        assert set_optimizer_function_defaults(method, *flags) == expected


@pytest.mark.parametrize(
    "use_grad, use_hess, use_hessp",
    [(False, False, False), (True, False, False), (True, True, False), (True, False, True)],
)
def test_scipy_optimize_funcs_from_loss(simple_loss_and_inputs, use_grad, use_hess, use_hessp):
    loss, inputs = simple_loss_and_inputs
    f_fused, f_hessp = scipy_optimize_funcs_from_loss(
        loss,
        inputs,
        use_grad=use_grad,
        use_hess=use_hess,
        use_hessp=use_hessp,
        inputs_are_flat=True,
    )
    x = np.array([1.0, 2.0])
    if not use_grad:
        assert np.isclose(f_fused(x), 5.0)
        return
    loss_val, grad_val, *rest = f_fused(x)
    assert np.isclose(loss_val, 5.0)
    np.testing.assert_allclose(grad_val, 2 * x)
    if use_hess:
        np.testing.assert_allclose(rest[0], 2 * np.eye(2))
    if use_hessp:
        np.testing.assert_allclose(f_hessp(x, np.array([1.0, 0.0])), [2.0, 0.0])
    else:
        assert f_hessp is None


def test_scipy_optimize_funcs_from_loss_hess_without_grad(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    with pytest.raises(ValueError, match="Cannot compute hessian without"):
        scipy_optimize_funcs_from_loss(
            loss, inputs, {"x": np.zeros(2)}, use_grad=False, use_hess=True
        )


def test_scipy_optimize_funcs_from_loss_flat_input(simple_loss_and_inputs):
    loss, [x] = simple_loss_and_inputs
    f_fused, _ = scipy_optimize_funcs_from_loss(loss, x, use_grad=True, inputs_are_flat=True)
    loss_val, grad_val = f_fused(np.array([1.0, 2.0]))
    assert np.isclose(loss_val, 5.0)
    np.testing.assert_allclose(grad_val, [2.0, 4.0])


@pytest.mark.parametrize("use_hess", [True, False])
def test_compute_inverse_hessian_is_exact(use_hess):
    x = pt.vector("x", shape=(2,))
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    f_fused, f_hessp = scipy_optimize_funcs_from_loss(
        loss=0.5 * x @ A @ x,
        inputs=[x],
        initial_point_dict={"x": np.zeros(2)},
        use_grad=True,
        use_hess=use_hess,
        use_hessp=not use_hess,
    )
    H_inv = scipy_interface._compute_inverse_hessian(np.ones(2), f_fused, f_hessp, use_hess)
    np.testing.assert_allclose(H_inv, np.linalg.inv(A))


def test_compute_inverse_hessian_indefinite():
    x = pt.vector("x", shape=(2,))
    A = np.array([[1.0, 0.0], [0.0, -1.0]])  # saddle point: not a minimum
    _, f_hessp = scipy_optimize_funcs_from_loss(
        loss=0.5 * x @ A @ x,
        inputs=[x],
        initial_point_dict={"x": np.zeros(2)},
        use_grad=True,
        use_hessp=True,
    )
    with pytest.warns(UserWarning, match="not positive definite"):
        H_inv = scipy_interface._compute_inverse_hessian(np.zeros(2), f_hessp=f_hessp)
    assert np.all(np.isfinite(H_inv))
    np.testing.assert_allclose(H_inv[0, 0], 1.0)  # the well-defined direction is untouched
    assert np.all(np.linalg.eigvalsh(H_inv) > 0)


def test_compute_inverse_hessian_requires_second_order():
    with pytest.raises(ValueError, match="Either `f_hessp`"):
        scipy_interface._compute_inverse_hessian(np.zeros(2))


def test_scipy_optimize_funcs_from_loss_jax_mode():
    pytest.importorskip("jax")
    x = pt.tensor("x", shape=(2,))
    loss = (x[0] ** 2 + 2) + (x[0] * x[1] + 3)
    f_fused, f_hessp = scipy_optimize_funcs_from_loss(
        loss=loss,
        inputs=[x],
        initial_point_dict={"x": np.array([1.0, 2.0])},
        use_grad=True,
        use_hess=True,
        use_hessp=True,
        compile_kwargs={"mode": "JAX"},
    )
    x_val = np.array([1.0, 2.0])
    z, grad, hess = f_fused(x_val)
    np.testing.assert_allclose(z, 8.0)
    np.testing.assert_allclose(np.asarray(grad).squeeze(), [2 * x_val[0] + x_val[1], x_val[0]])
    np.testing.assert_allclose(np.asarray(hess).squeeze(), [[2, 1], [1, 0]])
    np.testing.assert_allclose(np.asarray(f_hessp(x_val, np.array([1.0, 0.0]))).squeeze(), [2, 1])
