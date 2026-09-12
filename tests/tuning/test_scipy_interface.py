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
    get_nearest_psd,
    scipy_optimize_funcs_from_loss,
    set_optimizer_function_defaults,
)


@pytest.fixture
def simple_loss_and_inputs():
    x = pt.vector("x")
    return pt.sum(x**2), [x]


def test_get_nearest_psd_returns_psd():
    psd = get_nearest_psd(np.array([[2, -3], [-3, 2]]))
    np.testing.assert_allclose(psd, psd.T)
    assert np.all(np.linalg.eigvalsh(psd) >= -1e-12)


def test_get_nearest_psd_given_psd_input():
    L = np.random.default_rng(0).normal(size=(2, 2))
    A = L @ L.T
    np.testing.assert_allclose(get_nearest_psd(A), A)


def test_set_optimizer_function_defaults_warns_and_prefers_hessp(caplog):
    with caplog.at_level("WARNING"):
        use_grad, use_hess, use_hessp = set_optimizer_function_defaults(
            "trust-ncg", True, True, True
        )
    assert caplog.messages[0].startswith('Both "use_hess" and "use_hessp" are set to True')
    assert (use_grad, use_hess, use_hessp) == (True, False, True)


def test_set_optimizer_function_defaults_infers_hess_and_hessp():
    assert set_optimizer_function_defaults("trust-ncg", None, None, True) == (True, False, True)
    assert set_optimizer_function_defaults("trust-ncg", None, True, None) == (True, True, False)
    assert set_optimizer_function_defaults("trust-ncg", None, None, None) == (True, False, True)
    assert set_optimizer_function_defaults("L-BFGS-B", None, None, None) == (True, False, False)
    assert set_optimizer_function_defaults("powell", None, None, None) == (False, False, False)


@pytest.mark.parametrize(
    "compute_grad, compute_hess, compute_hessp",
    [(False, False, False), (True, False, False), (True, True, False), (True, False, True)],
)
def test_compile_functions_for_scipy_optimize(
    simple_loss_and_inputs, compute_grad, compute_hess, compute_hessp
):
    loss, inputs = simple_loss_and_inputs
    funcs = scipy_interface._compile_functions_for_scipy_optimize(
        loss,
        inputs,
        compute_grad=compute_grad,
        compute_hess=compute_hess,
        compute_hessp=compute_hessp,
    )
    x = np.array([1.0, 2.0])
    if not compute_grad:
        [f_loss] = funcs
        assert np.isclose(f_loss(x), 5.0)
        return

    f_fused, f_hessp = funcs
    loss_val, grad_val, *rest = f_fused(x)
    assert np.isclose(loss_val, 5.0)
    np.testing.assert_allclose(grad_val, 2 * x)
    if compute_hess:
        np.testing.assert_allclose(rest[0], 2 * np.eye(2))
    if compute_hessp:
        np.testing.assert_allclose(f_hessp(x, np.array([1.0, 0.0])), [2.0, 0.0])
    else:
        assert f_hessp is None


def test_scipy_optimize_funcs_from_loss_invalid_args(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    point = {"x": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="Invalid gradient backend"):
        scipy_optimize_funcs_from_loss(loss, inputs, point, use_grad=True, gradient_backend="foo")
    with pytest.raises(ValueError, match="Cannot compute hessian without"):
        scipy_optimize_funcs_from_loss(loss, inputs, point, use_grad=False, use_hess=True)
    pytest.importorskip("jax")
    with pytest.raises(ValueError, match="jax gradients can only be used"):
        scipy_optimize_funcs_from_loss(
            loss,
            inputs,
            point,
            use_grad=True,
            gradient_backend="jax",
            compile_kwargs={"mode": "NUMBA"},
        )


@pytest.mark.parametrize("gradient_backend", ["pytensor", "jax"])
def test_scipy_optimize_funcs_from_loss_jax(gradient_backend):
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
        gradient_backend=gradient_backend,
        compile_kwargs={"mode": "JAX"},
    )
    x_val = np.array([1.0, 2.0])
    z, grad, hess = f_fused(x_val)
    np.testing.assert_allclose(z, 8.0)
    np.testing.assert_allclose(np.asarray(grad).squeeze(), [2 * x_val[0] + x_val[1], x_val[0]])
    np.testing.assert_allclose(np.asarray(hess).squeeze(), [[2, 1], [1, 0]])
    np.testing.assert_allclose(np.asarray(f_hessp(x_val, np.array([1.0, 0.0]))).squeeze(), [2, 1])
