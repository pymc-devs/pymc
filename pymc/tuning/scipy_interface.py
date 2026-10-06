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

"""Compile model log-densities into the callables expected by ``scipy.optimize``."""

import warnings

from collections.abc import Callable
from typing import cast

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable

from pymc.pytensorf import compile, floatX, join_nonshared_inputs, rewrite_pregrad


def set_optimizer_function_defaults(
    method: str, use_grad: bool | None, use_hess: bool | None, use_hessp: bool | None
) -> tuple[bool, bool, bool]:
    """Resolve ``None`` flags from what ``method`` uses, ignoring flags it can't use.

    Of a Hessian and a Hessian-vector product only one is used: the explicit one, else ``hessp``.
    """
    from better_optimize.constants import MINIMIZE_MODE_KWARGS

    info = MINIMIZE_MODE_KWARGS[method]
    for flag, value, key in (
        ("use_grad", use_grad, "uses_grad"),
        ("use_hess", use_hess, "uses_hess"),
        ("use_hessp", use_hessp, "uses_hessp"),
    ):
        if value and not info[key]:
            warnings.warn(
                f"Method {method!r} does not use `{flag}`; it will be ignored.", UserWarning
            )
    hess_was_none, hessp_was_none = use_hess is None, use_hessp is None
    use_grad = info["uses_grad"] and (info["uses_grad"] if use_grad is None else use_grad)
    use_hess = info["uses_hess"] and (info["uses_hess"] if hess_was_none else use_hess)
    use_hessp = info["uses_hessp"] and (info["uses_hessp"] if hessp_was_none else use_hessp)
    if use_hess and use_hessp:
        if not (hess_was_none or hessp_was_none):
            warnings.warn(
                "Only one of `use_hess` and `use_hessp` is used; using `use_hessp`.", UserWarning
            )
        if hessp_was_none and not hess_was_none:
            use_hessp = False
        else:
            use_hess = False
    return bool(use_grad), bool(use_hess), bool(use_hessp)


def scipy_optimize_funcs_from_loss(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    initial_point_dict: dict[str, np.ndarray] | None = None,
    use_grad: bool | None = None,
    use_hess: bool | None = None,
    use_hessp: bool | None = None,
    compile_kwargs: dict | None = None,
    inputs_are_flat: bool = False,
) -> tuple[Callable, Callable | None]:
    """Compile ``loss`` into scipy-compatible ``(f_fused, f_hessp)`` callables of one flat vector.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss to minimize.
    inputs : list of TensorVariable
        Input variables, joined into a single raveled vector unless ``inputs_are_flat``.
    initial_point_dict : dict, optional
        Maps input names to values; only used to determine input shapes.
    use_grad, use_hess, use_hessp : bool, optional
        Which derivatives to compile into the returned functions.
    compile_kwargs : dict, optional
        Keyword arguments passed on to :func:`pymc.compile`.
    inputs_are_flat : bool
        Set when ``inputs`` already is a single flat vector.

    Returns
    -------
    f_fused : Callable
        Returns the loss, or ``(loss, grad)`` / ``(loss, grad, hess)`` when derivatives are requested.
    f_hessp : Callable or None
        Hessian-vector product function, if requested.
    """
    if use_hess and not use_grad:
        raise ValueError("Cannot compute hessian without also computing the gradient")
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs
    if not isinstance(inputs, list):
        inputs = [inputs]
    if inputs_are_flat:
        [flat_input] = inputs
    else:
        outputs, flat_input = join_nonshared_inputs(
            point=initial_point_dict or {}, outputs=[loss], inputs=inputs
        )
        loss = cast(TensorVariable, outputs[0])
    loss = rewrite_pregrad(loss)

    def trusted(fn):  # scipy hands over float64; cast here so PyTensor can skip its input checks
        fn.trust_input = True
        return lambda *args: fn(*(np.asarray(arg, dtype=flat_input.dtype) for arg in args))

    f_hessp = None
    if use_hessp:
        p = pt.tensor("p", shape=flat_input.type.shape, dtype=flat_input.dtype)
        hessp = pytensor.gradient.hessian_vector_product(loss, [flat_input], p)
        f_hessp = trusted(compile([flat_input, p], hessp[0], **compile_kwargs))

    outputs = [loss]
    if use_grad:
        grad = cast(TensorVariable, pytensor.gradient.grad(loss, flat_input))
        outputs.append(grad)
    if use_hess:
        outputs.append(pytensor.gradient.jacobian(grad, [flat_input])[0])
    f_fused = trusted(
        compile([flat_input], outputs if len(outputs) > 1 else loss, **compile_kwargs)
    )
    return f_fused, f_hessp


def _compute_inverse_hessian(
    optimal_point: np.ndarray,
    f_fused: Callable | None = None,
    f_hessp: Callable | None = None,
    use_hess: bool = False,
) -> np.ndarray:
    """Inverse of the exact Hessian at ``optimal_point`` (never an optimizer's approximation).

    Eigenvalues are clipped to a relative tolerance, warning when clearly negative (not a minimum).
    """
    x_star = floatX(np.asarray(optimal_point))
    if use_hess and f_fused is not None:
        _, _, H = f_fused(x_star)
    elif f_hessp is not None:
        basis = floatX(np.eye(len(x_star)))
        H = np.stack([np.asarray(f_hessp(x_star, e)) for e in basis], axis=-1)
    else:
        raise ValueError("Either `f_hessp` or a fused hessian (`use_hess=True`) is required.")
    H = np.asarray(H, dtype="float64")
    eigval, eigvec = np.linalg.eigh((H + H.T) / 2)
    # Same relative tolerance as np.linalg.matrix_rank, at the precision H was computed in
    tol = max(
        np.abs(eigval).max() * len(eigval) * np.finfo(pytensor.config.floatX).eps,
        np.finfo("float64").tiny,
    )
    if eigval.min() < -tol:
        warnings.warn(
            f"The Hessian at the optimum is not positive definite (smallest eigenvalue {eigval.min():.3g}), "
            "so the point may be a saddle point rather than a minimum. Its eigenvalues were clipped to "
            "compute `fit.covariance_matrix`.",
            UserWarning,
        )
    return (eigvec / np.maximum(eigval, tol)) @ eigvec.T
