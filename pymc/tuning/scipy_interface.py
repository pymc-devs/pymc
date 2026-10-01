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

from __future__ import annotations

import warnings

from collections.abc import Callable
from importlib.util import find_spec
from typing import Literal, cast, get_args

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile import Function
from pytensor.compile.mode import get_mode
from pytensor.link.jax.linker import JAXLinker
from pytensor.tensor import TensorVariable

from pymc.pytensorf import compile, floatX, join_nonshared_inputs, rewrite_pregrad

GradientBackend = Literal["pytensor", "jax"]
VALID_BACKENDS = get_args(GradientBackend)


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


def _compile_grad_and_hess_to_jax(
    f_fused: Callable, use_hess: bool, use_hessp: bool
) -> tuple[Callable, Callable | None]:
    """Derive gradient (and optionally hessian / hessp) of a JAX-compiled loss with ``jax`` autodiff."""
    import jax

    orig_loss_fn = f_fused.vm.jit_fn  # type: ignore[attr-defined]
    f_hessp = None

    if use_hess:

        @jax.jit
        def loss_fn_fused(x):
            loss_and_grad = jax.value_and_grad(lambda x: orig_loss_fn(x)[0])(x)
            hess = jax.hessian(lambda x: orig_loss_fn(x)[0])(x)
            return *loss_and_grad, hess

    else:

        @jax.jit
        def loss_fn_fused(x):
            return jax.value_and_grad(lambda x: orig_loss_fn(x)[0])(x)

    if use_hessp:

        @jax.jit
        def f_hessp(x, p):
            _, u = jax.jvp(lambda x: loss_fn_fused(x)[1], (x,), (p,))
            return jax.numpy.stack(u)

    return loss_fn_fused, f_hessp


def _compile_functions_for_scipy_optimize(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    compute_grad: bool,
    compute_hess: bool,
    compute_hessp: bool,
    compile_kwargs: dict | None = None,
) -> list[Function | None]:
    """Compile ``loss`` of one flat input into ``[f_fused, f_hessp]``, or ``[f_loss]`` without derivatives.

    ``f_fused`` returns the loss, optionally with gradient and dense Hessian; ``f_hessp`` may be None.
    """
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs
    loss = rewrite_pregrad(loss)

    if not (compute_grad or compute_hess or compute_hessp):
        return [compile(inputs, loss, **compile_kwargs)]

    [flat_input] = inputs
    f_hessp = None
    if compute_hessp:
        p = pt.tensor("p", shape=flat_input.type.shape)
        hessp = pytensor.gradient.hessian_vector_product(loss, [flat_input], p)
        f_hessp = compile([flat_input, p], hessp[0], **compile_kwargs)

    outputs = [loss]
    if compute_grad:
        grad = cast(TensorVariable, pytensor.gradient.grad(loss, flat_input))
        outputs.append(grad)
    if compute_hess:
        outputs.append(pytensor.gradient.jacobian(grad, [flat_input])[0])

    return [compile(inputs, outputs, **compile_kwargs), f_hessp]


def scipy_optimize_funcs_from_loss(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    initial_point_dict: dict[str, np.ndarray] | None = None,
    use_grad: bool | None = None,
    use_hess: bool | None = None,
    use_hessp: bool | None = None,
    gradient_backend: GradientBackend = "pytensor",
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
    gradient_backend : {"pytensor", "jax"}
        Whether derivatives are taken symbolically by pytensor or by ``jax`` autodiff on the
        compiled loss. The latter requires a JAX compile mode.
    compile_kwargs : dict, optional
        Keyword arguments passed on to :func:`pymc.compile`.
    inputs_are_flat : bool
        Set when ``inputs`` already is a single flat vector.

    Returns
    -------
    f_fused : Callable
        Returns the loss, optionally fused with the gradient and hessian.
    f_hessp : Callable or None
        Hessian-vector product function, if requested.
    """
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs.copy()

    if use_hess and not use_grad:
        raise ValueError("Cannot compute hessian without also computing the gradient")
    if gradient_backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid gradient backend: {gradient_backend}. Must be one of {VALID_BACKENDS}"
        )

    use_jax_gradients = (gradient_backend == "jax") and use_grad
    if use_jax_gradients:
        if not find_spec("jax"):
            raise ImportError("JAX must be installed to use JAX gradients")
        mode = compile_kwargs.setdefault("mode", "JAX")
        if not isinstance(get_mode(mode).linker, JAXLinker):
            raise ValueError(
                'jax gradients can only be used when ``compile_kwargs["mode"]`` is set to "JAX"'
            )

    if not isinstance(inputs, list):
        inputs = [inputs]

    if inputs_are_flat:
        [flat_input] = inputs
    else:
        outputs, flat_input = join_nonshared_inputs(
            point=initial_point_dict or {}, outputs=[loss], inputs=inputs
        )
        loss = cast(TensorVariable, outputs[0])

    if use_jax_gradients:
        # The jax autodiff path bypasses the pytensor function wrapper, so it cannot see shared variables.
        from pymc.sampling.jax import _replace_shared_variables

        [loss] = _replace_shared_variables([loss])

    compute_grad = bool(use_grad and not use_jax_gradients)
    compute_hess = bool(use_hess and not use_jax_gradients)
    compute_hessp = bool(use_hessp and not use_jax_gradients)

    funcs = _compile_functions_for_scipy_optimize(
        loss=loss,
        inputs=[flat_input],
        compute_grad=compute_grad,
        compute_hess=compute_hess,
        compute_hessp=compute_hessp,
        compile_kwargs=compile_kwargs,
    )
    f_fused: Callable = cast(Callable, funcs[0])
    f_hessp: Callable | None = funcs[1] if compute_hessp else None

    if use_jax_gradients:
        f_fused, f_hessp = _compile_grad_and_hess_to_jax(f_fused, bool(use_hess), bool(use_hessp))

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
