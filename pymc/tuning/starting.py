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

"""Maximum a posteriori (MAP) estimation."""

import warnings

from collections.abc import Sequence
from itertools import product
from typing import Any, Literal, cast

import numpy as np
import pytensor.gradient as tg
import pytensor.tensor as pt
import xarray as xr

from better_optimize import basinhopping, minimize
from better_optimize.constants import MINIMIZE_MODE_KWARGS, minimize_method
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorVariable
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import LinearOperator
from xarray import DataTree

from pymc.backends.arviz import to_inference_data
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.blocking import DictToArrayBijection, PointType, RaveledVars
from pymc.initial_point import StartDict
from pymc.model import Model, modelcontext
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import inputvars, resolve_backend_compile_kwargs
from pymc.tuning.scipy_interface import (
    _compute_inverse_hessian,
    scipy_optimize_funcs_from_loss,
    set_optimizer_function_defaults,
)
from pymc.util import (
    RandomState,
    get_default_varnames,
    get_random_generator,
    get_value_vars_from_user_vars,
)
from pymc.vartypes import discrete_types, typefilter

__all__ = ["find_MAP"]

_METHODS = {k.lower(): k for k in MINIMIZE_MODE_KWARGS} | {"basinhopping": "basinhopping"}
_LEGACY_KWARGS = {"start": "initvals", "seed": "random_seed", "maxeval": "maxiter"}


def _canonical_method(method: str) -> str:
    try:
        return _METHODS[method.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unknown method {method!r}. Valid methods are {list(_METHODS)}")


def _value_var_names(vars: Sequence[TensorVariable], model: Model) -> list[str]:
    try:
        value_vars = get_value_vars_from_user_vars(vars, model)
    except ValueError as exc:
        # Accommodate Deterministics / Potentials by optimizing the free RVs they depend on
        value_vars = inputvars(model.replace_rvs_by_values(vars))
        if not value_vars:
            raise exc
        warnings.warn(
            "Intermediate variables (such as Deterministic or Potential) were passed. "
            "find_MAP will optimize the underlying free_RVs instead.",
            UserWarning,
        )
    return [str(var.name) for var in value_vars]


def _unpacked_names(point_map_info, model: Model) -> list[str]:
    """One coordinate-aware label per scalar element of the raveled parameter vector."""
    value_to_rv = {value.name: rv.name for value, rv in model.values_to_rvs.items()}
    names = []
    for name, shape, *_ in point_map_info:
        if not shape:
            names.append(name)
            continue
        dims = model.named_vars_to_dims.get(value_to_rv.get(name, name)) or ()
        dims = dims if len(dims) == len(shape) else (None,) * len(shape)
        axes = [
            coord if (coord := model.coords.get(dim)) is not None and len(coord) == n else range(n)
            for dim, n in zip(dims, shape)
        ]
        names.extend(f"{name}[{','.join(map(str, idx))}]" for idx in product(*axes))
    return names


def _fit_dataset(mu: RaveledVars, H_inv: np.ndarray | None, names: list[str]) -> xr.Dataset:
    data = {"mean_vector": xr.DataArray(mu.data, dims=["rows"], coords={"rows": names})}
    if H_inv is not None:
        data["covariance_matrix"] = xr.DataArray(
            H_inv, dims=["rows", "columns"], coords={"rows": names, "columns": names}
        )
    return xr.Dataset(data)


def _optimizer_result_to_dataset(
    result: OptimizeResult, method: str, names: list[str]
) -> xr.Dataset:
    """Store every field of a scipy ``OptimizeResult``, labelling per-parameter fields by ``names``."""
    if "lowest_optimization_result" in result:
        # basinhopping nests the inner optimizer's result; flatten it over the outer fields
        result = OptimizeResult(
            {k: v for k, v in result.items() if k != "lowest_optimization_result"}
            | dict(result["lowest_optimization_result"])
        )
    vec: tuple[str, ...] = ("variables",)
    mat: tuple[str, ...] = ("variables", "variables_aux")
    data = {"method": xr.DataArray(method)}
    for key, value in result.items():
        if value is None:
            continue
        if isinstance(value, LinearOperator):
            value = np.column_stack([value.matvec(e) for e in np.eye(len(names))])
        elif key == "message":
            value = str(value)
        value = np.asarray(value)
        if key in ("x", "jac", "hess", "hess_inv") and value.ndim in (1, 2):
            dims = vec if value.ndim == 1 else mat
        else:
            dims = tuple(f"{key}_dim_{i}" for i in range(value.ndim))
        data[key] = xr.DataArray(value, dims=dims)
    coords = {d: names for d in mat if any(d in da.dims for da in data.values())}
    return xr.Dataset(data, coords=coords)


def find_MAP(
    method: minimize_method | Literal["basinhopping"] = "L-BFGS-B",
    *,
    vars: Sequence[TensorVariable] | None = None,
    use_grad: bool | None = None,
    use_hess: bool | None = None,
    use_hessp: bool | None = None,
    initvals: StartDict | None = None,
    jitter: bool = True,
    jitter_max_retries: int = 10,
    random_seed: RandomState = None,
    progressbar: bool = True,
    include_transformed: bool = False,
    compute_hessian: bool = False,
    return_inferencedata: bool = True,
    idata_kwargs: dict[str, Any] | None = None,
    freeze_model: bool = True,
    model: Model | None = None,
    backend: str | None = None,
    compile_kwargs: dict | None = None,
    **optimizer_kwargs,
) -> DataTree | PointType:
    """Find the local maximum a posteriori point of a model with ``scipy.optimize``.

    `find_MAP` should not be used to initialize the NUTS sampler. Simply call
    ``pymc.sample()`` and it will automatically initialize NUTS in a better way.

    Parameters
    ----------
    method : str
        Optimization method. Any ``scipy.optimize.minimize`` method (Nelder-Mead, Powell, CG,
        BFGS, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact,
        trust-krylov, Newton-CG) or ``"basinhopping"``. Defaults to ``"L-BFGS-B"``.
    vars : list of TensorVariable, optional
        Free random variables (or their value variables) to optimize over. All other variables
        are held fixed at their initial values. Defaults to all continuous variables. Passing
        discrete variables switches to the gradient-free ``"powell"`` method.
    use_grad, use_hess, use_hessp : bool, optional
        Whether to compile and pass the gradient, hessian and hessian-vector product to the
        optimizer. ``None`` (default) chooses based on ``method``. If gradients are requested
        automatically but the model has none, ``"powell"`` is used instead.
    initvals : dict, optional
        Initial values for (transformed) variables, overriding the model defaults. Partial
        initialization is permitted, as in :func:`pymc.sample`.
    jitter : bool, default True
        Add U(-1, 1) jitter to the initial point of the optimized variables, as ``pymc.sample``
        does. This avoids getting stuck at saddle points of the default initial point (e.g.
        products of zero-centered variables). Set ``random_seed`` for reproducible results.
    jitter_max_retries : int
        Maximum number of attempts at drawing a jittered initial point with finite log-probability.
    random_seed : int, array-like of int, or Generator, optional
        Seed for jitter and stochastic optimizers (basinhopping). With a fixed seed the result is
        fully reproducible.
    progressbar : bool, default True
        Whether to display a progress bar.
    include_transformed : bool, default False
        Whether to also return the values of transformed (unconstrained) variables, e.g. ``sigma_log__``.
    compute_hessian : bool, default False
        Compute the inverse hessian at the optimum and store it as ``fit.covariance_matrix``.
        Needed for a Laplace approximation, but expensive for large models.
    return_inferencedata : bool, default True
        If True return an :class:`arviz.InferenceData` with the MAP point as a single-draw
        ``posterior`` (plus ``fit``, ``optimizer_result``, ``observed_data`` and
        ``constant_data`` groups). If False return a ``dict`` mapping variable names to values.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`.
    freeze_model : bool, default True
        Freeze data and dimension lengths before compiling, which allows constant folding and
        is required by some backends (JAX).
    model : Model (optional if in ``with`` context)
    backend : str, optional
        Computational backend, one of "numba", "c" or "jax". Defaults to the PyTensor default mode.
    compile_kwargs : dict, optional
        Keyword arguments for the compiled functions. ``compile_kwargs["mode"]`` cannot be combined
        with ``backend``. ``compile_kwargs["gradient_backend"]="jax"`` takes derivatives with
        JAX autodiff instead of PyTensor (requires the JAX backend).
    **optimizer_kwargs
        Passed on to ``scipy.optimize.minimize`` (e.g. ``maxiter``, ``tol``), or
        ``scipy.optimize.basinhopping`` when ``method="basinhopping"``, in which case
        ``minimizer_kwargs["method"]`` selects the inner optimizer (default ``"L-BFGS-B"``).

    Returns
    -------
    arviz.InferenceData or dict
        MAP estimate, see ``return_inferencedata``.
    """
    if isinstance(method, dict):
        optimizer_kwargs["start"], method = method, "L-BFGS-B"
    for old, new in _LEGACY_KWARGS.items():
        if old in optimizer_kwargs:
            warnings.warn(
                f"`{old}` is deprecated, use `{new}` instead.", FutureWarning, stacklevel=2
            )
            optimizer_kwargs[new] = optimizer_kwargs.pop(old)
    initvals = optimizer_kwargs.pop("initvals", initvals)
    random_seed = optimizer_kwargs.pop("random_seed", random_seed)
    return_raw = optimizer_kwargs.pop("return_raw", False)
    if return_raw:
        warnings.warn(
            "`return_raw` is deprecated. The optimizer result is stored in the `optimizer_result` "
            "group of the returned InferenceData.",
            FutureWarning,
            stacklevel=2,
        )
    if optimizer_kwargs.pop("progressbar_theme", None) is not None:
        warnings.warn("`progressbar_theme` is ignored by find_MAP.", FutureWarning, stacklevel=2)

    from pymc.sampling.mcmc import _init_jitter  # avoids a circular import

    model = cast(Model, modelcontext(model))
    # Resolve which value variables to optimize, in model order, before freezing reorders the graph
    names = (
        {var.name for var in model.continuous_value_vars}
        if vars is None
        else set(_value_var_names(vars, model))
    )
    var_names = [str(var.name) for var in model.value_vars if var.name in names]
    if freeze_model:
        model = freeze_dims_and_data(model)
    compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
    gradient_backend = compile_kwargs.pop("gradient_backend", "pytensor")
    idata_kwargs = {} if idata_kwargs is None else idata_kwargs

    value_vars = {var.name: var for var in model.value_vars}
    vars = [value_vars[name] for name in var_names]
    if not vars:
        raise ValueError("Model has no unobserved continuous variables.")
    discrete = typefilter(vars, discrete_types)

    rng = get_random_generator(random_seed)
    [start] = _init_jitter(
        model,
        initvals,
        [int(rng.integers(2**30))],
        jitter,
        jitter_max_retries,
        jitter_rvs=[model.values_to_rvs[var] for var in vars if var not in discrete],
    )

    method = _canonical_method(method)
    do_basinhopping = method == "basinhopping"
    minimizer_kwargs = dict(optimizer_kwargs.pop("minimizer_kwargs", {}))
    if do_basinhopping:
        method = _canonical_method(minimizer_kwargs.pop("method", "L-BFGS-B"))

    auto_grad = use_grad is None
    if auto_grad and discrete:
        warnings.warn(
            "Discrete variables are being optimized, so gradients are not available. "
            f"Using the gradient-free method 'powell' instead of '{method}'.",
            UserWarning,
        )
        method, use_grad = "powell", False
    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(
        method, use_grad, use_hess, use_hessp
    )

    loss = -cast(TensorVariable, model.logp(jacobian=False))
    if fixed := [var for var in model.value_vars if var not in vars]:
        loss = graph_replace(loss, {var: pt.constant(start[var.name], var.name) for var in fixed})
    x0 = DictToArrayBijection.map({name: start[name] for name in var_names})

    def compile_funcs(use_grad, use_hess, use_hessp):
        return scipy_optimize_funcs_from_loss(
            loss=loss,
            inputs=vars,
            initial_point_dict=DictToArrayBijection.rmap(x0),
            use_grad=use_grad,
            use_hess=use_hess,
            use_hessp=use_hessp,
            gradient_backend=gradient_backend,
            compile_kwargs=compile_kwargs,
        )

    try:
        f_fused, f_hessp = compile_funcs(use_grad, use_hess, use_hessp)
    except (NotImplementedError, tg.NullTypeGradError) as exc:
        if not (auto_grad and use_grad):
            raise
        warnings.warn(
            f"Gradient not available ({exc}). Using the gradient-free method 'powell' instead of "
            f"'{method}'.",
            UserWarning,
        )
        method, use_grad, use_hess, use_hessp = "powell", False, False, False
        f_fused, f_hessp = compile_funcs(use_grad, use_hess, use_hessp)

    out = f_fused(x0.data)
    if not np.isfinite(out[0] if isinstance(out, tuple | list) else out):
        model.check_start_vals(start)

    if do_basinhopping:
        minimizer_kwargs = {"method": method, "hessp": f_hessp, **minimizer_kwargs}
        optimizer_kwargs.setdefault("rng", rng)
        res = basinhopping(
            func=f_fused,
            x0=x0.data,
            progressbar=progressbar,
            minimizer_kwargs=minimizer_kwargs,
            **optimizer_kwargs,
        )
    else:
        res = minimize(
            f=f_fused,
            x0=x0.data,
            hessp=f_hessp,
            progressbar=progressbar,
            method=method,
            **optimizer_kwargs,
        )

    H_inv = None
    if compute_hessian:
        H_inv = _compute_inverse_hessian(res, None, f_fused, f_hessp, use_hess, method)
        if H_inv is None:  # gradient-free optimizer: compile a hessian-vector product for it
            _, f_hessp = compile_funcs(False, False, True)
            H_inv = _compute_inverse_hessian(res, None, None, f_hessp, False, method)
    x_star = RaveledVars(np.asarray(res.x), x0.point_map_info)
    point = DictToArrayBijection.rmap(x_star, start)

    # Free, transformed and deterministic values at the optimum; one-shot, so avoid a heavy compile
    unobserved = model.unobserved_value_vars
    fn = model.compile_fn(
        unobserved,
        inputs=model.value_vars,
        point_fn=False,
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    values = dict(zip([var.name for var in unobserved], fn(**point)))

    if not return_inferencedata:
        result = {
            var.name: values[var.name]
            for var in get_default_varnames(unobserved, include_transformed)
        }
        return (result, res) if return_raw else result  # type: ignore[return-value]

    trace = NDArray(
        model=model,
        fn=fn,
        var_shapes={k: v.shape for k, v in values.items()},
        var_dtypes={k: v.dtype for k, v in values.items()},
    )
    trace.setup(draws=1, chain=0)
    trace.record(point, in_warmup=False)
    trace.close()
    # Label transformed values (e.g. ``sigma_log__``) with their RV's dims when the shapes match
    dims = {
        str(value.name): list(model.named_vars_to_dims[rv.name])
        for value, rv in model.values_to_rvs.items()
        if rv.name in model.named_vars_to_dims
        and value.name in values
        and values[value.name].shape == values[rv.name].shape
    }
    idata_kwargs = {**idata_kwargs, "dims": dims | idata_kwargs.get("dims", {})}
    idata = to_inference_data(
        MultiTrace([trace]), model=model, include_transformed=include_transformed, **idata_kwargs
    )
    labels = _unpacked_names(x0.point_map_info, model)
    idata["fit"] = DataTree(dataset=_fit_dataset(x_star, H_inv, labels))
    idata["optimizer_result"] = DataTree(
        dataset=_optimizer_result_to_dataset(
            res, "basinhopping" if do_basinhopping else method, labels
        )
    )
    return (idata, res) if return_raw else idata  # type: ignore[return-value]
