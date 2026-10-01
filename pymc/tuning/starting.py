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

from __future__ import annotations

import warnings

from collections.abc import Sequence
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import pytensor.gradient as tg
import pytensor.tensor as pt
import xarray as xr

from pytensor.compile import Function
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorVariable
from xarray import DataTree

from pymc.backends.arviz import to_inference_data
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.blocking import DictToArrayBijection, PointType, RaveledVars
from pymc.initial_point import StartDict
from pymc.model import Model, modelcontext
from pymc.progress_bar import ProgressBarOptions
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

if TYPE_CHECKING:  # better_optimize and scipy.optimize are heavy; import on first use
    from better_optimize.constants import minimize_method
    from scipy.optimize import OptimizeResult

__all__ = ["find_MAP"]

_LEGACY_KWARGS = {"start": "initvals", "seed": "random_seed", "maxeval": "maxiter"}


def _canonical_method(method: str) -> str:
    from better_optimize.constants import MINIMIZE_MODE_KWARGS

    methods = {k.lower(): k for k in MINIMIZE_MODE_KWARGS} | {"basinhopping": "basinhopping"}
    try:
        return methods[method.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unknown method {method!r}. Valid methods are {list(methods.values())}")


def _value_vars(vars: Sequence[TensorVariable], model: Model) -> list[Variable]:
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
    return value_vars


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
    from scipy.optimize import LbfgsInvHessProduct, OptimizeResult

    if "lowest_optimization_result" in result:
        # basinhopping nests the inner optimizer's result; outer totals (nit, nfev, ...) take precedence
        inner = dict(result["lowest_optimization_result"])
        result = OptimizeResult(
            inner | {k: v for k, v in result.items() if k != "lowest_optimization_result"}
        )
    n = len(names)
    data: dict[str, xr.DataArray] = {}

    def add(key, value):
        if value is None:
            return
        if isinstance(value, LbfgsInvHessProduct):
            # L-BFGS-B's inverse Hessian is m correction pairs; densifying it is O(n^2) memory
            for suffix, pairs in (("sk", value.sk), ("yk", value.yk)):
                data[f"{key}_{suffix}"] = xr.DataArray(
                    np.asarray(pairs), dims=("lbfgs_corrections", "variables")
                )
            return
        if key == "message":
            value = str(value)
        try:
            value = np.asarray(value)
        except ValueError:  # ragged tuple, e.g. nelder-mead's (vertices, values) final_simplex
            for i, v in enumerate(value):
                add(f"{key}_{i}", v)
            return
        dims = [f"{key}_dim_{i}" for i in range(value.ndim)]
        if (
            value.ndim in (1, 2) and value.shape[-1] == n
        ):  # per-parameter axis, only when sizes match
            dims[-1] = "variables"
            if value.shape == (n, n):
                dims = ["variables", "variables_aux"]
        data[key] = xr.DataArray(value, dims=dims)

    for key, value in result.items():
        add(key, value)
    data["method"] = xr.DataArray(method)  # trust-constr reports its own sub-method under this key
    coords = {
        d: names
        for d in ("variables", "variables_aux")
        if any(d in da.dims for da in data.values())
    }
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
    progressbar: bool | ProgressBarOptions = True,
    compute_hessian: bool = False,
    return_inferencedata: bool = True,
    idata_kwargs: dict[str, Any] | None = None,
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
    progressbar : bool or ProgressBarOptions, default True
        Whether to display the optimizer's progress bar. The string options of
        :func:`pymc.sample` are accepted and simply enable it.
    compute_hessian : bool, default False
        Store the inverse Hessian of the negative ``model.logp(jacobian=False)`` at the optimum,
        taken over the optimized (unconstrained) value variables, as ``fit.covariance_matrix``.
        This needs ``n`` Hessian-vector products and an ``n x n`` matrix, so it is expensive for
        large models.
    return_inferencedata : bool, default True
        Return an :class:`arviz.InferenceData` with the MAP point as a single-draw ``posterior``
        (plus ``fit``, ``optimizer_result``, ``observed_data`` and ``constant_data`` groups).

        .. deprecated::
            ``return_inferencedata=False``, which returns a ``dict`` mapping variable names to
            values, will be removed in a future release.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`, e.g. ``include_transformed=True`` to
        also return transformed (unconstrained) values such as ``sigma_log__``.
    model : Model (optional if in ``with`` context)
        Pass a model from :func:`pymc.model.transform.freeze_model` for constant folding and
        compiled functions cached across calls.
    backend : str, optional
        Computational backend, one of "numba", "c" or "jax". Defaults to the PyTensor default mode.
    compile_kwargs : dict, optional
        Keyword arguments for the compiled functions. ``compile_kwargs["mode"]`` cannot be combined
        with ``backend``.
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
    explicit = {"initvals": initvals, "random_seed": random_seed}
    for old, new in _LEGACY_KWARGS.items():
        if old in optimizer_kwargs:
            if new in optimizer_kwargs or explicit.get(new) is not None:
                raise ValueError(f"Cannot pass both `{old}` and `{new}`; `{old}` is deprecated.")
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
    idata_kwargs = {} if idata_kwargs is None else dict(idata_kwargs)
    if "include_transformed" in optimizer_kwargs:
        if "include_transformed" in idata_kwargs:
            raise ValueError("Pass `include_transformed` only via `idata_kwargs`.")
        warnings.warn(
            "`include_transformed` is deprecated, pass it via `idata_kwargs` as in `pymc.sample`.",
            FutureWarning,
            stacklevel=2,
        )
        idata_kwargs["include_transformed"] = optimizer_kwargs.pop("include_transformed")

    if not return_inferencedata:
        warnings.warn(
            "`return_inferencedata=False` is deprecated and will be removed in a future release. "
            "Use the default `return_inferencedata=True` and work with the returned "
            "`InferenceData` object.",
            FutureWarning,
            stacklevel=2,
        )
    fit = _fit_MAP(
        method,
        vars=vars,
        use_grad=use_grad,
        use_hess=use_hess,
        use_hessp=use_hessp,
        initvals=initvals,
        jitter=jitter,
        jitter_max_retries=jitter_max_retries,
        random_seed=random_seed,
        progressbar=bool(progressbar),
        compute_hessian=compute_hessian,
        model=model,
        backend=backend,
        compile_kwargs=compile_kwargs,
        **optimizer_kwargs,
    )
    out = (
        _map_to_inference_data(fit, idata_kwargs)
        if return_inferencedata
        else fit.as_point(idata_kwargs.get("include_transformed", False))
    )
    return (out, fit.res) if return_raw else out  # type: ignore[return-value]


class _MAPFit(NamedTuple):
    model: Model
    point: PointType  # value variables at the optimum
    values: dict[str, np.ndarray]  # every unobserved value: free, transformed and deterministic
    fn: Function
    res: OptimizeResult
    x_star: RaveledVars
    H_inv: np.ndarray | None
    method: str

    def as_point(self, include_transformed: bool) -> PointType:
        names = get_default_varnames(self.values, include_transformed)
        return {name: self.values[name] for name in names}


def _fit_MAP(
    method: str,
    *,
    vars: Sequence[TensorVariable] | None,
    use_grad: bool | None,
    use_hess: bool | None,
    use_hessp: bool | None,
    initvals: StartDict | None,
    jitter: bool,
    jitter_max_retries: int,
    random_seed: RandomState,
    progressbar: bool,
    compute_hessian: bool,
    model: Model | None,
    backend: str | None,
    compile_kwargs: dict | None,
    **optimizer_kwargs,
) -> _MAPFit:
    """Run the optimization behind :func:`find_MAP`; no defaults, so they live only there."""
    from better_optimize import basinhopping, minimize

    from pymc.sampling.mcmc import _init_jitter  # avoids a circular import

    model = cast(Model, modelcontext(model))
    selected = set(model.continuous_value_vars if vars is None else _value_vars(vars, model))
    vars = [var for var in model.value_vars if var in selected]  # model order
    compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)

    if not vars:
        raise ValueError("Model has no unobserved continuous variables.")
    discrete = typefilter(vars, discrete_types)
    if compute_hessian and discrete:
        raise ValueError(
            f"`compute_hessian` is undefined for discrete variables {[v.name for v in discrete]}; "
            "exclude them via `vars`."
        )

    rng = get_random_generator(random_seed)
    [start] = _init_jitter(
        model,
        initvals,
        [int(rng.integers(2**30))],
        jitter,
        jitter_max_retries,
        jitter_rvs=[model.values_to_rvs[var] for var in vars if var not in discrete],
        # Only checks a few jittered starts for finiteness; a full backend compile is wasted here
        logp_fn=model.compile_logp(mode="FAST_COMPILE") if jitter else None,
    )

    method = _canonical_method(method)
    do_basinhopping = method == "basinhopping"
    minimizer_kwargs = dict(optimizer_kwargs.pop("minimizer_kwargs", {}))
    if do_basinhopping:
        method = _canonical_method(minimizer_kwargs.pop("method", "L-BFGS-B"))

    from better_optimize.constants import MINIMIZE_MODE_KWARGS

    auto_grad = use_grad is None
    if auto_grad and discrete and MINIMIZE_MODE_KWARGS[method]["uses_grad"]:
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
    x0 = DictToArrayBijection.map({str(v.name): start[str(v.name)] for v in vars})

    def compile_funcs(use_grad, use_hess, use_hessp):
        return scipy_optimize_funcs_from_loss(
            loss=loss,
            inputs=vars,
            initial_point_dict=DictToArrayBijection.rmap(x0),
            use_grad=use_grad,
            use_hess=use_hess,
            use_hessp=use_hessp,
            compile_kwargs=compile_kwargs,
        )

    # Compile the hessp compute_hessian needs alongside the loss, rather than recompiling the loss later
    need_hessp = compute_hessian and not use_hess
    try:
        f_fused, f_hessp = compile_funcs(use_grad, use_hess, use_hessp or need_hessp)
    except (NotImplementedError, tg.NullTypeGradError) as exc:
        if not (auto_grad and use_grad):
            raise
        warnings.warn(
            f"Gradient not available ({exc}). Using the gradient-free method 'powell' instead of "
            f"'{method}'.",
            UserWarning,
        )
        method, use_grad, use_hess, use_hessp = "powell", False, False, False
        f_fused, f_hessp = compile_funcs(use_grad, use_hess, need_hessp)

    out = f_fused(x0.data)
    if not np.isfinite(out[0] if isinstance(out, tuple | list) else out):
        model.check_start_vals(start)

    optimizer_hessp = f_hessp if use_hessp else None
    if do_basinhopping:
        minimizer_kwargs = {"method": method, "hessp": optimizer_hessp, **minimizer_kwargs}
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
            hessp=optimizer_hessp,
            progressbar=progressbar,
            method=method,
            **optimizer_kwargs,
        )

    if not res.get("success", True):
        warnings.warn(f"The optimizer did not converge: {res.get('message', '')}", UserWarning)

    H_inv = _compute_inverse_hessian(res.x, f_fused, f_hessp, use_hess) if compute_hessian else None
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

    return _MAPFit(
        model, point, values, fn, res, x_star, H_inv, "basinhopping" if do_basinhopping else method
    )


def _find_MAP_point(
    *,
    model: Model,
    initvals: StartDict | None,
    jitter: bool,
    jitter_max_retries: int,
    random_seed: RandomState,
    progressbar: bool,
    compile_kwargs: dict | None,
) -> PointType:
    """MAP point, transformed values included, for internal callers like ``init="jitter+map"``."""
    fit = _fit_MAP(
        "L-BFGS-B",
        vars=None,
        use_grad=None,
        use_hess=None,
        use_hessp=None,
        initvals=initvals,
        jitter=jitter,
        jitter_max_retries=jitter_max_retries,
        random_seed=random_seed,
        progressbar=progressbar,
        compute_hessian=False,
        model=model,
        backend=None,
        compile_kwargs=compile_kwargs,
    )
    return fit.as_point(include_transformed=True)


def _map_to_inference_data(fit: _MAPFit, idata_kwargs: dict[str, Any]) -> DataTree:
    model, values = fit.model, fit.values
    trace = NDArray(
        model=model,
        fn=fit.fn,
        var_shapes={k: v.shape for k, v in values.items()},
        var_dtypes={k: v.dtype for k, v in values.items()},
    )
    trace.setup(draws=1, chain=0)
    trace.record(fit.point, in_warmup=False)
    trace.close()
    idata = to_inference_data(MultiTrace([trace]), model=model, **idata_kwargs)
    if not idata["sample_stats"].data_vars:  # a single optimum has no sampler stats
        del idata["sample_stats"]
    labels = _unpacked_names(fit.x_star.point_map_info, model)
    idata["fit"] = DataTree(dataset=_fit_dataset(fit.x_star, fit.H_inv, labels))
    idata["optimizer_result"] = DataTree(
        dataset=_optimizer_result_to_dataset(fit.res, fit.method, labels)
    )
    return idata
