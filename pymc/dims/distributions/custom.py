#   Copyright 2026 - present The PyMC Developers
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
from collections.abc import Callable, Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.utils import safe_signature
from pytensor.xtensor.basic import xtensor_from_tensor
from pytensor.xtensor.random.variable import shared_rng as xtensor_shared_rng
from pytensor.xtensor.type import XTensorVariable

from pymc.dims.distributions.core import DimDistribution, expand_dist_dims
from pymc.distributions.distribution import _call_rv_op, _support_point
from pymc.logprob.abstract import _logcdf, _logprob
from pymc.model.core import new_or_existing_block_model_access


class _DimCustomDistRV(RandomVariable):
    """Minimal RandomVariable base for the black-box path.

    Only ``signature`` is set on dynamic subclasses to avoid
    ``FutureWarning`` from ``ndim_supp``/``ndims_params`` class attributes.
    """

    name = "DimCustomDistRV"
    _print_name = ("DimCustomDist", "\\operatorname{DimCustomDist}")

    @classmethod
    def rng_fn(cls, rng, *args):
        args = list(args)
        size = args.pop(-1)
        return cls._random_fn(*args, rng=rng, size=size)


def _default_not_implemented(rv_name, method_name):
    msg = (
        f"Attempted to run {method_name} on the CustomDist '{rv_name}', "
        f"but this method had not been provided when the distribution was "
        f"constructed. Please re-build your model and provide a callable "
        f"to '{rv_name}'s {method_name} keyword argument.\n"
    )

    def func(*args, **kwargs):
        raise NotImplementedError(msg)

    return func


def _default_support_point(rv, size=None, *rv_inputs):
    return pt.zeros_like(rv)


def _prep_logp_params(dist_params, param_dims, size):
    """Prepare params for logp dispatch.

    Params with non-empty dims are wrapped as XTensorVariable for dim-aware ops.
    Params without dims (None) or with empty dims (scalar) stay as plain tensors.
    """
    del size  # Unused; params may have extra batch dims from explicit_expand_dims,
    # but broadcasting handles that at the tensor level.
    result = []
    for p, dims in zip(dist_params, param_dims):
        if dims:
            result.append(xtensor_from_tensor(p, dims=dims))
        else:
            result.append(p)
    return result


class CustomDist(DimDistribution):
    """Dims-aware CustomDist for pymc.dims.

    Provides ``dist`` (symbolic) and/or ``logp`` construction paths,
    operating on ``XTensorVariable`` with named dims.

    Symbolic path (``dist`` function receives XTensorVariable params)::

        import pytensor.xtensor.math as ptxm


        def logitnormal_dist(mu, sigma):
            return ptxm.sigmoid(pmd.Normal.dist(mu=mu, sigma=sigma))


        pmd.CustomDist("x", mu, sigma, dist=logitnormal_dist, dims="city")

    When ``dist`` is provided without ``logp``, PyMC auto-derives the logp
    from the inner graph.  When ``logp`` is also given, it overrides the
    auto-derived logp while ``dist`` still drives the random path.

    Logp path (``logp`` function receives the ``value`` and all params as
    ``XTensorVariable`` — use ``ptx.*`` for dim-aware operations)::

        import pytensor.xtensor.math as ptx


        def normal_logp(value, mu, sigma):
            return ptx.sum(
                -0.5 * ((value - mu) / sigma) ** 2 - ptx.log(sigma * ptx.sqrt(2 * np.pi))
            )


        pmd.CustomDist("y", mu, sigma, logp=normal_logp, observed=y, dims="city")

    When ``logp`` is provided without ``dist``, prior/posterior predictive
    sampling is not available — only MCMC (logp evaluation).  For both,
    provide ``dist`` + ``logp`` together.

    For tensor-level operations use ``value.values`` to access the
    underlying tensor::

        import pytensor.tensor as pt


        def normal_logp(value, mu, sigma):
            v = value.values
            return pt.sum(-0.5 * ((v - mu) / sigma) ** 2 - pt.log(sigma * pt.sqrt(2 * np.pi)))


        pmd.CustomDist("y", mu, sigma, logp=normal_logp, observed=y, dims="city")
    """

    _forward_dim_lengths = True

    @classmethod
    def dist(
        cls,
        *dist_params,
        dist: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        dim_lengths: dict | None = None,
        core_dims: str | Sequence[str] | None = None,
        **kwargs,
    ):
        kwargs.update(
            dist=dist,
            logp=logp,
            logcdf=logcdf,
            support_point=support_point,
            ndim_supp=ndim_supp,
            ndims_params=ndims_params,
            signature=signature,
            dtype=dtype,
        )
        return super().dist(
            list(dist_params),
            dim_lengths=dim_lengths,
            core_dims=core_dims,
            **kwargs,
        )

    @classmethod
    def _infer_output_dims(cls, params, extra_dims, core_dims, dim_lengths=None):
        """Infer output dims from params, extra_dims and core_dims."""
        param_dims = set()
        for p in params:
            try:
                param_dims |= set(p.dims)
            except AttributeError:
                pass
        if dim_lengths:
            batch_dims = tuple(dim_lengths.keys())
        elif extra_dims:
            batch_dims = tuple(d for d in extra_dims if d in param_dims) + tuple(
                d for d in extra_dims if d not in param_dims
            )
        else:
            batch_dims = tuple(param_dims)
        return batch_dims + (core_dims if core_dims else ())

    @classmethod
    def xrv_op(
        cls,
        *dist_params,
        dist: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        class_name: str = "CustomDist",
        core_dims: str | Sequence[str] | None = None,
        extra_dims: dict[str, int] | None = None,
        rng=None,
        return_next_rng: bool = False,
        **kwargs,
    ):
        dim_lengths = kwargs.pop("dim_lengths", None)
        if dist is not None:
            return cls._symbolic_xrv_op(
                list(dist_params),
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                class_name=class_name,
                core_dims=core_dims,
                extra_dims=extra_dims or {},
                dim_lengths=dim_lengths,
                rng=rng,
                return_next_rng=return_next_rng,
            )
        else:
            return cls._blackbox_xrv_op(
                list(dist_params),
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                dtype=dtype,
                class_name=class_name,
                core_dims=core_dims,
                extra_dims=extra_dims or {},
                dim_lengths=dim_lengths,
                rng=rng,
                return_next_rng=return_next_rng,
            )

    @classmethod
    def _symbolic_xrv_op(
        cls,
        dist_params: list,
        *,
        dist: Callable,
        logp: Callable | None,
        logcdf: Callable | None,
        support_point: Callable | None,
        class_name: str,
        core_dims: str | Sequence[str] | None,
        extra_dims: dict[str, int],
        dim_lengths: dict | None,
        rng,
        return_next_rng: bool,
    ):
        xtensor_params = [cls._as_xtensor(p) for p in dist_params]

        with new_or_existing_block_model_access(
            error_msg_on_access="Model variables cannot be created in the dist function. Use the `.dist` API"
        ):
            rv = dist(*xtensor_params)

        if isinstance(rv, XTensorVariable):
            missing_extra_dims = {d: s for d, s in extra_dims.items() if d not in rv.dims}
            if missing_extra_dims:
                rv = expand_dist_dims(rv, missing_extra_dims)
        else:
            output_dims = cls._infer_output_dims(xtensor_params, extra_dims, core_dims, dim_lengths)
            rv = xtensor_from_tensor(rv, dims=output_dims)

        # If no user-provided functions to override, return the symbolic RV as-is
        if logp is None and logcdf is None and support_point is None:
            if return_next_rng:
                return xtensor_shared_rng(seed=None), rv
            return rv

        # Hybrid: use dist for sampling but user functions for logp/logcdf/support_point
        # Walk the owner chain to find the underlying RandomVariable op for its rng_fn
        tensor_rv = rv.values if isinstance(rv, XTensorVariable) else rv
        current = tensor_rv
        while current.owner is not None:
            op = current.owner.op
            if hasattr(op, "rng_fn"):
                orig_op = op
                break
            if hasattr(op, "core_op") and hasattr(op.core_op, "rng_fn"):
                orig_op = op.core_op
                break
            current = current.owner.inputs[0]
        else:
            raise ValueError(
                "Could not find a RandomVariable op in the dist graph. "
                "The dist function must return a distribution with a sampler."
            )

        # Compile the full dist output graph for sampling.
        # This is essential for compound processes (Poisson→Gamma→...) — the
        # graph contains multiple RandomVariables whose chained evaluation
        # produces correct draws from the compound, bypassing a single op's rng_fn.
        from pytensor.graph.basic import Constant
        from pytensor.graph.traversal import graph_inputs

        # Only pass variables that are actual graph inputs (skip constants/unused params)
        graph_deps = set(graph_inputs([rv.values]))
        _input_indices = [
            i
            for i, p in enumerate(xtensor_params)
            if p in graph_deps and not isinstance(p, Constant)
        ]
        sample_inputs = [xtensor_params[i] for i in _input_indices]
        _sample_fn = pytensor.function(
            inputs=sample_inputs,
            outputs=rv.values,
        )

        def random_fn(*args, rng=None, size=None):
            fn_args = [args[i] for i in _input_indices]
            result = _sample_fn(*fn_args)
            if size is not None and result.shape != tuple(size):
                result = np.broadcast_to(result, tuple(size)).copy()
            return result

        ndim_supp = getattr(orig_op, "ndim_supp", 0)
        ndims_params = [0] * len(xtensor_params)
        hybrid_signature = safe_signature(
            core_inputs_ndim=ndims_params,
            core_outputs_ndim=[ndim_supp],
        )

        output_dims = rv.type.dims if isinstance(rv, XTensorVariable) else ()
        param_dims = []
        for p in xtensor_params:
            try:
                param_dims.append(p.dims)
            except AttributeError:
                param_dims.append(None)

        rv_type = type(
            class_name,
            (_DimCustomDistRV,),
            {
                "signature": hybrid_signature,
                "dtype": str(tensor_rv.dtype),
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
                "_random_fn": random_fn,
                "_param_dims": tuple(param_dims),
                "_output_dims": output_dims,
            },
        )

        if logp is not None:

            @_logprob.register(rv_type)
            def _custom_dist_logp(op, values, rng, size, *dist_params, **kwargs):
                value_xt = xtensor_from_tensor(values[0], dims=op._output_dims)
                xtensor_params = _prep_logp_params(dist_params, op._param_dims, size)
                result = logp(value_xt, *xtensor_params)
                return result.values if isinstance(result, XTensorVariable) else result

        if logcdf is not None:

            @_logcdf.register(rv_type)
            def _custom_dist_logcdf(op, value, rng, size, *dist_params, **kwargs):
                value_xt = xtensor_from_tensor(value, dims=op._output_dims)
                xtensor_params = _prep_logp_params(dist_params, op._param_dims, size)
                result = logcdf(value_xt, *xtensor_params)
                return result.values if isinstance(result, XTensorVariable) else result

        _support_point_fn = support_point if support_point is not None else _default_support_point

        @_support_point.register(rv_type)
        def _custom_dist_support_point(op, rv, rng, size, *dist_params):
            return _support_point_fn(rv, size, *dist_params)

        size = None
        if output_dims and dim_lengths:
            size = tuple(dim_lengths[d] for d in output_dims if d in dim_lengths)
        if not size:
            size = tuple(extra_dims.values()) if extra_dims else None
        rv_op = rv_type()
        _, new_tensor_rv = _call_rv_op(
            rv_op, *[p.values for p in xtensor_params], size=size, rng=rng
        )
        rv = xtensor_from_tensor(new_tensor_rv, dims=output_dims)

        if return_next_rng:
            return rng, rv
        return rv

    @classmethod
    def _blackbox_xrv_op(
        cls,
        dist_params: list,
        *,
        logp: Callable | None,
        logcdf: Callable | None,
        support_point: Callable | None,
        ndim_supp: int | None,
        ndims_params: Sequence[int] | None,
        signature: str | None,
        dtype: str,
        class_name: str,
        core_dims: str | Sequence[str] | None,
        extra_dims: dict[str, int],
        dim_lengths: dict | None,
        rng,
        return_next_rng: bool,
    ):
        # Strip dims from XTensor params for the RandomVariable internals
        # but store the original dims so _logprob can reconstruct XTensorVariables
        tensor_params = []
        param_dims = []
        for p in dist_params:
            try:
                tensor_params.append(p.values)
                param_dims.append(p.dims)
            except AttributeError:
                tensor_params.append(p)
                param_dims.append(None)

        # Build signature if not provided
        if signature is None:
            if ndim_supp is None:
                ndim_supp = 0
            if ndims_params is None:
                ndims_params = [0] * len(tensor_params)
            signature = safe_signature(
                core_inputs_ndim=ndims_params,
                core_outputs_ndim=[ndim_supp],
            )

        # Infer output dims for the XTensor wrapping
        output_dims = cls._infer_output_dims(dist_params, extra_dims, core_dims, dim_lengths)

        # Dynamically create a RandomVariable subclass with ONLY signature
        # (no ndim_supp/ndims_params class attributes) to avoid deprecation warnings.
        # Store dims info for _logprob/_logcdf/_support_point to reconstruct
        # XTensorVariables from tensor params during logp computation.
        # NOTE: user callables (logp, logcdf, support_point) are captured in
        # closures below, NOT stored as class attributes, to avoid Python's
        # descriptor protocol binding them to the op instance.
        rv_type = type(
            class_name,
            (_DimCustomDistRV,),
            {
                "signature": signature,
                "dtype": dtype,
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
                "_random_fn": _default_not_implemented(class_name, "random"),
                "_param_dims": tuple(param_dims),
                "_output_dims": output_dims,
            },
        )

        # Dispatch logprob — reconstruct XTensor value and params
        _logp_fn = logp if logp is not None else _default_not_implemented(class_name, "logp")

        @_logprob.register(rv_type)
        def _custom_dist_logp(op, values, rng, size, *dist_params, **kwargs):
            value_xt = xtensor_from_tensor(values[0], dims=op._output_dims)
            xtensor_params = _prep_logp_params(dist_params, op._param_dims, size)
            result = _logp_fn(value_xt, *xtensor_params)
            return result.values if isinstance(result, XTensorVariable) else result

        # Dispatch logcdf (only when user provided it)
        if logcdf is not None:

            @_logcdf.register(rv_type)
            def _custom_dist_logcdf(op, value, rng, size, *dist_params, **kwargs):
                value_xt = xtensor_from_tensor(value, dims=op._output_dims)
                xtensor_params = _prep_logp_params(dist_params, op._param_dims, size)
                result = logcdf(value_xt, *xtensor_params)
                return result.values if isinstance(result, XTensorVariable) else result

        # Dispatch support_point
        _support_point_fn = support_point if support_point is not None else _default_support_point

        @_support_point.register(rv_type)
        def _custom_dist_support_point(op, rv, rng, size, *dist_params):
            return _support_point_fn(rv, size, *dist_params)

        # Convert extra_dims to size for RandomVariable
        size = tuple(extra_dims.values()) if extra_dims else None

        # Create the RV — _call_rv_op handles rng default + return_next_rng
        rv_op = rv_type()
        _, tensor_rv = _call_rv_op(rv_op, *tensor_params, size=size, rng=rng)

        # Wrap as XTensorVariable with inferred dims
        rv = xtensor_from_tensor(tensor_rv, dims=output_dims)

        if return_next_rng:
            return rng, rv
        return rv
