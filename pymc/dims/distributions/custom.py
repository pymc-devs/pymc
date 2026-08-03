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

import pytensor.tensor as pt

from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.utils import safe_signature
from pytensor.xtensor.basic import xtensor_from_tensor
from pytensor.xtensor.random.variable import shared_rng as xtensor_shared_rng
from pytensor.xtensor.type import XTensorVariable

from pymc import SymbolicRandomVariable
from pymc.dims.distributions.core import DimDistribution, expand_dist_dims
from pymc.distributions.custom import _infer_final_signature
from pymc.distributions.distribution import _call_rv_op, _support_point
from pymc.logprob.abstract import _logcdf, _logprob
from pymc.model.core import new_or_existing_block_model_access
from pymc.pytensorf import collect_default_updates


class DimSymbolicRandomVariable(SymbolicRandomVariable):
    """XTensor-aware SymbolicRandomVariable for dims-supporting CustomDist.

    Stores output and param dims so that ``_logprob`` dispatch can
    reconstruct ``XTensorVariable`` inputs from the tensor-level graph.
    """

    default_output = 0
    _output_dims: tuple[str, ...] = ()
    _param_dims: tuple[tuple[str, ...] | None, ...] = ()


class _DimCustomDistRV(RandomVariable):
    """Minimal RandomVariable base for the arbitrarily-defined path.

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
    del size
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


        with pm.Model(coords={"city": range(5)}):
            pmd.CustomDist("x", mu, sigma, dist=logitnormal_dist, dims="city")

    When ``dist`` is provided without ``logp``, PyMC auto-derives the logp
    from the inner graph (via ``DimSymbolicRandomVariable`` with
    ``inline_logprob=True``).  When ``logp`` is also given, it overrides the
    auto-derived logp while ``dist`` still drives the random path.

    Arbitrarily-defined logp path (``logp`` function receives the ``value``
    and all params as ``XTensorVariable``)::

        import pytensor.xtensor.math as ptx


        def normal_logp(value, mu, sigma):
            return ptx.sum(
                -0.5 * ((value - mu) / sigma) ** 2 - ptx.log(sigma * ptx.sqrt(2 * np.pi))
            )


        with pm.Model(coords={"city": range(5)}):
            pmd.CustomDist("y", mu, sigma, logp=normal_logp, dims="city")

    When ``logp`` is provided without ``dist``, prior/posterior predictive
    sampling is not available — only MCMC (logp evaluation).  For both,
    provide ``dist`` + ``logp`` together.
    """

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
        if extra_dims:
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
        kwargs.pop("dim_lengths", None)
        if dist is not None:
            return cls._symbolic_xrv_op(
                list(dist_params),
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                class_name=class_name,
                core_dims=core_dims,
                extra_dims=extra_dims or {},
                rng=rng,
                return_next_rng=return_next_rng,
            )
        else:
            return cls._arbitrary_xrv_op(
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
        ndim_supp: int | None,
        ndims_params: Sequence[int] | None,
        signature: str | None,
        class_name: str,
        core_dims: str | Sequence[str] | None,
        extra_dims: dict[str, int],
        rng,
        return_next_rng: bool,
    ):
        xtensor_params = [cls._as_xtensor(p) for p in dist_params]

        with new_or_existing_block_model_access(
            error_msg_on_access="Model variables cannot be created in the dist function. Use the `.dist` API"
        ):
            rv = dist(*xtensor_params)

        if not isinstance(rv, XTensorVariable):
            raise TypeError(
                "The `dist` function must return an XTensorVariable. "
                "Use `pmd.Normal.dist(...)` or `xtensor_from_tensor(rv, dims=...)` "
                "to ensure dims are attached to the output."
            )

        missing_extra_dims = {d: s for d, s in extra_dims.items() if d not in rv.dims}
        if missing_extra_dims:
            rv = expand_dist_dims(rv, missing_extra_dims)

        # If no user-provided functions to override, return the symbolic RV as-is.
        # This avoids the DimSymbolicRandomVariable OpFromGraph wrapper for the
        # common case where logp is auto-derived from the inner graph.
        if logp is None and logcdf is None and support_point is None:
            if return_next_rng:
                return xtensor_shared_rng(seed=None), rv
            return rv

        output_dims = rv.type.dims
        param_dims = tuple(p.dims for p in xtensor_params)

        # Build dummy inner graph (same tensor types as actual params)
        dummy_tensor_params = [p.values.type() for p in xtensor_params]
        dummy_xtensor_params = [
            xtensor_from_tensor(p, dims=pd) for p, pd in zip(dummy_tensor_params, param_dims)
        ]
        with new_or_existing_block_model_access(
            error_msg_on_access="Model variables cannot be created in the dist function. Use the `.dist` API"
        ):
            dummy_rv_xt = dist(*dummy_xtensor_params)

        if not isinstance(dummy_rv_xt, XTensorVariable):
            raise TypeError(
                "The `dist` function must return an XTensorVariable. "
                "Use `pmd.Normal.dist(...)` or `xtensor_from_tensor(rv, dims=...)` "
                "to ensure dims are attached to the output."
            )

        if missing_extra_dims:
            dummy_rv_xt = expand_dist_dims(dummy_rv_xt, missing_extra_dims)

        tensor_dummy_rv = dummy_rv_xt.values

        # Find RNG updates from inner graph
        updates = collect_default_updates(inputs=dummy_tensor_params, outputs=(tensor_dummy_rv,))
        if updates:
            rngs, rngs_updates = zip(*updates.items())
        else:
            rngs, rngs_updates = (), ()

        # Build extended signature
        if ndims_params is None:
            ndims_params = [0] * len(xtensor_params)
        if ndim_supp is None:
            ndim_supp = 0
        sig = safe_signature(
            core_inputs_ndim=ndims_params,
            core_outputs_ndim=[ndim_supp],
        )
        n_inputs = len(dummy_tensor_params) + len(rngs)
        n_outputs = 1 + len(rngs_updates)
        n_rngs = len(rngs)
        extended_sig = _infer_final_signature(sig, n_inputs, n_outputs, n_rngs, add_size=False)

        # Create DimSymbolicRandomVariable subclass
        rv_type = type(
            class_name,
            (DimSymbolicRandomVariable,),
            {
                "inline_logprob": False,
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
                "_output_dims": output_dims,
                "_param_dims": param_dims,
            },
        )

        n_params = len(param_dims)

        if logp is not None:

            @_logprob.register(rv_type)
            def _custom_dist_logp(op, values, *inputs, **kwargs):
                [value] = values
                value_xt = xtensor_from_tensor(value, dims=op._output_dims)
                xtensor_params = _prep_logp_params(
                    list(inputs[:n_params]), op._param_dims, size=None
                )
                result = logp(value_xt, *xtensor_params)
                return result.values if isinstance(result, XTensorVariable) else result

        if logcdf is not None:

            @_logcdf.register(rv_type)
            def _custom_dist_logcdf(op, value, *inputs, **kwargs):
                value_xt = xtensor_from_tensor(value, dims=op._output_dims)
                xtensor_params = _prep_logp_params(
                    list(inputs[:n_params]), op._param_dims, size=None
                )
                result = logcdf(value_xt, *xtensor_params)
                return result.values if isinstance(result, XTensorVariable) else result

        _support_point_fn = support_point if support_point is not None else _default_support_point

        @_support_point.register(rv_type)
        def _custom_dist_support_point(op, rv, *inputs):
            return _support_point_fn(rv, None, *inputs[:n_params])

        # Build OpFromGraph
        # strict=False because the inner graph may reference shared
        # dim_lengths variables (extra_dims from XRV nodes).
        ofg_inputs = [*dummy_tensor_params, *rngs]
        ofg_outputs = [tensor_dummy_rv, *rngs_updates]
        rv_op = rv_type(
            inputs=ofg_inputs,
            outputs=ofg_outputs,
            extended_signature=extended_sig,
            strict=False,
        )

        # Call with concrete inputs
        tensor_params = [p.values for p in xtensor_params]
        if rng is not None:
            if len(rngs) != 1:
                raise ValueError(
                    f"CustomDist received an explicit rng but it requires {len(rngs)} rngs."
                )
            actual_rngs = (rng,)
        else:
            actual_rngs = rngs

        result = rv_op(*tensor_params, *actual_rngs)

        # result is the sample (default_output=0), RNG update is first output
        # of the Apply node.
        # Wrap as XTensorVariable with output dims.
        rv_out = xtensor_from_tensor(result, dims=output_dims)

        if return_next_rng:
            if actual_rngs:
                next_rng = rv_out.owner.outputs[0]
            else:
                next_rng = xtensor_shared_rng(seed=None)
            return next_rng, rv_out
        return rv_out

    @classmethod
    def _arbitrary_xrv_op(
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
        output_dims = cls._infer_output_dims(dist_params, extra_dims, core_dims)

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
