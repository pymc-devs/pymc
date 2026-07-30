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
from typing import Any

import pytensor.tensor as pt

from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.tensor.basic import infer_static_shape
from pytensor.xtensor import as_xtensor, broadcast
from pytensor.xtensor import zeros_like as x_zeros_like
from pytensor.xtensor.basic import xtensor_from_tensor
from pytensor.xtensor.random import shared_rng
from pytensor.xtensor.random.type import xrandom_generator_type
from pytensor.xtensor.type import XTensorVariable

from pymc.dims.distributions.core import DimDistribution, DimSymbolicRandomVariable
from pymc.distributions.custom import default_not_implemented
from pymc.distributions.distribution import _support_point
from pymc.logprob.abstract import _logcdf, _logprob
from pymc.model.core import new_or_existing_block_model_access
from pymc.pytensorf import collect_default_updates

BLOCK_MODEL_ACCESS_ERROR_MSG = (
    "Model variables cannot be created in the dist function. Use the `.dist` API"
)

DIST_NOT_XTENSOR_ERROR_MSG = (
    "The `dist` function must return an XTensorVariable. "
    "Use `pmd.Normal.dist(...)` or `xtensor_from_tensor(rv, dims=...)` "
    "to ensure dims are attached to the output."
)


class NonRandomCustomDist(Op):
    """Placeholder Op for a CustomDist defined without a dist function.

    Takes its shape as explicit scalar inputs and is only useful for its type
    information; evaluating it raises.
    """

    __props__ = ("dtype",)

    def __init__(self, dtype: str):
        self.dtype = config.floatX if dtype == "floatX" else dtype

    def make_node(self, *shape):
        shape, static_shape = infer_static_shape(shape)
        out = pt.tensor(dtype=self.dtype, shape=static_shape)
        return Apply(self, list(shape), [out])

    def infer_shape(self, fgraph, node, input_shapes):
        return [node.inputs]

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError(
            "Attempted to draw values from a CustomDist that was constructed without a "
            "`dist` function. Please re-build your model and provide a callable to the "
            "`dist` keyword argument to allow forward sampling."
        )


def _non_random_dist_maker(dtype: str) -> Callable:
    """Create a placeholder dist for a CustomDist defined without a dist function.

    The returned function gives a placeholder variable the extra dims and the
    batch dims of the params, so that everything but forward sampling works.
    """

    def non_random_dist(*args):
        *xtensor_params, extra_dims = args
        dim_lengths = dict(extra_dims)
        for param in xtensor_params:
            for dim, length in zip(param.type.dims, tuple(param.values.shape)):
                dim_lengths.setdefault(dim, length)
        placeholder = NonRandomCustomDist(dtype=dtype)(*dim_lengths.values())
        return xtensor_from_tensor(placeholder, dims=tuple(dim_lengths))

    return non_random_dist


class DimCustomDistRV(DimSymbolicRandomVariable):
    """Dims-native demarcation of a CustomDist random graph.

    The inputs are the params (with their own dims), the extra dim lengths,
    and the piped RNG. User-provided logp/logcdf/support_point functions are
    dispatched on subclasses of this Op, with xtensor semantics.
    """


class CustomDist(DimDistribution):
    """Dims-aware CustomDist for pymc.dims.

    The ``dist`` function receives the parameters as ``XTensorVariable``,
    followed by ``extra_dims``: a dict mapping the dims requested via ``dims``
    or ``observed`` that are not implied by any parameter to their lengths.
    It is the counterpart of the ``size`` argument of :class:`~pymc.CustomDist`,
    and the same argument used when implementing a new ``pymc.dims``
    distribution. The function must return an ``XTensorVariable`` random graph
    with those dims, built from other ``pymc.dims`` distributions and/or
    ``pytensor.xtensor`` operations::

        import pytensor.xtensor.math as ptxm


        def logitnormal_dist(mu, sigma, extra_dims):
            return ptxm.sigmoid(pmd.Normal.dist(mu=mu, sigma=sigma, dim_lengths=extra_dims))


        with pm.Model(coords={"city": range(5)}):
            pmd.CustomDist("x", mu, sigma, dist=logitnormal_dist, dims="city")

    When only ``dist`` is provided, the logp is derived automatically from the
    graph, exactly as it is for the built-in ``pymc.dims`` distributions.

    User-provided ``logp``/``logcdf``/``support_point`` override the derived
    ones. They are dispatched with xtensor semantics: the value, the random
    variable and all the params are ``XTensorVariable`` with named dims, and
    the functions may return an ``XTensorVariable`` (or a plain tensor).

    A distribution can also be defined without ``dist``, through ``logp``
    (and/or ``logcdf``, ``support_point``). The random graph is then a
    placeholder with the extra dims and the batch dims of the parameters,
    which raises if forward sampling is attempted::

        with pm.Model(coords={"city": range(5)}):
            pmd.CustomDist("y", mu, sigma, logp=normal_logp, dims="city")
    """

    @classmethod
    def dist(
        cls,
        *dist_params,
        dist: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
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
            dtype=dtype,
        )
        return super().dist(
            list(dist_params),
            dim_lengths=dim_lengths,
            core_dims=core_dims,
            **kwargs,
        )

    @classmethod
    def xrv_op(
        cls,
        *dist_params,
        dist: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        dtype: str = "floatX",
        class_name: str = "CustomDist",
        core_dims: str | Sequence[str] | None = None,
        extra_dims: dict[str, Any] | None = None,
        rng=None,
        # The next rng is always returned; the argument exists only while
        # the pytensor XRV constructors transition to doing the same
        return_next_rng: bool = True,
    ):
        assert return_next_rng, "return_next_rng=False is not supported"
        # core_dims is not needed: all dims, core or batched, are known from
        # the dist function output.
        xtensor_params = [cls._as_xtensor(p) for p in dist_params]
        extra_dims = extra_dims or {}
        extra_dim_names = tuple(extra_dims)

        if dist is None:
            if logp is None:
                # Match the lazy failure of pm.CustomDist when the logp is requested
                logp = default_not_implemented(class_name, "logp")
            dist = _non_random_dist_maker(dtype)

        # Build the inner graph on dummy inputs. Like the size argument of
        # pm.CustomDist, extra_dims (the dims requested via dims/observed that
        # are not implied by any param) are passed as the last argument of the
        # dist function, which is responsible for using them.
        dummy_params = [param.type() for param in xtensor_params]
        dummy_extra_lengths = [pt.scalar(f"{dim}_length", dtype="int64") for dim in extra_dim_names]
        with new_or_existing_block_model_access(error_msg_on_access=BLOCK_MODEL_ACCESS_ERROR_MSG):
            rv = dist(*dummy_params, dict(zip(extra_dim_names, dummy_extra_lengths)))
        if not isinstance(rv, XTensorVariable):
            raise TypeError(DIST_NOT_XTENSOR_ERROR_MSG)
        if missing_dims := (set(extra_dim_names) - set(rv.type.dims)):
            raise ValueError(
                f"The `dist` function output is missing dims {sorted(missing_dims)}. "
                "Dims that are not implied by the params must be added through the "
                "`extra_dims` argument, as in "
                "`pmd.Normal.dist(mu, sigma, dim_lengths=extra_dims)`."
            )

        # Pipe a single RNG through the inner random operations, chaining each
        # one onto the state left by the previous, so that the Op has one RNG
        # input and one final state output, like a RandomVariable
        dummy_rng = xrandom_generator_type("rng")
        updates = collect_default_updates(
            inputs=[*dummy_params, *dummy_extra_lengths], outputs=(rv,)
        )
        if updates:
            fgraph = FunctionGraph(outputs=[rv, *updates.values()], clone=False)
            # Chain in topological order of the consuming nodes, as a random
            # operation may depend on the draws of another
            node_order = {node: i for i, node in enumerate(fgraph.toposort())}
            ordered_rngs = sorted(
                updates,
                key=lambda rng: min(node_order[client] for client, _ in fgraph.clients[rng]),
            )
            chained_rngs = [dummy_rng, *(updates[rng] for rng in ordered_rngs)]
            fgraph.replace_all(list(zip(ordered_rngs, chained_rngs[:-1])), import_missing=True)
            next_rng = chained_rngs[-1]
        else:
            next_rng = dummy_rng

        def rv_op(*params, extra_dims, rng=None):
            return cls.xrv_op(
                *params,
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                class_name=class_name,
                extra_dims=extra_dims,
                rng=rng,
            )

        rv_type = type(
            class_name,
            (DimCustomDistRV,),
            {
                "inline_logprob": logp is None,
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
                "rv_op": staticmethod(rv_op),
            },
        )

        # ---- Dispatch the user overrides with xtensor semantics ----
        if logp is not None:

            @_logprob.register(rv_type)
            def xcustom_dist_logp(op, values, *inputs, **kwargs):
                [value] = values
                return logp(value, *inputs[: op.n_params])

        if logcdf is not None:

            @_logcdf.register(rv_type)
            def xcustom_dist_logcdf(op, value, *inputs, **kwargs):
                return logcdf(value, *inputs[: op.n_params])

        @_support_point.register(rv_type)
        def xcustom_dist_support_point(op, rv_out, *inputs):
            params = inputs[: op.n_params]
            if support_point is not None:
                return support_point(rv_out, *params)
            # The dims counterpart of the tensor template `pt.full(size, param)`:
            # a zero broadcast to the params and extra dim lengths, so that the
            # initial point never evaluates the random graph
            zero = as_xtensor(pt.zeros((), dtype=rv_out.type.dtype))
            extra_lengths = inputs[op.n_params : op.n_params + len(op.extra_dims)]
            if op.extra_dims:
                zero = zero.expand_dims(dim=dict(zip(op.extra_dims, extra_lengths)))
            out_dims = rv_out.type.dims
            reduced_dims = {dim for p in params for dim in p.type.dims if dim not in out_dims}
            sp, *_ = broadcast(zero, *params, exclude=sorted(reduced_dims))
            if set(sp.type.dims) != set(out_dims):
                # Some output dims are only created inside the dist graph
                return x_zeros_like(rv_out)
            return sp.transpose(*out_dims)

        xop = rv_type(
            inputs=[*dummy_params, *dummy_extra_lengths, dummy_rng],
            outputs=[rv, next_rng],
            extra_dims=extra_dim_names,
        )
        if rng is None:
            rng = shared_rng(seed=None)
        out, out_next_rng = xop(*xtensor_params, *extra_dims.values(), rng)
        return out_next_rng, out
