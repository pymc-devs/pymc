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
from functools import partial
from typing import Any

import pytensor.tensor as pt

from pytensor.graph.fg import FunctionGraph
from pytensor.xtensor import as_xtensor, broadcast, zeros_like
from pytensor.xtensor.basic import xtensor_from_tensor
from pytensor.xtensor.random import shared_rng
from pytensor.xtensor.random.type import xrandom_generator_type
from pytensor.xtensor.type import XTensorVariable

from pymc.dims.distributions.core import DimDistribution, DimSymbolicRandomVariable
from pymc.distributions.custom import CustomDist as TensorCustomDist
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


def _non_random_dist(*args, dtype, class_name):
    """Use the regular CustomDist placeholder with named parameter and extra dims."""
    *params, extra_dims = args
    dim_lengths = dict(extra_dims)
    for param in params:
        for dim, length in param.sizes.items():
            dim_lengths.setdefault(dim, length)
    placeholder = TensorCustomDist.dist(
        size=tuple(dim_lengths.values()),
        dtype=dtype,
        class_name=class_name,
        random=default_not_implemented(class_name, "dist"),
    )
    return xtensor_from_tensor(placeholder, dims=tuple(dim_lengths))


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

        import numpy as np
        import pymc as pm
        import pymc.dims as pmd
        import pytensor.xtensor.math as ptxm


        def logitnormal_dist(mu, sigma, extra_dims):
            return ptxm.sigmoid(pmd.Normal.dist(mu=mu, sigma=sigma, dim_lengths=extra_dims))


        data = pmd.as_xtensor([0.2, 0.4, 0.5, 0.7, 0.8], dims=("city",))
        with pm.Model(coords={"city": range(5)}) as model:
            mu = pmd.Normal("mu", mu=0, sigma=1)
            sigma = pmd.HalfNormal("sigma", sigma=1)
            pmd.CustomDist("x", mu, sigma, dist=logitnormal_dist, observed=data, dims="city")

    When only ``dist`` is provided, the logp is derived automatically from the
    graph, exactly as it is for the built-in ``pymc.dims`` distributions.

    User-provided ``logp``/``logcdf``/``support_point`` override the defaults.
    They are dispatched with xtensor semantics: the value, the random
    variable and all the params are ``XTensorVariable`` with named dims, and
    the functions may return an ``XTensorVariable`` (or a plain tensor).

    A distribution can also be defined without ``dist``, through ``logp``
    (and/or ``logcdf``, ``support_point``). The random graph is then a
    placeholder with the extra dims and the batch dims of the parameters,
    which raises if forward sampling is attempted::

        def normal_logp(value, mu, sigma):
            return -0.5 * ((value - mu) / sigma) ** 2 - ptxm.log(sigma * np.sqrt(2 * np.pi))


        with pm.Model(coords={"city": range(5)}) as model:
            mu = pmd.Normal("mu", mu=0, sigma=1)
            sigma = pmd.HalfNormal("sigma", sigma=1)
            pmd.CustomDist("y", mu, sigma, logp=normal_logp, observed=data, dims="city")

    For a latent continuous distribution with constrained support, provide an appropriate
    ``default_transform`` from :mod:`pymc.dims.distributions.transforms`.
    CustomDist does not infer a transform from ``dist``. Using the same
    ``logitnormal_dist`` function, whose support is (0, 1), specify a log-odds
    transform. A prior draw provides an initial value inside the support,
    since the default zero support point is outside it::

        from pymc.dims.distributions.transforms import LogOddsTransform

        with pm.Model(coords={"city": range(5)}) as model:
            mu = pmd.Normal("mu", mu=0, sigma=1)
            sigma = pmd.HalfNormal("sigma", sigma=1)
            pmd.CustomDist(
                "x",
                mu,
                sigma,
                dist=logitnormal_dist,
                dims="city",
                default_transform=LogOddsTransform(),
                initval="prior",
            )
    """

    @classmethod
    def dist(cls, *dist_params, **kwargs):
        return super().dist(dist_params, **kwargs)

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
            dist = partial(_non_random_dist, dtype=dtype, class_name=class_name)

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
            (DimSymbolicRandomVariable,),
            {
                "inline_logprob": logp is None,
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
                "rv_op": staticmethod(rv_op),
            },
        )

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
            # Initial points use FAST_COMPILE, where zeros_like(rv) may evaluate rv.
            zero = as_xtensor(pt.zeros((), dtype=rv_out.dtype))
            extra_lengths = dict(zip(op.extra_dims, inputs[op.n_params : -1]))
            zero = zero.expand_dims(dim=extra_lengths)
            reduced_dims = {dim for p in params for dim in p.dims if dim not in rv_out.dims}
            sp, *_ = broadcast(zero, *params, exclude=sorted(reduced_dims))
            if set(sp.dims) != set(rv_out.dims):
                return zeros_like(rv_out)
            return sp.transpose(*rv_out.dims)

        xop = rv_type(
            inputs=[*dummy_params, *dummy_extra_lengths, dummy_rng],
            outputs=[rv, next_rng],
            extra_dims=extra_dim_names,
        )
        if rng is None:
            rng = shared_rng(seed=None)
        out, out_next_rng = xop(*xtensor_params, *extra_dims.values(), rng)
        return out_next_rng, out
