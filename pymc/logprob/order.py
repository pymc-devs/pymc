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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import pytensor.tensor as pt

from numpy.lib.array_utils import normalize_axis_tuple
from pytensor.graph import ancestors
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar import Add, Mul
from pytensor.tensor import get_underlying_scalar_constant_value
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import Argmax, Max, variadic_add, variadic_mul
from pytensor.tensor.random.basic import ExponentialRV, GumbelRV
from pytensor.tensor.rewriting.basic import broadcasted_by
from pytensor.tensor.type_other import NoneTypeT

from pymc.distributions.continuous import WeibullBetaRV
from pymc.logprob.abstract import (
    _logcdf_helper,
    logprob_query,
)
from pymc.logprob.query import (
    bind_value,
    density_sources,
    infer_support_axes,
    other_query_uses,
    query_parts,
    rewrite_logprob_query,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import filter_measurable_variables
from pymc.math import logdiffexp
from pymc.pytensorf import constant_fold


def max_logprob(fgraph, value, base_rv):
    r"""Compute the log-likelihood graph for the `Max` operation."""
    base_rv_shape = constant_fold(tuple(base_rv.shape), raise_not_constant=False)
    bcast_value = pt.broadcast_to(value, base_rv_shape)
    logprob = logprob_query(bind_value(fgraph, base_rv, bcast_value))[(0,) * base_rv.ndim]
    logcdf = _logcdf_helper(base_rv, bcast_value)[(0,) * base_rv.ndim]

    n = pt.prod(base_rv_shape)
    return (n - 1) * logcdf + logprob + pt.math.log(n)


def max_logprob_discrete(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation.

    The formula that we use here is :
    .. math::
        \ln(P_{(n)}(x)) = \ln(F(x)^n - F(x-1)^n)
    where $P_{(n)}(x)$ represents the p.m.f of the maximum statistic and $F(x)$ represents the c.d.f of the i.i.d. variables.
    """
    (value,) = values

    base_rv_shape = constant_fold(tuple(base_rv.shape), raise_not_constant=False)
    bcast_value = pt.broadcast_to(value, base_rv_shape)
    logcdf = _logcdf_helper(base_rv, bcast_value)[(0,) * base_rv.ndim]
    logcdf_prev = _logcdf_helper(base_rv, bcast_value - 1)[(0,) * base_rv.ndim]

    n = pt.prod(base_rv_shape)
    return logdiffexp(n * logcdf, n * logcdf_prev)


@node_rewriter([ExponentialRV, GumbelRV, WeibullBetaRV])
def lift_loc_scale(fgraph, node):
    """Rewrite rv(loc, scale) * s + l as rv(loc * s + l, scale * s), when they admit such parametrization.

    Currently, this rewrite targets just RVs accepted by categorical_from_argmax, but can be generalized beyond this use case.
    """
    rv = node.out
    clients = fgraph.clients[rv]
    if len(clients) != 1:
        return None
    [(add_mul_node, idx)] = clients

    if not (
        isinstance(add_mul_node.op, Elemwise) and isinstance(add_mul_node.op.scalar_op, Add | Mul)
    ):
        return None

    match node.op:
        case ExponentialRV():
            loc = 0
            rng, size, scale = node.inputs
        case WeibullBetaRV():
            loc = 0
            rng, size, shape, scale = node.inputs
        case GumbelRV():
            rng, size, loc, scale = node.inputs
        case _:
            raise NotImplementedError(f"Unexpected op {node.op}")

    negative_scale = False
    if isinstance(add_mul_node.op.scalar_op, Add):
        if not isinstance(node.op, GumbelRV):
            # Only Gumbel allows lifting loc
            return None
        loc += variadic_add(*(t for i, t in enumerate(add_mul_node.inputs) if i != idx))
    else:
        extra_scale = variadic_mul(*(t for i, t in enumerate(add_mul_node.inputs) if i != idx))
        # The rewrite is only valid if the scale non-negative.
        try:
            [extra_scale_const] = constant_fold([extra_scale])
        except NotScalarConstantError:
            # Not-constant scale. Get out
            # TODO: Allow more cases when we have standard machinery to infer sign of symbolic operations
            return None
        unique_sign = np.unique(np.sign(extra_scale_const))
        if unique_sign.size == 2:
            # There is mixed sign in the scale (or zero). Get out
            return None
        negative_scale = unique_sign == -1
        if negative_scale:
            if (extra_scale_const == -1).all():
                # There is no scale to lift, it's just a negated rv (happens in argmin(rv))
                return None
            # Scale is homogenously negative, make it positive, and return rv * -1 later so other rewrites can handle it
            extra_scale *= -1
        loc *= extra_scale
        scale *= extra_scale

    # We can't lift the argument if either
    # 1. the argument it's broadcasting the RV
    # 2. the RV shows up in the lifted args (as in rv + rv or rv * rv)
    # We check with loc as that is altered in both branches
    if broadcasted_by(rv, loc) or rv in ancestors([loc]):
        return None

    match node.op:
        case ExponentialRV():
            lifted_rv = node.op.make_node(rng, size, scale).out
        case WeibullBetaRV():
            # WeibullBetaRV is a SymbolicRandomVariable, we can't simply pass arguments of a different type
            lifted_rv = WeibullBetaRV.rv_op(shape, scale, rng=rng, size=size)
        case GumbelRV():
            lifted_rv = node.op.make_node(rng, size, loc, scale).out

    if negative_scale:
        lifted_rv *= -1

    return {add_mul_node.out: lifted_rv}


@node_rewriter([Argmax])
def categorical_from_argmax(fgraph, node):
    """Convert closed from argmax/argmin to equivalent categorical."""

    def is_minus_1(x):
        try:
            return get_underlying_scalar_constant_value(x, max_recur=3) == -1
        except NotScalarConstantError:
            return False

    [base_var] = node.inputs
    base_node = base_var.owner

    if base_node is None:
        return None
    if not filter_measurable_variables([base_var]):
        return None

    argmax_axes = node.op.axis
    if argmax_axes is None:
        argmax_axes = tuple(range(base_var.ndim))
    else:
        argmax_axes = normalize_axis_tuple(argmax_axes, base_var.ndim)

    probs = None

    if isinstance(base_node.op, GumbelRV):
        # argmax(gumbel(loc, scale)) -> categorical(exp(loc / scale))
        rng, size, loc, scale = base_node.inputs

        # gumbel scale has to be constant across the argmax axes
        if not all(b for i, b in enumerate(scale.type.broadcastable) if i in argmax_axes):
            return None

        # Dividing by scale also broadcasts probs to the size implied by (loc, scale)
        probs = pt.exp(loc / scale)

    # Check if we have an Argmin
    # Argmin is internally represented as Argmax(-x), and -x is canonicalized as x * -1
    elif (
        len(base_node.inputs) == 2
        and isinstance(base_node.op, Elemwise)
        and isinstance(base_node.op.scalar_op, Mul)
        and ((right_neg := is_minus_1(base_node.inputs[1])) or is_minus_1(base_node.inputs[0]))
    ):
        base_var = base_node.inputs[0 if right_neg else 1]
        base_node = base_var.owner
        if base_node is None:
            return None

        if isinstance(base_node.op, ExponentialRV):
            # argmin(exponential(rate)) -> categorical(rate)
            rng, size, scale = base_var.owner.inputs
            probs = 1 / scale

        elif isinstance(base_node.op, WeibullBetaRV):
            # argmin(weibull(shape, scale)) -> categorical(scale ** -shape / Σ(scale ** -shape))
            rng, size, shape, scale = base_node.inputs

            # weibull shape has to be constant across the argmin axes
            if not all(b for i, b in enumerate(shape.type.broadcastable) if i in argmax_axes):
                return None

            probs = scale**-shape

    if probs is None:
        return None

    if not isinstance(size.type, NoneTypeT):
        # Make probs explicit to facilitate logic below
        probs = pt.broadcast_to(probs, size)

    # Join axes probs at the last axis (core axis of Categorical)
    n_axes = len(argmax_axes)
    probs = pt.moveaxis(probs, argmax_axes, tuple(range(-n_axes, 0)))
    probs = pt.join_dims(probs, -n_axes, n_axes)

    # Normalize probs and create categorical
    probs /= probs.sum(-1, keepdims=True)
    return [pt.random.categorical(probs, rng=rng, size=None)]


measurable_ir_rewrites_db.register("lift_loc_scale", lift_loc_scale, "basic", "lift_rv_args")

measurable_ir_rewrites_db.register(
    "categorical_from_argmax",
    categorical_from_argmax,
    "basic",
    "order",
    "argmax",
)


@infer_support_axes.register(Max)
def measure_max(op, var):
    return ()


@rewrite_logprob_query.register(Max)
def rewrite_max_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    [base] = rv.owner.inputs
    if not filter_measurable_variables([base]):
        return None
    if other_query_uses(fgraph, density_sources(base), {query}):
        return None
    latent = base
    while isinstance(latent.owner_op, Elemwise):
        candidates = filter_measurable_variables(latent.owner.inputs)
        if len(candidates) != 1:
            return None
        [latent] = candidates
    latent_op = latent.owner_op
    if not (hasattr(latent_op, "dist_params") and getattr(latent_op, "ndim_supp", None) == 0):
        return None
    if not all(all(p.type.broadcastable) for p in latent_op.dist_params(latent.owner)):
        return None
    if op.axis is not None and tuple(sorted(op.axis)) != tuple(range(base.ndim)):
        return None
    if latent.dtype.startswith("int"):
        return [max_logprob_discrete(op, [value], base, **kwargs)]
    return [max_logprob(fgraph, value, base)]
