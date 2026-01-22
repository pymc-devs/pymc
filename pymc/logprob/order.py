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
from typing import cast

import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.extra_ops import broadcast_shape
from pytensor.tensor.math import Max, Sum
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.type_other import NoneConst, NoneTypeT
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableOp,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import filter_measurable_variables
from pymc.math import logdiffexp
from pymc.pytensorf import constant_fold


class MeasurableMax(MeasurableOp, Max):
    """A placeholder used to specify a log-likelihood for a max sub-graph."""


class MeasurableMaxDiscrete(MeasurableOp, Max):
    """A placeholder used to specify a log-likelihood for sub-graphs of maxima of discrete variables."""


@node_rewriter([Max])
def find_measurable_max(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    if isinstance(node.op, MeasurableMax | MeasurableMaxDiscrete):
        return None

    [base_var] = node.inputs

    if base_var.owner is None:
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    # We allow Max of RandomVariables or Elemwise of univariate RandomVariables
    if isinstance(base_var.owner.op, MeasurableElemwise):
        latent_base_vars = [
            var
            for var in base_var.owner.inputs
            if (var.owner and isinstance(var.owner.op, MeasurableOp))
        ]
        if len(latent_base_vars) != 1:
            return None
        [latent_base_var] = latent_base_vars
    else:
        latent_base_var = base_var

    latent_op = latent_base_var.owner.op
    if not (hasattr(latent_op, "dist_params") and getattr(latent_op, "ndim_supp") == 0):
        return None

    # univariate i.i.d. test which also rules out other distributions
    if not all(
        all(params.type.broadcastable) for params in latent_op.dist_params(latent_base_var.owner)
    ):
        return None

    base_var = cast(TensorVariable, base_var)

    if node.op.axis is None:
        axis = tuple(range(base_var.ndim))
    else:
        # Check whether axis covers all dimensions
        axis = tuple(sorted(node.op.axis))
        if axis != tuple(range(base_var.ndim)):
            return None

    # distinguish measurable discrete and continuous (because logprob is different)
    measurable_max_class = (
        MeasurableMaxDiscrete if latent_base_var.type.dtype.startswith("int") else MeasurableMax
    )
    max_rv = cast(TensorVariable, measurable_max_class(axis)(base_var))
    return [max_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_max",
    find_measurable_max,
    "basic",
    "max",
)


@node_rewriter([Sum])
def find_measurable_sum(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    [base_var] = node.inputs
    if base_var.owner is None:
        return None
    if not filter_measurable_variables(node.inputs):
        return None
    latent_op = base_var.owner.op
    if not isinstance(latent_op, NormalRV):
        return None
    if getattr(latent_op, "ndim_supp", None) != 0:
        return None
    base_var = cast(TensorVariable, base_var)
    if node.op.axis is None:
        axis = tuple(range(base_var.ndim))
    else:
        axis = tuple(sorted(node.op.axis))
        if axis != tuple(range(base_var.ndim)):
            return None

    mu, sigma = latent_op.dist_params(base_var.owner)
    mu_t = pt.as_tensor_variable(mu)
    sigma_t = pt.as_tensor_variable(sigma)
    size = base_var.owner.inputs[1]

    # If size is specified, it defines the broadcast target; otherwise derive from params
    if isinstance(size.type, NoneTypeT):
        target_shape = broadcast_shape(mu_t, sigma_t)
    else:
        target_shape = size

    mu_b = pt.broadcast_to(mu_t, target_shape)
    sigma_b = pt.broadcast_to(sigma_t, target_shape)

    mu_sum = pt.sum(mu_b)
    sigma_sum = pt.sqrt(pt.sum(pt.square(sigma_b)))

    # Create a scalar NormalRV for the sum
    rng = base_var.owner.inputs[0]
    sum_node = latent_op.make_node(rng, NoneConst, mu_sum, sigma_sum)
    sum_rv = cast(TensorVariable, sum_node.outputs[1])
    return [sum_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_sum",
    find_measurable_sum,
    "basic",
    "sum",
)


@_logprob.register(MeasurableMax)
def max_logprob(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation."""
    (value,) = values

    base_rv_shape = constant_fold(tuple(base_rv.shape), raise_not_constant=False)
    bcast_value = pt.broadcast_to(value, base_rv_shape)
    logprob = _logprob_helper(base_rv, bcast_value)[0]
    logcdf = _logcdf_helper(base_rv, bcast_value)[0]

    n = pt.prod(base_rv_shape)
    return (n - 1) * logcdf + logprob + pt.math.log(n)


@_logprob.register(MeasurableMaxDiscrete)
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
    logcdf = _logcdf_helper(base_rv, bcast_value)[0]
    logcdf_prev = _logcdf_helper(base_rv, bcast_value - 1)[0]

    n = pt.prod(base_rv_shape)
    return logdiffexp(n * logcdf, n * logcdf_prev)
