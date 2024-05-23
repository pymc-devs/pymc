#   Copyright 2024 The PyMC Developers
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
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Max
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    MeasurableVariable,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import find_negated_var
from pymc.math import logdiffexp
from pymc.pytensorf import constant_fold


class MeasurableMax(Max):
    """A placeholder used to specify a log-likelihood for a max sub-graph."""


MeasurableVariable.register(MeasurableMax)


class MeasurableMaxDiscrete(Max):
    """A placeholder used to specify a log-likelihood for sub-graphs of maxima of discrete variables"""


MeasurableVariable.register(MeasurableMaxDiscrete)


@node_rewriter([Max])
def find_measurable_max(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableMax):
        return None  # pragma: no cover

    base_var = cast(TensorVariable, node.inputs[0])

    if base_var.owner is None:
        return None

    if not rv_map_feature.request_measurable(node.inputs):
        return None

    # Non-univariate distributions and non-RVs must be rejected
    if not (isinstance(base_var.owner.op, RandomVariable) and base_var.owner.op.ndim_supp == 0):
        return None

    # univariate i.i.d. test which also rules out other distributions
    for params in base_var.owner.inputs[3:]:
        if params.type.ndim != 0:
            return None

    # Check whether axis covers all dimensions
    axis = set(node.op.axis)
    base_var_dims = set(range(base_var.ndim))
    if axis != base_var_dims:
        return None

    # distinguish measurable discrete and continuous (because logprob is different)
    measurable_max: Max
    if base_var.owner.op.dtype.startswith("int"):
        measurable_max = MeasurableMaxDiscrete(list(axis))
    else:
        measurable_max = MeasurableMax(list(axis))

    max_rv_node = measurable_max.make_node(base_var)
    max_rv = max_rv_node.outputs

    return max_rv


measurable_ir_rewrites_db.register(
    "find_measurable_max",
    find_measurable_max,
    "basic",
    "max",
)


@_logprob.register(MeasurableMax)
def max_logprob(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation."""
    (value,) = values

    logprob = _logprob_helper(base_rv, value)
    logcdf = _logcdf_helper(base_rv, value)

    [n] = constant_fold([base_rv.size])
    logprob = (n - 1) * logcdf + logprob + pt.math.log(n)

    return logprob


@_logprob.register(MeasurableMaxDiscrete)
def max_logprob_discrete(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation.

    The formula that we use here is :
    .. math::
        \ln(P_{(n)}(x)) = \ln(F(x)^n - F(x-1)^n)
    where $P_{(n)}(x)$ represents the p.m.f of the maximum statistic and $F(x)$ represents the c.d.f of the i.i.d. variables.
    """
    (value,) = values
    logcdf = _logcdf_helper(base_rv, value)
    logcdf_prev = _logcdf_helper(base_rv, value - 1)

    [n] = constant_fold([base_rv.size])

    logprob = logdiffexp(n * logcdf, n * logcdf_prev)

    return logprob


class MeasurableMaxNeg(Max):
    """A placeholder used to specify a log-likelihood for a max(neg(x)) sub-graph.
    This shows up in the graph of min, which is (neg(max(neg(x)))."""


MeasurableVariable.register(MeasurableMaxNeg)


class MeasurableDiscreteMaxNeg(Max):
    """A placeholder used to specify a log-likelihood for sub-graphs of negative maxima of discrete variables"""


MeasurableVariable.register(MeasurableDiscreteMaxNeg)


@node_rewriter(tracks=[Max])
def find_measurable_max_neg(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableMaxNeg):
        return None  # pragma: no cover

    base_var = cast(TensorVariable, node.inputs[0])

    # Min is the Max of the negation of the same distribution. Hence, op must be Elemwise
    if not (base_var.owner is not None and isinstance(base_var.owner.op, Elemwise)):
        return None

    base_rv = find_negated_var(base_var)

    # negation is rv * (-1). Hence the scalar_op must be Mul
    if base_rv is None:
        return None

    # Non-univariate distributions and non-RVs must be rejected
    if not (isinstance(base_rv.owner.op, RandomVariable) and base_rv.owner.op.ndim_supp == 0):
        return None

    # univariate i.i.d. test which also rules out other distributions
    for params in base_rv.owner.inputs[3:]:
        if params.type.ndim != 0:
            return None

    # Check whether axis is supported or not
    axis = set(node.op.axis)
    base_var_dims = set(range(base_var.ndim))
    if axis != base_var_dims:
        return None

    if not rv_map_feature.request_measurable([base_rv]):
        return None

    # distinguish measurable discrete and continuous (because logprob is different)
    measurable_min: Max
    if base_rv.owner.op.dtype.startswith("int"):
        measurable_min = MeasurableDiscreteMaxNeg(list(axis))
    else:
        measurable_min = MeasurableMaxNeg(list(axis))

    return measurable_min.make_node(base_rv).outputs


measurable_ir_rewrites_db.register(
    "find_measurable_max_neg",
    find_measurable_max_neg,
    "basic",
    "min",
)


@_logprob.register(MeasurableMaxNeg)
def max_neg_logprob(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation.
    The formula that we use here is :
        \ln(f_{(n)}(x)) = \ln(n) + (n-1) \ln(1 - F(x)) + \ln(f(x))
    where f(x) represents the p.d.f and F(x) represents the c.d.f of the distribution respectively.
    """
    (value,) = values

    logprob = _logprob_helper(base_rv, -value)
    logcdf = _logcdf_helper(base_rv, -value)

    [n] = constant_fold([base_rv.size])
    logprob = (n - 1) * pt.math.log(1 - pt.math.exp(logcdf)) + logprob + pt.math.log(n)

    return logprob


@_logprob.register(MeasurableDiscreteMaxNeg)
def discrete_max_neg_logprob(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation.

    The formula that we use here is :
    .. math::
        \ln(P_{(n)}(x)) = \ln((1 - F(x - 1))^n - (1 - F(x))^n)
    where $P_{(n)}(x)$ represents the p.m.f of the maximum statistic and $F(x)$ represents the c.d.f of the i.i.d. variables.
    """

    (value,) = values

    # The cdf of a negative variable is the survival at the negated value
    logcdf = pt.log1mexp(_logcdf_helper(base_rv, -value))
    logcdf_prev = pt.log1mexp(_logcdf_helper(base_rv, -(value + 1)))

    [n] = constant_fold([base_rv.size])

    # Now we can use the same expression as the discrete max
    logprob = pt.where(
        pt.and_(pt.eq(logcdf, -pt.inf), pt.eq(logcdf_prev, -pt.inf)),
        -pt.inf,
        logdiffexp(n * logcdf_prev, n * logcdf),
    )

    return logprob
