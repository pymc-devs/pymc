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
from pytensor.tensor.math import Max
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.sort import SortOp
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    MeasurableOp,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import (
    CheckParameterValue,
    check_potential_measurability,
    filter_measurable_variables,
)
from pymc.math import logdiffexp
from pymc.pytensorf import constant_fold


def _underlying_iid_rv(variable) -> TensorVariable | None:
    # Check whether an IID base RV is connected to the variable through identical elemwise operations
    from pymc.distributions.distribution import SymbolicRandomVariable
    from pymc.logprob.transforms import MeasurableTransform

    def iid_elemwise_root(var: TensorVariable) -> TensorVariable | None:
        node = var.owner
        if isinstance(node.op, RandomVariable | SymbolicRandomVariable):
            return var
        elif isinstance(node.op, MeasurableTransform):
            if len(node.inputs == 1):
                return iid_elemwise_root(node.inputs[0])
            else:
                # If the non-measurable inputs are broadcasted, it is still an IID operation.
                measurable_inp = node.op.measurable_input_idx
                other_inputs = [inp for i, inp in node.inputs if i != measurable_inp]
                if all(all(other_inp.type.broadcastable) for other_inp in other_inputs):
                    return iid_elemwise_root(node.inputs[measurable_inp])
        return None

    # Check that the root is a univariate distribution linked by only elemwise operations
    latent_base_var = iid_elemwise_root(variable)

    if latent_base_var is None:
        return None

    latent_op = latent_base_var.owner.op

    if not (hasattr(latent_op, "dist_params") and getattr(latent_op, "ndim_supp") == 0):
        return None

    if not all(
        all(params.type.broadcastable) for params in latent_op.dist_params(latent_base_var.owner)
    ):
        return None

    return cast(TensorVariable, latent_base_var)


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

    # We allow Max of RandomVariables or IID Elemwise of univariate RandomVariables
    latent_base_var = _underlying_iid_rv(base_var)

    if not latent_base_var:
        return None

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
    find_measurable_max.__name__,
    find_measurable_max,
    "basic",
    "max",
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


class MeasurableSort(MeasurableOp, SortOp):
    """A placeholder used to specify a log-likelihood for a sort sub-graph."""


@_logprob.register(MeasurableSort)
def sort_logprob(op, values, base_rv, axis, **kwargs):
    r"""Compute the log-likelihood graph for the `Sort` operation."""
    (value,) = values

    logprob = _logprob_helper(base_rv, value).sum(axis=-1)

    base_rv_shape = constant_fold(tuple(base_rv.shape), raise_not_constant=False)
    n = pt.prod(base_rv_shape, axis=-1)
    sorted_logp = pt.gammaln(n + 1) + logprob

    # The sorted value is not really a parameter, but we include the check in
    # `CheckParameterValue` to avoid costly sorting if `check_bounds=False` in a PyMC model
    return CheckParameterValue("value must be sorted", can_be_replaced_by_ninf=True)(
        sorted_logp, pt.eq(value, value.sort(axis=axis, kind=op.kind)).all()
    )


@node_rewriter(tracks=[SortOp])
def find_measurable_sort(fgraph, node):
    if isinstance(node.op, MeasurableSort):
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    [base_var, axis] = node.inputs

    # We allow Max of RandomVariables or IID Elemwise of univariate RandomVariables
    if _underlying_iid_rv(base_var) is None:
        return None

    # Check axis is not potentially measurable
    if check_potential_measurability([axis]):
        return None

    return [MeasurableSort(**node.op._props_dict())(base_var, axis)]


measurable_ir_rewrites_db.register(
    find_measurable_sort.__name__,
    find_measurable_sort,
    "basic",
    "sort",
)
