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

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Node
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar.basic import GE, GT, LE, LT, Invert
from pytensor.tensor import TensorVariable
from pytensor.tensor.math import ge, gt, invert, le, lt

from pymc.logprob.abstract import (
    MeasurableElemwise,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import check_potential_measurability, filter_measurable_variables


class MeasurableComparison(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a binary comparison RV sub-graph."""

    valid_scalar_types = (GT, LT, GE, LE)


@node_rewriter(tracks=[gt, lt, ge, le])
def find_measurable_comparisons(fgraph: FunctionGraph, node: Node) -> list[TensorVariable] | None:
    measurable_inputs = filter_measurable_variables(node.inputs)

    if len(measurable_inputs) != 1:
        return None

    # Make the measurable base_var always be the first input to the MeasurableComparison node
    [measurable_var] = measurable_inputs
    measurable_var_idx = node.inputs.index(measurable_var)

    # deny broadcasting of the measurable input
    if measurable_var.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    # Check that the other input is not potentially measurable, in which case this rewrite
    # would be invalid
    const = node.inputs[(measurable_var_idx + 1) % 2]

    # check for potential measurability of const
    if check_potential_measurability([const]):
        return None

    node_scalar_op = node.op.scalar_op

    # Change the Op if the base_var is the second input in node.inputs. e.g. pt.lt(const, dist) -> pt.gt(dist, const)
    if measurable_var_idx == 1:
        if isinstance(node_scalar_op, LT):
            node_scalar_op = GT()
        elif isinstance(node_scalar_op, GT):
            node_scalar_op = LT()
        elif isinstance(node_scalar_op, GE):
            node_scalar_op = LE()
        elif isinstance(node_scalar_op, LE):
            node_scalar_op = GE()

    compared_op = MeasurableComparison(node_scalar_op)
    compared_rv = compared_op.make_node(measurable_var, const).default_output()
    return [compared_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_comparisons",
    find_measurable_comparisons,
    "basic",
    "comparison",
)


@_logprob.register(MeasurableComparison)
def comparison_logprob(op, values, base_rv, operand, **kwargs):
    (value,) = values

    base_rv_op = base_rv.owner.op

    logcdf = _logcdf_helper(base_rv, operand, **kwargs)
    logccdf = pt.log1mexp(logcdf)

    condn_exp = pt.eq(value, np.array(True))

    if isinstance(op.scalar_op, GT | GE):
        logprob = pt.switch(condn_exp, logccdf, logcdf)
    elif isinstance(op.scalar_op, LT | LE):
        logprob = pt.switch(condn_exp, logcdf, logccdf)
    else:
        raise TypeError(f"Unsupported scalar_op {op.scalar_op}")

    if base_rv.dtype.startswith("int"):
        logpmf = _logprob_helper(base_rv, operand, **kwargs)
        logcdf_prev = _logcdf_helper(base_rv, operand - 1, **kwargs)
        if isinstance(op.scalar_op, LT):
            return pt.switch(condn_exp, logcdf_prev, pt.logaddexp(logccdf, logpmf))
        elif isinstance(op.scalar_op, GE):
            return pt.switch(condn_exp, pt.logaddexp(logccdf, logpmf), logcdf_prev)

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    return logprob


class MeasurableBitwise(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a bitwise operation RV sub-graph."""

    valid_scalar_types = (Invert,)


@node_rewriter(tracks=[invert])
def find_measurable_bitwise(fgraph: FunctionGraph, node: Node) -> list[TensorVariable] | None:
    base_var = node.inputs[0]

    if not base_var.dtype.startswith("bool"):
        raise None

    if not filter_measurable_variables([base_var]):
        return None

    node_scalar_op = node.op.scalar_op
    bitwise_op = MeasurableBitwise(node_scalar_op)
    bitwise_rv = bitwise_op.make_node(base_var).default_output()
    return [bitwise_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_bitwise",
    find_measurable_bitwise,
    "basic",
    "bitwise",
)


@_logprob.register(MeasurableBitwise)
def bitwise_not_logprob(op, values, base_rv, **kwargs):
    (value,) = values

    logprob = _logprob_helper(base_rv, invert(value), **kwargs)

    return logprob
