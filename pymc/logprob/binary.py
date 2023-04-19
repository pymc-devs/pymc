#   Copyright 2023 The PyMC Developers
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
from typing import List, Optional

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Node
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar.basic import GE, GT, LE, LT
from pytensor.tensor.math import ge, gt, le, lt

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import check_potential_measurability, ignore_logprob


class MeasurableComparison(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a binary comparison RV sub-graph."""

    valid_scalar_types = (GT, LT, GE, LE)


@node_rewriter(tracks=[gt, lt, ge, le])
def find_measurable_comparisons(
    fgraph: FunctionGraph, node: Node
) -> Optional[List[MeasurableComparison]]:
    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableComparison):
        return None  # pragma: no cover

    (compared_var,) = node.outputs
    base_var, const = node.inputs

    if not (
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and base_var not in rv_map_feature.rv_values
    ):
        return None

    # check for potential measurability of const
    if not check_potential_measurability((const,), rv_map_feature):
        return None

    # Make base_var unmeasurable
    unmeasurable_base_var = ignore_logprob(base_var)

    compared_op = MeasurableComparison(node.op.scalar_op)
    compared_rv = compared_op.make_node(unmeasurable_base_var, const).default_output()
    compared_rv.name = compared_var.name
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

    if isinstance(op.scalar_op, (GT, GE)):
        logprob = pt.switch(condn_exp, logccdf, logcdf)
    elif isinstance(op.scalar_op, (LT, LE)):
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
