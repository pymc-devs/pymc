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

import numpy as np
import pytensor.tensor as pt

from pytensor.scalar.basic import GE, GT, LE, LT, Invert
from pytensor.tensor.math import invert

from pymc.logprob.abstract import (
    _logccdf_helper,
    _logcdf_helper,
    logprob_query,
)
from pymc.logprob.query import (
    bind_value,
    density_sources,
    other_query_uses,
    query_parts,
    rewrite_logprob_query,
)
from pymc.logprob.utils import filter_measurable_variables


@rewrite_logprob_query.register(GT)
@rewrite_logprob_query.register(GE)
@rewrite_logprob_query.register(LT)
@rewrite_logprob_query.register(LE)
def rewrite_comparison_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    candidates = filter_measurable_variables(rv.owner.inputs)
    if len(candidates) != 1:
        return None
    (base_rv,) = candidates
    if other_query_uses(fgraph, density_sources(base_rv), {query}):
        return None
    if base_rv.broadcastable != rv.broadcastable:
        return None
    index = rv.owner.inputs.index(base_rv)
    operand = rv.owner.inputs[1 - index]
    if index == 1:
        op = {LT: GT, GT: LT, LE: GE, GE: LE}[type(op)]()

    base_rv_op = base_rv.owner.op

    threshold = (
        pt.ceil(operand) - 1
        if base_rv.dtype.startswith("int") and isinstance(op, LT | GE)
        else operand
    )
    logcdf = _logcdf_helper(base_rv, threshold)
    logccdf = _logccdf_helper(base_rv, threshold)

    condn_exp = pt.eq(value, np.array(True))

    if isinstance(op, GT | GE):
        logprob = pt.switch(condn_exp, logccdf, logcdf)
    elif isinstance(op, LT | LE):
        logprob = pt.switch(condn_exp, logcdf, logccdf)
    else:
        raise TypeError(f"Unsupported scalar_op {op}")

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    return [logprob]


@rewrite_logprob_query.register(Invert)
def rewrite_bitwise_not_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    if rv.dtype != "bool":
        return None
    return [logprob_query(bind_value(fgraph, rv.owner.inputs[0], invert(value)))]
