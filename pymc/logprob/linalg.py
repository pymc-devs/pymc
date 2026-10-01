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
import pytensor.tensor as pt

from pytensor.tensor.math import _matmul

from pymc.logprob.abstract import logprob_query
from pymc.logprob.query import (
    bind_value,
    infer_support_axes,
    query_parts,
    rewrite_logprob_query,
)
from pymc.logprob.utils import filter_measurable_variables


@rewrite_logprob_query.register(type(_matmul))
def rewrite_matmul_logprob(op, fgraph, query, **kwargs):
    rv, y_value = query_parts(query)
    if op.core_op != _matmul.core_op:
        return None
    l, r = rv.owner.inputs  # noqa: E741
    candidates = filter_measurable_variables([l, r])
    if len(candidates) != 1:
        return None
    [base] = candidates
    if base.type.broadcastable[:-2] != rv.type.broadcastable[:-2]:
        return None
    A = l if base is r else r
    if (
        A.type.shape[-1] is not None
        and A.type.shape[-2] is not None
        and A.type.shape[-1] != A.type.shape[-2]
    ):
        return None
    if base is r:
        A, x = l, r
        x_value = pt.linalg.solve(A, y_value)
    else:
        x, A = l, r
        x_value = pt.linalg.solve(A.mT, y_value.mT).mT

    x_logp = logprob_query(bind_value(fgraph, x, x_value))

    # The operation has a support dimensionality of 2
    # We need to reduce it if it's still present in the base logp
    if x_logp.type.ndim == x_value.type.ndim:
        x_logp = pt.sum(x_logp, axis=(-1, -2))
    elif x_logp.type.ndim == x_value.type.ndim - 1:
        x_logp = pt.sum(x_logp, axis=-1)

    _, log_abs_jac_det = pt.linalg.slogdet(A)

    return [x_logp - log_abs_jac_det]


@infer_support_axes.register(type(_matmul))
def measure_matmul(op, var):
    if op.core_op != _matmul.core_op:
        return infer_support_axes.dispatch(object)(op, var)
    return tuple(range(var.ndim - 2, var.ndim))
