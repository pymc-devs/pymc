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
import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.math import _matrix_matrix_matmul

from pymc.logprob.abstract import MeasurableBlockwise, MeasurableOp, _logprob, _logprob_helper
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import check_potential_measurability, filter_measurable_variables


class MeasurableMatMul(MeasurableBlockwise):
    """Measurable matrix multiplication operation."""

    right_measurable: bool

    def __init__(self, measurable_right: bool, **kwargs):
        self.right_measurable = measurable_right
        super().__init__(**kwargs)


@_logprob.register(MeasurableMatMul)
def logprob_measurable_matmul(op, values, l, r):  # noqa: E741
    [y_value] = values
    if op.right_measurable:
        A, x = l, r
        x_value = pt.linalg.solve(A, y_value)
    else:
        x, A = l, r
        x_value = pt.linalg.solve(A.mT, y_value.mT).mT

    x_logp = _logprob_helper(x, x_value)

    # The operation has a support dimensionality of 2
    # We need to reduce it if it's still present in the base logp
    if x_logp.type.ndim == x_value.type.ndim:
        x_logp = pt.sum(x_logp, axis=(-1, -2))
    elif x_logp.type.ndim == x_value.type.ndim - 1:
        x_logp = pt.sum(x_logp, axis=-1)

    _, log_abs_jac_det = pt.linalg.slogdet(A)

    return x_logp - log_abs_jac_det


@node_rewriter(tracks=[_matrix_matrix_matmul])
def find_measurable_matmul(fgraph, node):
    """Find measurable matrix-matrix multiplication operations."""
    if isinstance(node.op, MeasurableOp):
        return None

    [out] = node.outputs
    [l, r] = node.inputs  # noqa: E741

    # Check that not both a and r are measurable
    measurable_inputs = filter_measurable_variables([l, r])
    if len(measurable_inputs) != 1:
        return None

    [measurable_input] = measurable_inputs

    # Check the measurable input is not broadcasted
    if measurable_input.type.broadcastable[:-2] != out.type.broadcastable[:-2]:
        return None

    measurable_right = measurable_input is r
    A = l if measurable_right else r

    # Check if the static shape already reveals a non-square matrix,
    if (
        A.type.shape[-1] is not None
        and A.type.shape[-2] is not None
        and A.type.shape[-1] != A.type.shape[-2]
    ):
        return None

    # Check the other input is not potentially measurable
    if check_potential_measurability([A]):
        return None

    measurable_matmul = MeasurableMatMul(measurable_right=measurable_right, **node.op._props_dict())
    return [measurable_matmul(l, r)]


measurable_ir_rewrites_db.register(
    find_measurable_matmul.__name__,
    find_measurable_matmul,
    "basic",
    "linalg",
)
