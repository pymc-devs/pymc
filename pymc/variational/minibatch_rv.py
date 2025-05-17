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
from collections.abc import Sequence
from typing import Any, cast

import pytensor.tensor as pt

from pytensor import Variable, config
from pytensor.graph import Apply, Op
from pytensor.tensor import NoneConst, TensorVariable, as_tensor_variable

from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.basic import logp


class MinibatchRandomVariable(MeasurableOp, Op):
    """RV whose logprob should be rescaled to match total_size."""

    __props__ = ()
    view_map = {0: [0]}

    def make_node(self, rv, *total_size):
        rv = as_tensor_variable(rv)
        total_size = [
            as_tensor_variable(t, dtype="int64", ndim=0) if t is not None else NoneConst
            for t in total_size
        ]
        assert len(total_size) == rv.ndim
        out = rv.type()
        return Apply(self, [rv, *total_size], [out])

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0]


minibatch_rv = MinibatchRandomVariable()


EllipsisType = Any  # EllipsisType is not present in Python 3.8 yet


def create_minibatch_rv(
    rv: TensorVariable,
    total_size: int | None | Sequence[int | EllipsisType | None],
) -> TensorVariable:
    """Create variable whose logp is rescaled by total_size."""
    if isinstance(total_size, int):
        if rv.ndim <= 1:
            total_size = [total_size]
        else:
            missing_ndims = rv.ndim - 1
            total_size = [total_size] + [None] * missing_ndims
    elif isinstance(total_size, list | tuple):
        total_size = list(total_size)
        if Ellipsis in total_size:
            # Replace Ellipsis by None
            if total_size.count(Ellipsis) > 1:
                raise ValueError("Only one Ellipsis can be present in total_size")
            sep = total_size.index(Ellipsis)
            begin = total_size[:sep]
            end = total_size[sep + 1 :]
            missing_ndims = max((rv.ndim - len(begin) - len(end), 0))
            total_size = begin + [None] * missing_ndims + end
        if len(total_size) > rv.ndim:
            raise ValueError(f"Length of total_size {total_size} is langer than RV ndim {rv.ndim}")
    else:
        raise TypeError(f"Invalid type for total_size: {total_size}")

    return cast(TensorVariable, minibatch_rv(rv, *total_size))


def get_scaling(total_size: Sequence[Variable], shape: TensorVariable) -> TensorVariable:
    """Get scaling constant for logp."""
    # mypy doesn't understand we can convert a shape TensorVariable into a tuple
    shape = tuple(shape)  # type: ignore[assignment]

    # Scalar RV
    if len(shape) == 0:  # type: ignore[arg-type]
        coef = total_size[0] if not NoneConst.equals(total_size[0]) else 1.0
    else:
        coefs = [t / shape[i] for i, t in enumerate(total_size) if not NoneConst.equals(t)]
        coef = pt.prod(coefs)

    return pt.cast(coef, dtype=config.floatX)


@_logprob.register(MinibatchRandomVariable)
def minibatch_rv_logprob(op, values, *inputs, **kwargs):
    [value] = values
    rv, *total_size = inputs
    return logp(rv, value, **kwargs) * get_scaling(total_size, value.shape)
