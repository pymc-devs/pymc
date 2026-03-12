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
from pytensor.tensor.type_other import NoneTypeT

from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.basic import logp


class MinibatchRandomVariable(MeasurableOp, Op):
    """RV whose logprob should be rescaled to match total_size."""

    __props__ = ()
    view_map = {0: [0]}

    def make_node(self, rv, *total_size):
        rv = as_tensor_variable(rv)
        total_size = [
            t
            if isinstance(t, Variable)
            else (NoneConst if t is None else as_tensor_variable(t, dtype="int64", ndim=0))
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
    total_size: int | TensorVariable | Sequence[int | TensorVariable | EllipsisType | None],
) -> TensorVariable:
    """Create variable whose logp is rescaled by total_size."""
    rv_ndim_supp = rv.owner.op.ndim_supp

    if isinstance(total_size, int):
        total_size = (total_size, *([None] * rv_ndim_supp))
    elif isinstance(total_size, TensorVariable):
        if total_size.type.ndim == 0:
            total_size = (total_size, *([None] * rv_ndim_supp))
        elif total_size.type.ndim == 1:
            total_size = tuple(total_size)
        else:
            raise ValueError(
                f"Total size must be a 0d or 1d vector got {total_size} with {total_size.type.ndim} dimensions"
            )

    if not isinstance(total_size, list | tuple):
        raise ValueError(f"Invalid type for total_size {total_size}: {type(total_size)}")

    if Ellipsis in total_size:
        # Replace Ellipsis by None
        if total_size.count(Ellipsis) > 1:
            raise ValueError("Only one Ellipsis can be present in total_size")
        sep = total_size.index(Ellipsis)
        begin = total_size[:sep]
        end = total_size[sep + 1 :]
        missing_ndims = max((rv_ndim_supp - len(begin) - len(end), 0))
        total_size = (*begin, *([None] * missing_ndims), *end)

    if (len(total_size) - rv_ndim_supp) not in (0, 1):
        raise ValueError(
            f"Length of total_size {total_size} not compatble with ndim_supp of RV {rv}, "
            f"got {len(total_size)} but must be {rv_ndim_supp} or {rv_ndim_supp - 1}"
        )

    out = minibatch_rv(rv, *total_size)
    assert isinstance(out.owner.op, MinibatchRandomVariable)
    return cast(TensorVariable, out)


def get_scaling(
    total_size: Sequence[TensorVariable], shape: TensorVariable | Sequence[TensorVariable]
) -> TensorVariable:
    """Get scaling constant for logp."""
    # mypy doesn't understand we can convert a shape TensorVariable into a tuple
    shape = tuple(shape)

    if len(total_size) == (len(shape) - 1):
        # This happens when RV has no batch dimensions
        # In that case the total_size corresponds to a dummy shape of 1
        total_size = (1, *total_size)

    assert len(shape) == len(total_size)

    coefs = [
        size / dim_length
        for size, dim_length in zip(total_size, shape)
        if not isinstance(size.type, NoneTypeT)
    ]
    coef = pt.prod(coefs) if len(coefs) > 1 else coefs[0]

    return pt.cast(coef, dtype=config.floatX)


@_logprob.register(MinibatchRandomVariable)
def minibatch_rv_logprob(op, values, *inputs, **kwargs):
    [value] = values
    rv, *total_size = inputs
    raw_logp = logp(rv, value, **kwargs)
    scaled_logp = raw_logp * get_scaling(total_size, raw_logp.shape)
    return scaled_logp
