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

import abc
import warnings

from collections.abc import Sequence
from functools import singledispatch

from pytensor.configdefaults import config
from pytensor.graph import Apply, Op, Variable
from pytensor.tensor import TensorVariable, log1mexp, tensor
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.xtensor.type import XTensorType


@singledispatch
def _logprob(
    op: Op,
    values: Sequence[TensorVariable],
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the log-density/mass of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new density/mass graphs
    for a ``RandomVariable``, register a new function on this dispatcher.

    Pass all values involved in the node's density together.
    """
    raise NotImplementedError(f"Logprob method not implemented for {op}")


def _logprob_helper(rv, *values, **kwargs):
    """Call the density dispatcher for an RV with an implemented formula."""
    logprob = _logprob(rv.owner.op, values, *rv.owner.inputs, **kwargs)
    if name := rv.name:
        if isinstance(logprob, list | tuple):
            for i, term in enumerate(logprob):
                term.name = f"{name}_logprob.{i}"
        else:
            logprob.name = f"{name}_logprob"
    return logprob


@singledispatch
def _logcdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
):
    """Create a graph for the logcdf of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new logcdf graphs
    for a ``RandomVariable``, register a new function on this dispatcher.
    """
    raise NotImplementedError(f"LogCDF method not implemented for {op}")


def _logcdf_helper(rv, value):
    """Help call `_logcdf` dispatcher."""
    logcdf = _logcdf(rv.owner.op, value, *rv.owner.inputs)

    if rv.name:
        logcdf.name = f"{rv.name}_logcdf"

    return logcdf


@singledispatch
def _logccdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
):
    """Create a graph for the log complementary CDF (log survival function) of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new logccdf graphs
    for a ``RandomVariable``, register a new function on this dispatcher.

    The log complementary CDF is defined as log(1 - CDF(x)), also known as the
    log survival function. For distributions with a numerically stable implementation,
    this should be used instead of computing log(1 - exp(logcdf)).
    """
    raise NotImplementedError(f"LogCCDF method not implemented for {op}")


def _logccdf_helper(rv, value):
    """Helper that calls `_logccdf` dispatcher with fallback to log1mexp(logcdf).

    If a numerically stable `_logccdf` implementation is registered for the
    distribution, it will be used. Otherwise, falls back to computing
    `log(1 - exp(logcdf))` which may be numerically unstable in the tails.
    """
    try:
        logccdf = _logccdf(rv.owner.op, value, *rv.owner.inputs)
    except NotImplementedError:
        logcdf = _logcdf_helper(rv, value)
        logccdf = log1mexp(logcdf)

    if rv.name:
        logccdf.name = f"{rv.name}_logccdf"

    return logccdf


@singledispatch
def _icdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
):
    """Create a graph for the inverse CDF of a `RandomVariable`.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.
    """
    raise NotImplementedError(f"Inverse CDF method not implemented for {op}")


def _icdf_helper(rv, value):
    """Help call `_icdf` dispatcher."""
    rv_icdf = _icdf(rv.owner.op, value, *rv.owner.inputs)

    if rv.name:
        rv_icdf.name = f"{rv.name}_icdf"

    return rv_icdf


class MeasurableOp(abc.ABC):
    """An operation whose outputs can be assigned a measure/log-probability.

    ``supp_axes`` lists support axes as negative indices, in ``node.outputs`` order,
    including ``None`` entries for RNG outputs. Set it when constructing the measurable Op,
    deriving it from inputs so equal nodes agree. Unlike ``values`` passed to ``_logprob``,
    it covers every output.

    ``None`` means unknown support axes; ``()`` means scalar support. `LogprobQuery`
    requires known axes to determine its output type.
    """

    supp_axes: tuple[tuple[int, ...] | None, ...] | None = None


MeasurableOp.register(RandomVariable)


class MeasurableTensorOp(MeasurableOp):
    """A recognized tensor operation, carrying density metadata in the IR."""

    native_type: type[Op]
    logp_ndims: tuple[int | None, ...] | None = None
    # Per-output recognition: True is ready, False is deterministic, None is unresolved.
    measurable_outputs: tuple[bool | None, ...] | None = None


def is_measurable(var):
    op = var.owner_op
    if not isinstance(op, MeasurableOp):
        return False
    if isinstance(op, MeasurableTensorOp) and op.logp_ndims is None:
        return False
    outputs = getattr(op, "measurable_outputs", None)
    return outputs is None or outputs[var.index] is True


class ValuedRV(Op):
    r"""Associate a random expression with its value in the density IR.

    The output marks a conditioning boundary for other expressions. A
    `LogprobQuery` retains this pair to derive its density; parameter uses are
    eventually replaced by the value. Inverting a query creates upstream
    bindings, so dependent densities also see values inferred through transforms.

    Keeping the pair in the graph protects conditioning points from algebraic
    rewrites and lets FunctionGraph maintain their dependencies.
    """

    view_map = {0: [0]}

    def make_node(self, rv, value):
        assert isinstance(rv, Variable)
        assert isinstance(value, Variable)
        return Apply(self, [rv, value], [rv.type(name=rv.name)])

    def perform(self, node, inputs, out):
        warnings.warn("ValuedVar should not be present in the final graph!")
        out[0][0] = inputs[0]

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]


valued_rv = ValuedRV()


def supp_axes(var: Variable) -> tuple[int, ...] | None:
    """Return support axes from the Op's `supp_axes` or `ndim_supp`, or None if unknown."""
    node = var.owner
    # getattr: RandomVariable is a virtual subclass of MeasurableOp, so it has no default
    if (declared := getattr(node.op, "supp_axes", None)) is not None:
        return declared[node.outputs.index(var)]
    # SymbolicRandomVariable may lack ndim_supp when no signature is defined.
    ndim_supp = getattr(node.op, "ndim_supp", None)
    return None if ndim_supp is None else tuple(range(-ndim_supp, 0))


class LogprobQuery(Op):
    """A density tensor with known rank and unknown axis lengths."""

    def make_node(self, binding):
        if not isinstance(binding.owner_op, ValuedRV):
            raise TypeError("LogprobQuery requires an explicit RV-value binding")
        from pymc.logprob.query import density_ndim, support_axes

        ndim = density_ndim(binding)
        if isinstance(binding.type, XTensorType):
            axes = support_axes(binding)
            dims = tuple(dim for axis, dim in enumerate(binding.type.dims) if axis not in axes)
            output = XTensorType(config.floatX, dims=dims)()
        else:
            output = tensor(dtype=config.floatX, shape=(None,) * ndim)
        return Apply(self, [binding], [output])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError("Unresolved LogprobQuery")


logprob_query = LogprobQuery()


def n_potential_valued_outputs(node) -> int:
    """Count non-RNG outputs that may participate in the node's density."""
    return sum(not isinstance(out.type, RandomType) for out in node.outputs)


@_logcdf.register(Elemwise)
def elemwise_logcdf(op, value, *inputs):
    return _logcdf(op.scalar_op, value, *inputs)


@_logccdf.register(Elemwise)
def elemwise_logccdf(op, value, *inputs):
    return _logccdf(op.scalar_op, value, *inputs)


@_icdf.register(Elemwise)
def elemwise_icdf(op, value, *inputs):
    return _icdf(op.scalar_op, value, *inputs)
