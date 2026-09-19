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
from itertools import zip_longest

from pytensor.configdefaults import config
from pytensor.graph import Apply, Op, Variable
from pytensor.graph.utils import MetaType
from pytensor.tensor import TensorVariable, log1mexp, tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.utils import broadcast_static_dim_lengths


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

    Calling the dispatcher directly asserts that ``values`` are all the values the density of
    ``op``'s node is over -- something only a caller that collected them can claim. An
    implementation that recurses into a measurable input holds just its own value, and must ask
    through `request_logprob` instead.
    """
    raise NotImplementedError(f"Logprob method not implemented for {op}")


def request_logprob(rv, value, **kwargs):
    """Return the log-density term of `rv` at `value`, or a stand-in for it.

    This is how a `_logprob` implementation recurses into its measurable input: holding one
    variable's value, it cannot promise that no other value reaches that variable's node. A node
    with a single output a value could attach to cannot be asked for more, and is dispatched on
    the spot; one that could be asked for more gets a `DensityQuery` placeholder, and
    `resolve_density_queries` answers every request about the node in one `_logprob` call.

    A node's values are therefore requested one variable at a time, each naming the output it
    belongs to, which is what lets requests arriving by different routes be recognized as being
    about the same node.

    Of the three entry points, `logp` derives a variable's density from scratch and resolves
    the queries it raises; `request_logprob` recurses within a derivation already underway; and
    the `_logprob` dispatcher is called directly only with all of a node's values in hand.
    """
    if n_potential_valued_outputs(rv.owner) > 1:
        return density_query(rv, value)

    logprob = _logprob(rv.owner.op, (value,), *rv.owner.inputs, **kwargs)

    if name := (rv.name or value.name):
        if isinstance(logprob, list | tuple):
            for i, term in enumerate(logprob):
                term.name = f"{name}_logprob.{i}"
        else:
            logprob.name = f"{name}_logprob"

    return logprob


def _logprob_helper(rv, *values, **kwargs):
    warnings.warn(
        "_logprob_helper has been renamed to request_logprob",
        FutureWarning,
        stacklevel=2,
    )
    if len(values) != 1:
        # The old API let a caller hand over every value of a node at once
        return _logprob(rv.owner.op, values, *rv.owner.inputs, **kwargs)
    return request_logprob(rv, values[0], **kwargs)


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

    ``supp_axes`` is meta-information about that measure: for each output, which of its axes the
    measure is defined over, counted from the right so that they do not depend on how many batch
    dimensions the variable carries. It is set when the Op is built, by whichever rewrite made
    the variable measurable and therefore knows what the measure it wraps looks like. Derive it
    from the node's inputs, so that nodes which compare equal necessarily agree on it.

    There is one entry per output, in ``node.outputs`` order, so an output that carries no
    measure of its own -- an RNG update, say -- pads the tuple with ``None``. That is not the
    indexing of the ``values`` a ``_logprob`` implementation receives, which only ever covers
    the outputs a value can attach to.

    ``None`` -- the default, or a single entry of it -- means this Op was not told, which is not
    the same as having no support axes. Whoever asks decides what to do without an answer; a
    `DensityQuery` refuses, having no way to type the term it stands for. See #6360, which also
    wants ``ndim_supp`` and a discrete/continuous/mixed ``type`` carried the same way.
    """

    supp_axes: tuple[tuple[int, ...] | None, ...] | None = None


MeasurableOp.register(RandomVariable)


class MeasurableElemwise(MeasurableOp, Elemwise):
    """Base class for Measurable Elemwise variables."""

    valid_scalar_types: tuple[MetaType, ...] = ()

    def __init__(self, scalar_op, *args, **kwargs):
        if not isinstance(scalar_op, self.valid_scalar_types):
            raise TypeError(
                f"scalar_op {scalar_op} is not valid for class {self.__class__}. "
                f"Acceptable types are {self.valid_scalar_types}"
            )
        super().__init__(scalar_op, *args, **kwargs)

    def __str__(self):
        """Return a string representation of the object."""
        return f"Measurable{super().__str__()}"


class MeasurableBlockwise(MeasurableOp, Blockwise):
    """Base class for Measurable Blockwise variables."""


class ValuedRV(Op):
    r"""Represents the association of a measurable variable and its value.

    A `ValuedVariable` node represents the pair :math:`(Y, y)`, where  `y` the value at which :math:`Y`'s density
    or probability mass function is evaluated.

    The log-probability function takes such pairs as input, which makes these nodes in a graph an intermediate form
    that serves to construct a log-probability from a model graph.


    Notes
    -----
    The introduction of these operations achieves two goals:
    1. Identify the conditioning points between multiple, potentially interdependent measurable variables,
    and introduce the respective value variables in the IR graph.
    2. Prevent automatic rewrites across conditioning points

    About point 2. In the current framework, a RV logp cannot depend on a transformation of the value variable
    of a second RV it depends on. While this is mathematically trivial, we don't have the machinery to achieve it.

    The only case we do something like this is in the ad-hoc transform_value rewrite, but there we are
    told explicitly what value variables must be transformed before being used in the density of dependent RVs.

    For example ,the following is not supported:

    ```python
    x_log = pt.random.normal()
    x = pt.exp(x_log)
    y = pt.random.normal(loc=x_log)

    x_value = pt.scalar()
    y_value = pt.scalar()
    conditional_logprob({x: x_value, y: y_value})
    ```

    Our framework doesn't know that the density of y should depend on a (log) transform of x_value.

    Importantly, we need to prevent this limitation from being introduced automatically by our IR rewrites.
    For example given the following:

    ```python
    a_base = pm.Normal.dist()
    a = a_base * 5
    b = pm.Normal.dist(a * 8)

    a_value = scalar()
    b_value = scalar()
    conditional_logp({a: a_value, b: b_value})
    ```

    We do not want `b` to be rewritten as `pm.Normal.dist(a_base * 40)`, as it would then be disconnected from the
    valued `a` associated with `pm.Normal.dist(a_base * 5). By introducing `ValuedRV` nodes the graph looks like:

    ```python
    a_base = pm.Normal.dist()
    a = valued_rv(a_base * 5, a_value)
    b = valued_rv(a * 8, b_value)
    ```

    Since, PyTensor doesn't know what to do with `ValuedRV` nodes, there is no risk of rewriting across them
    and breaking the dependency of `b` on `a`. The new nodes isolate the graphs between conditioning points.
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


class PromisedValuedRV(Op):
    r"""Marks a variable as being promised a valued variable that will only be assigned by the logprob method.

    Some measurable RVs like Join/MakeVector can combine multiple, potentially interdependent, RVs into a single
    composite valued node. Only in the logp function is this value split and sent to each component,
    but we still want to achieve the same goals that ValuedRVs achieve during the IR rewrites.

    Here is an example analogous to the one described in the docstrings of ValuedRV:

    ```python
    a_base = pt.random.normal()
    a = a_base * 5
    b = pt.random.normal(a * 8)
    ab = pt.stack([a, b])
    ab_value = pt.vector(shape=(2,))

    logp(ab, ab_value)
    ```

    The density of `ab[1]` (that is `b`) depends on `ab_value[1]` and `ab_value[0] * 8`, but this is not apparent
    in the IR representation because the values of `a` and `b` are merged together, and will only be split by the logp
    function (see why next). For the time being we introduce a PromisedValue to isolate the graphs of a and b, and
    freezing the dependency of `b` on `a` (not `a_base`).

    Now why use a new Op and not just ValuedRV? Just for convenience! In the end we still want a function from
    `ab_value` to `stack([logp(a), logp(b | a)])`, and if we split the values ahead of time we wouldn't know how to
    stack them later (or even know that we were supposed to).

    One final point, while this achieves the same goal as introducing ValuedRVs, it already constitutes a form of inference
    (knowing how/when to measure Join/MakeVectors), so we have to do it as an IR rewrite. However, we have to do it
    before any other rewrites, so you'll see that the related rewrites are registered in `early_measurable_ir_rewrites_db`.

    """

    def make_node(self, rv):
        assert isinstance(rv, Variable)
        return Apply(self, [rv], [rv.type(name=rv.name)])

    def perform(self, node, inputs, out):
        raise NotImplementedError("PromisedValuedRV should not be present in the final graph!")

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]


promised_valued_rv = PromisedValuedRV()


def supp_axes(var: Variable) -> tuple[int, ...] | None:
    """The axes of `var` that a measure over it is defined over, if that is known.

    Read off the Op that produced it -- see `MeasurableOp.supp_axes`. A RandomVariable is the
    case everything else is built out of, and already carries the equivalent as `ndim_supp`.
    """
    node = var.owner
    # getattr: RandomVariable is a virtual subclass of MeasurableOp, so it has no default
    if (declared := getattr(node.op, "supp_axes", None)) is not None:
        return declared[node.outputs.index(var)]
    # RandomVariable and SymbolicRandomVariable say it as a count of rightmost axes; a
    # SymbolicRandomVariable without a signature says nothing at all.
    ndim_supp = getattr(node.op, "ndim_supp", None)
    return None if ndim_supp is None else tuple(range(-ndim_supp, 0))


class DensityQuery(Op):
    r"""A deferred request for the log-density of a measurable variable at a value.

    `request_logprob` normally recurses immediately, which bottoms out on one variable at a
    time. That cannot work when the target is one output of a node whose density is joint over
    several values: the density would be derived with the other values missing. Such a target
    gets a `DensityQuery` instead, and `resolve_density_queries` answers every query on the same
    node in a single `_logprob` call.

    Because the queries are clients of the node they target, collecting them is a plain lookup
    and answering them a plain multi-output replacement -- there is nothing upstream to rewire.

    The output has to stand in for the term before the term exists, and a density reduces the
    axes its variable is defined over, so the value's own type will not do: the term's type
    drops the axes `supp_axes` declares from the shape the variable and the value broadcast to.
    """

    def make_node(self, rv, value):
        assert isinstance(rv, Variable)
        assert isinstance(value, Variable)
        axes = supp_axes(rv)
        if axes is None:
            raise NotImplementedError(
                f"{rv.owner.op} has several outputs whose values reach its density by separate "
                f"paths, so it must declare which axes its measure is over. "
                f"Set `supp_axes` on the Op."
            )
        # A density is continuous, whatever the dtype of the variable it measures.
        dtype = value.type.dtype if value.type.dtype.startswith("float") else config.floatX
        # A term is evaluated at the value and taken over the variable, so it comes out on the
        # shape the two broadcast to.
        shape = [
            broadcast_static_dim_lengths(lengths)
            for lengths in zip_longest(
                reversed(rv.type.shape), reversed(value.type.shape), fillvalue=1
            )
        ][::-1]
        shape = tuple(
            length for axis, length in enumerate(shape, start=-len(shape)) if axis not in axes
        )
        return Apply(self, [rv, value], [tensor(dtype=dtype, shape=shape)])

    def perform(self, node, inputs, out):
        raise NotImplementedError("DensityQuery should not be present in the final graph!")


density_query = DensityQuery()


def n_potential_valued_outputs(node) -> int:
    """How many outputs of `node` a value could be associated with -- all but its RNG outputs.

    More than one means the node's density may be joint over them -- something only the op
    knows -- so every value that does arrive has to reach `_logprob` in the same call.
    """
    return sum(not isinstance(out.type, RandomType) for out in node.outputs)
