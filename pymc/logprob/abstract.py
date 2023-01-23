#   Copyright 2022- The PyMC Developers
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

from copy import copy
from functools import singledispatch
from typing import Callable, List, Sequence, Tuple

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.graph.utils import MetaType
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable


def logprob(rv_var, *rv_values, **kwargs):
    """Create a graph for the log-probability of a ``RandomVariable``."""
    logprob = _logprob(rv_var.owner.op, rv_values, *rv_var.owner.inputs, **kwargs)

    for rv_var in rv_values:
        if rv_var.name:
            logprob.name = f"{rv_var.name}_logprob"

    return logprob


def logcdf(rv_var, rv_value, **kwargs):
    """Create a graph for the logcdf of a ``RandomVariable``."""
    logcdf = _logcdf(rv_var.owner.op, rv_value, *rv_var.owner.inputs, name=rv_var.name, **kwargs)

    if rv_var.name:
        logcdf.name = f"{rv_var.name}_logcdf"

    return logcdf


def icdf(rv, value, **kwargs):
    """Create a graph for the inverse CDF of a `RandomVariable`."""
    rv_icdf = _icdf(rv.owner.op, value, *rv.owner.inputs, **kwargs)
    if rv.name:
        rv_icdf.name = f"{rv.name}_icdf"
    return rv_icdf


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

    """
    raise NotImplementedError(f"Logprob method not implemented for {op}")


@singledispatch
def _logcdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the logcdf of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new logcdf graphs
    for a ``RandomVariable``, register a new function on this dispatcher.
    """
    raise NotImplementedError(f"Logcdf method not implemented for {op}")


@singledispatch
def _icdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the inverse CDF of a `RandomVariable`.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.
    """
    raise NotImplementedError(f"icdf not implemented for {op}")


class MeasurableVariable(abc.ABC):
    """A variable that can be assigned a measure/log-probability"""


MeasurableVariable.register(RandomVariable)


class UnmeasurableMeta(MetaType):
    def __new__(cls, name, bases, dict):
        if "id_obj" not in dict:
            dict["id_obj"] = None

        return super().__new__(cls, name, bases, dict)

    def __eq__(self, other):
        if isinstance(other, UnmeasurableMeta):
            return hash(self.id_obj) == hash(other.id_obj)
        return False

    def __hash__(self):
        return hash(self.id_obj)


class UnmeasurableVariable(metaclass=UnmeasurableMeta):
    """
    id_obj is an attribute, i.e. tuple of length two, of the unmeasurable class object.
    e.g. id_obj = (NormalRV, noop_measurable_outputs_fn)
    """


def get_measurable_outputs(op: Op, node: Apply) -> List[Variable]:
    """Return only the outputs that are measurable."""
    if isinstance(op, MeasurableVariable):
        return _get_measurable_outputs(op, node)
    else:
        return []


@singledispatch
def _get_measurable_outputs(op, node):
    return node.outputs


@_get_measurable_outputs.register(RandomVariable)
def _get_measurable_outputs_RandomVariable(op, node):
    return node.outputs[1:]


def noop_measurable_outputs_fn(*args, **kwargs):
    return []


def assign_custom_measurable_outputs(
    node: Apply,
    measurable_outputs_fn: Callable = noop_measurable_outputs_fn,
    type_prefix: str = "Unmeasurable",
) -> Apply:
    """Assign a custom ``_get_measurable_outputs`` dispatch function to a measurable variable instance.

    The node is cloned and a custom `Op` that's a copy of the original node's
    `Op` is created.  That custom `Op` replaces the old `Op` in the cloned
    node, and then a custom dispatch implementation is created for the clone
    `Op` in `_get_measurable_outputs`.

    If `measurable_outputs_fn` isn't specified, a no-op is used; the result is
    a clone of `node` that will effectively be ignored by
    `factorized_joint_logprob`.

    Parameters
    ----------
    node
        The node to recreate with a new cloned `Op`.
    measurable_outputs_fn
        The function that will be assigned to the new cloned `Op` in the
        `_get_measurable_outputs` dispatcher.
        The default is a no-op function (i.e. no measurable outputs)
    type_prefix
        The prefix used for the new type's name.
        The default is ``"Unmeasurable"``, which matches the default
        ``"measurable_outputs_fn"``.
    """

    new_node = node.clone()
    op_type = type(new_node.op)

    if op_type in _get_measurable_outputs.registry.keys() and isinstance(op_type, UnmeasurableMeta):
        if _get_measurable_outputs.registry[op_type] != measurable_outputs_fn:
            raise ValueError(
                f"The type {op_type.__name__} with hash value {hash(op_type)} "
                "has already been dispatched a measurable outputs function."
            )
        return node

    new_op_dict = op_type.__dict__.copy()
    new_op_dict["id_obj"] = (new_node.op, measurable_outputs_fn)
    new_op_dict.setdefault("original_op_type", op_type)

    new_op_type = type(
        f"{type_prefix}{op_type.__name__}", (op_type, UnmeasurableVariable), new_op_dict
    )
    new_node.op = copy(new_node.op)
    new_node.op.__class__ = new_op_type

    _get_measurable_outputs.register(new_op_type)(measurable_outputs_fn)

    return new_node


class MeasurableElemwise(Elemwise):
    """Base class for Measurable Elemwise variables"""

    valid_scalar_types: Tuple[MetaType, ...] = ()

    def __init__(self, scalar_op, *args, **kwargs):
        if not isinstance(scalar_op, self.valid_scalar_types):
            raise TypeError(
                f"scalar_op {scalar_op} is not valid for class {self.__class__}. "
                f"Acceptable types are {self.valid_scalar_types}"
            )
        super().__init__(scalar_op, *args, **kwargs)


MeasurableVariable.register(MeasurableElemwise)
