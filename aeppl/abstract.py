import abc
from copy import copy
from functools import singledispatch
from typing import Callable, List, Tuple

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.graph.utils import MetaType
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.random.op import RandomVariable


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
    return None


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
    ==========
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

    if op_type in _get_measurable_outputs.registry.keys() and isinstance(
        op_type, UnmeasurableMeta
    ):
        if _get_measurable_outputs.registry[op_type] != measurable_outputs_fn:
            raise ValueError(
                f"The type {op_type.__name__} with hash value {hash(op_type)} "
                "has already been dispatched a measurable outputs function."
            )
        return node

    new_op_dict = op_type.__dict__.copy()
    new_op_dict["id_obj"] = (new_node.op, measurable_outputs_fn)

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
