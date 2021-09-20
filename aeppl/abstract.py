import abc
from functools import singledispatch
from typing import List

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.tensor.random.op import RandomVariable


class MeasurableVariable(abc.ABC):
    """A variable that can be assigned a measure/log-probability"""


MeasurableVariable.register(RandomVariable)


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
