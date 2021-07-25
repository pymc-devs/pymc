import abc
from functools import singledispatch

from aesara.tensor.random.op import RandomVariable


class MeasurableVariable(abc.ABC):
    """A variable that can be assigned a measure/log-probability"""


MeasurableVariable.register(RandomVariable)


@singledispatch
def get_measurable_outputs(op, node):
    """Return only the outputs that are measurable."""
    return node.outputs


@get_measurable_outputs.register(RandomVariable)
def randomvariable_outputs(op, node):
    return node.outputs[1:]
