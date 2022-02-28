from aesara.graph.basic import walk
from aesara.graph.op import HasInnerGraph

from aeppl.abstract import MeasurableVariable


def assert_no_rvs(var):
    """Assert that there are no `MeasurableVariable` nodes in a graph."""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs

            return inputs

    for v in walk([var], expand, False):
        if v.owner and isinstance(v.owner.op, MeasurableVariable):
            raise AssertionError(f"Variable {v} is a MeasurableVariable")
