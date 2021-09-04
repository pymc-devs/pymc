from aesara.graph.basic import walk
from aesara.scan.op import Scan

from aeppl.abstract import MeasurableVariable


def assert_no_rvs(var):
    """Assert that there are no `MeasurableVariable` nodes in a graph."""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            # TODO: We need a better--potentially type-based--means of
            # determining whether or not an inner-graph is present
            if isinstance(owner.op, Scan):
                inputs += owner.op.outputs

            return inputs

    for v in walk([var], expand, False):
        if v.owner and isinstance(v.owner.op, MeasurableVariable):
            raise AssertionError(f"Variable {v} is a MeasurableVariable")
