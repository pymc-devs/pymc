import warnings

import aesara.tensor as at
from aesara.graph.basic import Apply
from aesara.graph.op import Op

from aeppl.abstract import MeasurableVariable


class DiracDelta(Op):
    """An `Op` that represents a Dirac-delta distribution."""

    __props__ = ("rtol", "atol")

    def __init__(self, rtol=1e-5, atol=1e-8):
        self.rtol = rtol
        self.atol = atol

    def make_node(self, x):
        x = at.as_tensor(x)
        return Apply(self, [x], [x.type()])

    def do_constant_folding(self, fgraph, node):
        # Without this, the `Op` would be removed from the graph during
        # canonicalization
        return False

    def perform(self, node, inp, out):
        (x,) = inp
        (z,) = out
        warnings.warn(
            "DiracDelta is a dummy Op that shouldn't be used in a compiled graph"
        )
        z[0] = x

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes


dirac_delta = DiracDelta()

MeasurableVariable.register(DiracDelta)
