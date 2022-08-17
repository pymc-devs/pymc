import aesara
import aesara.tensor as at
from aesara.graph.rewriting.basic import in2out
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.subtensor import Subtensor

from aeppl.dists import DiracDelta, dirac_delta
from aeppl.rewriting import local_lift_DiracDelta


def test_local_lift_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    Z_at = at.cast(dd_at, "int64")

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Elemwise)

    Z_at = dd_at.dimshuffle("x", 0)

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, DimShuffle)

    Z_at = dd_at[0]

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Subtensor)

    # Don't lift multi-output `Op`s
    c_at = at.matrix()
    dd_at = dirac_delta(c_at)
    Z_at = at.nlinalg.svd(dd_at)[0]

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert res is Z_at


def test_local_remove_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    fn = aesara.function([c_at], dd_at)
    assert not any(
        isinstance(node.op, DiracDelta) for node in fn.maker.fgraph.toposort()
    )
