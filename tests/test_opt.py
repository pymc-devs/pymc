import aesara
import aesara.tensor as at
from aesara.graph.opt import in2out
from aesara.graph.opt_utils import optimize_graph
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.subtensor import Subtensor

from aeppl.dists import DiracDelta, dirac_delta
from aeppl.opt import local_lift_DiracDelta, naive_bcast_rv_lift


def test_naive_bcast_rv_lift():
    r"""Make sure `test_naive_bcast_rv_lift` can handle useless scalar `BroadcastTo`\s."""
    X_rv = at.random.normal()
    Z_at = at.broadcast_to(X_rv, ())

    # Make sure we're testing what we intend to test
    assert isinstance(Z_at.owner.op, BroadcastTo)

    res = optimize_graph(Z_at, custom_opt=in2out(naive_bcast_rv_lift), clone=False)
    assert res is X_rv


def test_local_lift_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    Z_at = at.cast(dd_at, "int64")

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Elemwise)

    Z_at = dd_at.dimshuffle("x", 0)

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, DimShuffle)

    Z_at = dd_at[0]

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Subtensor)

    # Don't lift multi-output `Op`s
    c_at = at.matrix()
    dd_at = dirac_delta(c_at)
    Z_at = at.nlinalg.svd(dd_at)[0]

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert res is Z_at


def test_local_remove_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    fn = aesara.function([c_at], dd_at)
    assert not any(
        isinstance(node.op, DiracDelta) for node in fn.maker.fgraph.toposort()
    )
