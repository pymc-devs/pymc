import aesara.tensor as at
from aesara.graph.opt import in2out
from aesara.graph.opt_utils import optimize_graph
from aesara.tensor.extra_ops import BroadcastTo

from aeppl.opt import naive_bcast_rv_lift


def test_naive_bcast_rv_lift():
    r"""Make sure `test_naive_bcast_rv_lift` can handle useless scalar `BroadcastTo`\s."""
    X_rv = at.random.normal()
    Z_at = at.broadcast_to(X_rv, ())

    # Make sure we're testing what we intend to test
    assert isinstance(Z_at.owner.op, BroadcastTo)

    res = optimize_graph(Z_at, custom_opt=in2out(naive_bcast_rv_lift), clone=False)
    assert res is X_rv
